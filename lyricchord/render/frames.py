"""
Frame composition with Pillow.

A `FrameComposer` pre-renders everything static (background, panels, title) once,
then `render(t)` draws only the time-dependent parts: lyric lines, current/next
chord, the scrolling chord lane and the progress bar. Glyph rasterisation is the
slow part of Pillow drawing, so rendered text images are cached by
(text, font, colour) and pasted per frame.

Layout (16:9 reference, scaled to any resolution):

    +----------------------------------------------------------+
    |                Title                      Key - BPM      |
    |                artist                                    |
    |  +----------------------------------------------------+  |
    |  |              previous line (dim)                   |  |
    |  |         >> CURRENT LINE (bright, filled) <<        |  |
    |  |              next line (dim)                       |  |
    |  +----------------------------------------------------+  |
    |  +--------+ +------+ +-----------------------------+     |
    |  |  NOW   | | NEXT | | |Am  |F     |C   |G     ...  |    |
    |  |   Am   | |  F   | | (scrolling chord lane)       |    |
    |  +--------+ +------+ +-----------------------------+     |
    |  ======== progress =========================== 1:23/3:45 |
    +----------------------------------------------------------+
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from ..models import ChordEvent, ChordTrack, LyricLine, Lyrics, SongData, SongInfo
from ..utils.fonts import guess_bold_variant, load_font
from ..utils.text import format_time

RGB = Tuple[int, int, int]
Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- colours
def hex_to_rgb(value: str, default: RGB = (255, 255, 255)) -> RGB:
    v = (value or "").strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    try:
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except (ValueError, IndexError):
        return default


def mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))  # type: ignore[return-value]


def make_gradient(w: int, h: int, c0: RGB, c1: RGB, direction: str) -> Image.Image:
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    if direction == "horizontal":
        t = np.broadcast_to(xs, (h, w))
    elif direction == "vertical":
        t = np.broadcast_to(ys, (h, w))
    else:  # diagonal
        t = (xs + ys) / 2.0
    arr = (1 - t)[..., None] * np.array(c0, np.float32) + t[..., None] * np.array(c1, np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- layout
@dataclass
class Layout:
    W: int
    H: int
    s: float                 # scale factor relative to a 1080-pixel short side
    margin: int
    portrait: bool
    title_y: int
    artist_y: int
    badge_xy: Tuple[int, int]
    badge_align: str         # "right" | "center"
    lyric_box: Box
    chord_box: Box
    now_box: Box
    next_box: Box
    lane_box: Box
    progress_y: int


def compute_layout(W: int, H: int) -> Layout:
    s = min(W, H) / 1080.0
    portrait = H > W
    margin = int(W * 0.0625)
    pad = int(20 * s)
    if not portrait:
        chord_box = (margin, int(H * 0.665), W - margin, int(H * 0.925))
        cb = chord_box
        inner_h = cb[3] - cb[1] - 2 * pad
        now_w, next_w = int(W * 0.19), int(W * 0.13)
        now_box = (cb[0] + pad, cb[1] + pad, cb[0] + pad + now_w, cb[3] - pad)
        next_box = (now_box[2] + pad, cb[1] + pad + int(inner_h * 0.15),
                    now_box[2] + pad + next_w, cb[3] - pad - int(inner_h * 0.15))
        lane_box = (next_box[2] + 2 * pad, cb[1] + pad + int(inner_h * 0.2),
                    cb[2] - pad, cb[3] - pad - int(inner_h * 0.2))
        return Layout(W, H, s, margin, False, int(H * 0.06), int(H * 0.125),
                      (W - margin, int(H * 0.06)), "right",
                      (margin, int(H * 0.185), W - margin, int(H * 0.64)),
                      chord_box, now_box, next_box, lane_box, int(H * 0.958))
    chord_box = (margin, int(H * 0.625), W - margin, int(H * 0.93))
    cb = chord_box
    inner_w, inner_h = cb[2] - cb[0] - 2 * pad, cb[3] - cb[1] - 2 * pad
    row1 = int(inner_h * 0.6)
    now_box = (cb[0] + pad, cb[1] + pad, cb[0] + pad + int(inner_w * 0.58), cb[1] + pad + row1)
    next_box = (now_box[2] + pad, cb[1] + pad + int(row1 * 0.12), cb[2] - pad,
                cb[1] + pad + row1 - int(row1 * 0.12))
    lane_box = (cb[0] + pad, now_box[3] + pad, cb[2] - pad, cb[3] - pad)
    return Layout(W, H, s, margin, True, int(H * 0.045), int(H * 0.085),
                  (W // 2, int(H * 0.118)), "center",
                  (margin, int(H * 0.15), W - margin, int(H * 0.6)),
                  chord_box, now_box, next_box, lane_box, int(H * 0.96))


# --------------------------------------------------------------------------- composer
class FrameComposer:
    """Draws frames for one song. Call `render(t)` for raw RGB/RGBA bytes."""

    def __init__(self, song: SongData, settings: Settings, width: int, height: int,
                 transparent: bool = False):
        self.song = song
        self.st = settings
        self.W, self.H = width, height
        self.transparent = transparent
        self.mode = "RGBA" if transparent else "RGB"
        self.L = compute_layout(width, height)
        s = self.L.s

        reg = settings.font_path
        bold = settings.bold_font_path or guess_bold_variant(reg)
        self.f_title = load_font(bold, max(8, int(settings.title_size * s)), bold=True)
        self.f_artist = load_font(reg, max(8, int(settings.title_size * 0.62 * s)))
        self.f_lyric = load_font(bold, max(8, int(settings.lyric_size * s)), bold=True)
        self.f_context = load_font(reg, max(8, int(settings.context_lyric_size * s)))
        self.f_chord = load_font(bold, max(8, int(settings.chord_size * s)), bold=True)
        self.f_chord_next = load_font(bold, max(8, int(settings.chord_size * 0.48 * s)), bold=True)
        self.f_lane = load_font(bold, max(8, int(38 * s)), bold=True)
        self.f_small = load_font(reg, max(8, int(24 * s)))
        self.f_label = load_font(bold, max(8, int(20 * s)), bold=True)
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._bold_path = bold

        self.accent = hex_to_rgb(settings.accent_color, (56, 189, 248))
        self.text = hex_to_rgb(settings.text_color, (248, 250, 252))
        self.dim = hex_to_rgb(settings.dim_text_color, (148, 163, 184))
        self.panel = hex_to_rgb(settings.panel_color, (11, 18, 32))
        self.block = mix(self.panel, self.text, 0.22)       # lane block for upcoming chords
        self.box_fill = mix(self.panel, self.text, 0.10)    # NOW/NEXT box fill
        self.stroke = max(1, int(2 * s)) if transparent else 0

        self.lines: List[LyricLine] = list(song.lyrics.lines)
        self.line_starts = [l.start for l in self.lines]
        self.events: List[ChordEvent] = list(song.chords.events)
        self.event_starts = [e.start for e in self.events]

        self._text_cache: Dict[tuple, Tuple[Image.Image, int, int]] = {}
        self.static = self._build_static()

    # ------------------------------------------------------------------ text helpers
    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        size = max(8, int(size))
        if size not in self._font_cache:
            self._font_cache[size] = load_font(self._bold_path, size, bold=True)
        return self._font_cache[size]

    def _text_image(self, text: str, font: ImageFont.FreeTypeFont, color: RGB) -> Tuple[Image.Image, int, int]:
        """Rasterise text once; returns (image, left_offset, top_offset)."""
        key = (text, id(font), color)
        hit = self._text_cache.get(key)
        if hit is not None:
            return hit
        stroke = self.stroke
        l, t, r, b = font.getbbox(text, stroke_width=stroke)
        w, h = max(1, r - l + 2), max(1, b - t + 2)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((-l + 1, -t + 1), text, font=font, fill=color + (255,), stroke_width=stroke,
               stroke_fill=(0, 0, 0, 230) if stroke else None)
        if len(self._text_cache) > 3000:
            self._text_cache.clear()
        self._text_cache[key] = (img, l - 1, t - 1)
        return img, l - 1, t - 1

    def _blit(self, frame: Image.Image, text: str, font: ImageFont.FreeTypeFont, color: RGB,
              x: float, y: float, align: str = "left", fill_fraction: Optional[float] = None,
              fill_color: Optional[RGB] = None) -> int:
        """Draw text with its em-box top-left at (x, y). Returns the text width in px."""
        if not text:
            return 0
        width = font.getlength(text)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        img, l, t = self._text_image(text, font, color)
        pos = (int(round(x + l)), int(round(y + t)))
        frame.paste(img, pos, img)
        if fill_fraction and fill_color:
            fill_img, _, _ = self._text_image(text, font, fill_color)
            cut = int(fill_img.width * min(1.0, max(0.0, fill_fraction)))
            if cut > 0:
                part = fill_img.crop((0, 0, cut, fill_img.height))
                frame.paste(part, pos, part)
        return int(width)

    def _line_height(self, font: ImageFont.FreeTypeFont) -> int:
        ascent, descent = font.getmetrics()
        return ascent + descent

    def _wrap(self, text: str, font: ImageFont.FreeTypeFont, max_w: int, max_rows: int = 3) -> List[str]:
        words = text.split()
        rows: List[str] = []
        cur = ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if font.getlength(cand) <= max_w or not cur:
                cur = cand
            else:
                rows.append(cur)
                cur = w
        if cur:
            rows.append(cur)
        if len(rows) > max_rows:
            rows = rows[:max_rows]
            rows[-1] = self._ellipsize(rows[-1] + " ...", font, max_w)
        return rows

    def _ellipsize(self, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
        if font.getlength(text) <= max_w:
            return text
        while text and font.getlength(text + "...") > max_w:
            text = text[:-1]
        return text.rstrip() + "..."

    # ------------------------------------------------------------------ static layer
    def _build_static(self) -> Image.Image:
        st, L = self.st, self.L
        W, H = self.W, self.H
        if self.transparent:
            base = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * min(1.0, max(0.0, st.loop_dim)))))
        elif st.background_style == "solid":
            base = Image.new("RGBA", (W, H), hex_to_rgb(st.bg_color, (15, 23, 42)) + (255,))
        else:
            base = make_gradient(W, H, hex_to_rgb(st.gradient_start, (15, 23, 42)),
                                 hex_to_rgb(st.gradient_end, (30, 58, 138)), st.gradient_direction).convert("RGBA")

        # Translucent panels.
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        alpha = int(min(255, max(0, st.panel_alpha)))
        radius = int(24 * L.s)
        od.rounded_rectangle(L.lyric_box, radius=radius, fill=self.panel + (alpha,))
        od.rounded_rectangle(L.chord_box, radius=radius, fill=self.panel + (alpha,))
        base = Image.alpha_composite(base, overlay)

        d = ImageDraw.Draw(base)
        inner_alpha = 255 if not self.transparent else max(alpha, 200)
        d.rounded_rectangle(L.now_box, radius=int(18 * L.s), fill=self.box_fill + (inner_alpha,))
        d.rounded_rectangle(L.next_box, radius=int(16 * L.s), fill=self.box_fill + (inner_alpha,))
        if st.show_chord_timeline:
            d.rounded_rectangle(L.lane_box, radius=int(12 * L.s), fill=self.box_fill + (inner_alpha,))
            x0, y0, x1, y1 = L.lane_box
            d.rectangle((x0, y0 - int(6 * L.s), x0 + max(2, int(4 * L.s)), y1 + int(6 * L.s)), fill=self.accent + (255,))

        # Box captions.
        self._blit(base, "NOW", self.f_label, self.dim, L.now_box[0] + int(16 * L.s), L.now_box[1] + int(10 * L.s))
        self._blit(base, "NEXT", self.f_label, self.dim, L.next_box[0] + int(14 * L.s), L.next_box[1] + int(8 * L.s))

        # Header.
        info = self.song.info
        title = self._ellipsize(info.display_title, self.f_title, W - 2 * L.margin - int(300 * L.s) * (not L.portrait))
        self._blit(base, title, self.f_title, self.text, W / 2, L.title_y - self._line_height(self.f_title) / 2, "center")
        self._blit(base, info.display_artist, self.f_artist, self.dim, W / 2,
                   L.artist_y - self._line_height(self.f_artist) / 2, "center")
        if st.show_key_bpm:
            parts = []
            if self.song.chords.key:
                parts.append(f"Key: {self.song.chords.key}")
            if self.song.chords.bpm:
                parts.append(f"{int(round(self.song.chords.bpm))} BPM")
            if parts:
                badge = "   ·   ".join(parts)
                bx, by = L.badge_xy
                self._blit(base, badge, self.f_small, self.accent, bx,
                           by - self._line_height(self.f_small) / 2, L.badge_align)

        # Static hints for missing data.
        lx0, ly0, lx1, ly1 = L.lyric_box
        if not self.lines:
            msg = "Instrumental" if self.song.lyrics.synced else "Lyrics not found"
            self._blit(base, msg, self.f_context, self.dim, (lx0 + lx1) / 2,
                       (ly0 + ly1) / 2 - self._line_height(self.f_context) / 2, "center")
        elif not self.song.lyrics.synced:
            self._blit(base, "Lyric timing is approximate (no synced lyrics found)", self.f_small, self.dim,
                       lx0 + int(20 * L.s), ly1 - self._line_height(self.f_small) - int(10 * L.s))
        if not self.events:
            cx0, cy0, cx1, cy1 = L.lane_box if st.show_chord_timeline else L.chord_box
            self._blit(base, "No chords detected", self.f_context, self.dim, (cx0 + cx1) / 2,
                       (cy0 + cy1) / 2 - self._line_height(self.f_context) / 2, "center")
        return base if self.transparent else base.convert("RGB")

    # ------------------------------------------------------------------ dynamic layers
    def render_image(self, t: float) -> Image.Image:
        frame = self.static.copy()
        draw = ImageDraw.Draw(frame)
        self._draw_lyrics(frame, t)
        self._draw_chords(frame, draw, t)
        if self.st.show_progress_bar:
            self._draw_progress(frame, draw, t)
        return frame

    def render(self, t: float) -> bytes:
        return self.render_image(t).tobytes()

    def _draw_lyrics(self, frame: Image.Image, t: float) -> None:
        lines = self.lines
        if not lines:
            return
        L = self.L
        x0, y0, x1, y1 = L.lyric_box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        max_w = (x1 - x0) - int(60 * L.s)

        idx = bisect.bisect_right(self.line_starts, t) - 1
        active: Optional[LyricLine] = None
        if idx >= 0 and t < lines[idx].end and lines[idx].text.strip():
            active = lines[idx]
        # Lines around the centre line (skip LRC gap markers with empty text).
        center = active
        center_pos = idx
        if center is None:
            for j in range(idx + 1, len(lines)):
                if lines[j].text.strip():
                    center, center_pos = lines[j], j
                    break
        # After the last line has ended there is no centre line; show the last sung line as context.
        prev_end = center_pos - 1 if center is not None else idx
        prev_line = next((lines[j] for j in range(prev_end, -1, -1) if lines[j].text.strip()), None)
        after = [lines[j] for j in range(center_pos + 1, len(lines)) if lines[j].text.strip()][:2]

        # Current line block.
        lh = self._line_height(self.f_lyric)
        clh = self._line_height(self.f_context)
        gap = int(18 * L.s)
        rows = self._wrap(center.text, self.f_lyric, max_w) if center else []
        block_h = len(rows) * lh
        y = cy - block_h / 2
        if center is not None:
            color = self.text if active else self.dim
            progress = center.progress_at(t) if (active and self.st.karaoke_fill) else 0.0
            total_chars = max(1, sum(len(r) for r in rows))
            done_chars = progress * total_chars
            consumed = 0
            for row in rows:
                row_fill = None
                if progress > 0:
                    row_fill = max(0.0, min(1.0, (done_chars - consumed) / max(1, len(row))))
                consumed += len(row)
                self._blit(frame, row, self.f_lyric, color, cx, y, "center",
                           fill_fraction=row_fill, fill_color=self.accent)
                y += lh

        # Context: previous line above, upcoming lines below.
        if prev_line is not None:
            py = cy - block_h / 2 - gap - clh
            if py > y0 + int(12 * L.s):
                self._blit(frame, self._ellipsize(prev_line.text, self.f_context, max_w), self.f_context,
                           mix(self.dim, self.panel, 0.35), cx, py, "center")
        ny = cy + block_h / 2 + gap
        for ln in after:
            if ny + clh > y1 - int(12 * L.s):
                break
            self._blit(frame, self._ellipsize(ln.text, self.f_context, max_w), self.f_context, self.dim, cx, ny, "center")
            ny += clh + int(6 * L.s)

    @staticmethod
    def display_label(label: str) -> str:
        return "N.C." if label == "N" else label

    def _fit_font(self, text: str, base: ImageFont.FreeTypeFont, max_w: int) -> ImageFont.FreeTypeFont:
        font = base
        size = base.size
        while font.getlength(text) > max_w and size > 12:
            size = int(size * 0.85)
            font = self._font(size)
        return font

    def _draw_chords(self, frame: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
        events = self.events
        if not events:
            return
        L, st = self.L, self.st
        i = bisect.bisect_right(self.event_starts, t) - 1
        cur = events[i] if (i >= 0 and t < events[i].end) else None
        nxt: Optional[ChordEvent] = None
        for j in range(max(i + 1, 0), len(events)):
            if events[j].start > t and (cur is None or events[j].label != cur.label):
                nxt = events[j]
                break

        # NOW box
        x0, y0, x1, y1 = L.now_box
        label = self.display_label(cur.label) if cur else "—"
        font = self._fit_font(label, self.f_chord, (x1 - x0) - int(30 * L.s))
        img, _, _ = self._text_image(label, font, self.accent if cur else self.dim)
        cap = int(30 * L.s)
        frame.paste(img, (int((x0 + x1) / 2 - img.width / 2), int((y0 + cap + y1) / 2 - img.height / 2)), img)

        # NEXT box
        x0, y0, x1, y1 = L.next_box
        if nxt is not None:
            label = self.display_label(nxt.label)
            font = self._fit_font(label, self.f_chord_next, (x1 - x0) - int(24 * L.s))
            img, _, _ = self._text_image(label, font, self.text)
            cap = int(26 * L.s)
            frame.paste(img, (int((x0 + x1) / 2 - img.width / 2), int((y0 + cap + y1) / 2 - img.height / 2 - int(10 * L.s))), img)
            eta = f"in {max(0.0, nxt.start - t):.1f}s"
            self._blit(frame, eta, self.f_small, self.dim, (x0 + x1) / 2,
                       y1 - self._line_height(self.f_small) - int(8 * L.s), "center")

        # Scrolling lane
        if not st.show_chord_timeline:
            return
        lx0, ly0, lx1, ly1 = L.lane_box
        window = max(2.0, float(st.timeline_window_sec))
        pps = (lx1 - lx0) / window
        inset = int(6 * L.s)
        first = max(0, i)
        for ev in events[first:]:
            if ev.start >= t + window:
                break
            if ev.end <= t:
                continue
            bx0 = lx0 + max(0.0, ev.start - t) * pps
            bx1 = lx0 + min(window, ev.end - t) * pps
            if bx1 - bx0 < 2:
                continue
            is_cur = ev is cur
            fill = self.accent if is_cur else self.block
            draw.rounded_rectangle((int(bx0) + 1, ly0 + inset, int(bx1) - 1, ly1 - inset),
                                   radius=int(10 * L.s), fill=fill)
            text_color = self.panel if is_cur else self.text
            lbl = self.display_label(ev.label)
            font = self.f_lane
            if font.getlength(lbl) + int(16 * L.s) <= (bx1 - bx0):
                img, _, _ = self._text_image(lbl, font, text_color)
                frame.paste(img, (int((bx0 + bx1) / 2 - img.width / 2), int((ly0 + ly1) / 2 - img.height / 2)), img)

    def _draw_progress(self, frame: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
        L = self.L
        duration = max(1e-6, self.song.info.duration)
        x0, x1 = L.margin, self.W - L.margin
        h = max(3, int(6 * L.s))
        y = L.progress_y
        draw.rounded_rectangle((x0, y, x1, y + h), radius=h // 2, fill=self.block)
        fx = x0 + int((x1 - x0) * min(1.0, t / duration))
        if fx > x0 + h:
            draw.rounded_rectangle((x0, y, fx, y + h), radius=h // 2, fill=self.accent)
        ty = y + h + int(8 * L.s)
        self._blit(frame, format_time(t), self.f_small, self.dim, x0, ty)
        self._blit(frame, format_time(duration), self.f_small, self.dim, x1, ty, "right")


# --------------------------------------------------------------------------- preview
def demo_song(duration: float = 214.0) -> SongData:
    """Fake song used by the GUI's 'Preview layout' button."""
    info = SongInfo(Path("preview.mp3"), "Preview Song", "The Demo Band", duration=duration)
    texts = [
        "This is how the previous line looks",
        "And this line is being sung right now",
        "The next line waits down here",
        "Followed by one more for context",
        "Keep on strumming along",
    ]
    lines = [LyricLine(28.0 + 4 * k, 32.0 + 4 * k, txt) for k, txt in enumerate(texts)]
    chords = [ChordEvent(0, 30, "C"), ChordEvent(30, 34, "G"), ChordEvent(34, 38, "Am"),
              ChordEvent(38, 42, "F"), ChordEvent(42, 46, "C"), ChordEvent(46, 50, "G")]
    return SongData(info, Lyrics(lines, True, "demo"), ChordTrack(chords, "C major", 120.0, "demo"))


def render_preview(settings: Settings, t: float = 34.5, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Render one frame of the demo song with the given settings (for the GUI preview)."""
    W, H = size or settings.size
    transparent = settings.background_style == "loop"
    composer = FrameComposer(demo_song(), settings, W, H, transparent=transparent)
    img = composer.render_image(t)
    if transparent:
        # Composite over a neutral grey so the preview resembles video underneath.
        bg = make_gradient(W, H, (70, 70, 80), (30, 30, 40), "diagonal").convert("RGBA")
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")
