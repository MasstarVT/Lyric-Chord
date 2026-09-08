"""
Lyrics retrieval and LRC parsing.

Provider order:
  1. Sidecar file next to the audio: "<name>.lrc" (synced) or "<name>.txt" (plain)
  2. lrclib.net  - free, no API key, returns synced + plain lyrics
  3. syncedlyrics - aggregates Musixmatch / NetEase / Megalobiz / lrclib
Plain-text lyrics (no timestamps) are spread evenly across the song so the video
still shows something sensible; the `synced` flag lets the renderer say so.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from ..config import Settings
from ..models import Lyrics, LyricLine, LyricWord, SongInfo

log = logging.getLogger("lyricchord")

LRC_TAG = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
WORD_TAG = re.compile(r"<(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?>")
META_TAG = re.compile(r"^\[[a-zA-Z]+:[^\]]*\]$")

MAX_LINE_HOLD = 10.0   # seconds a lyric line stays on screen without a following line
HTTP_HEADERS = {"User-Agent": "LyricChord/1.0 (play-along video generator)"}
LRCLIB_BASE = "https://lrclib.net/api"


# --------------------------------------------------------------------------- parsing
def _tag_seconds(m: "re.Match[str]") -> float:
    mins, secs, frac = m.group(1), m.group(2), m.group(3) or "0"
    return int(mins) * 60 + int(secs) + int(frac) / (10 ** len(frac))


def _parse_words(body: str, line_start: float) -> Tuple[str, Optional[List[LyricWord]]]:
    """Parse enhanced-LRC word tags '<mm:ss.xx>word' into (plain_text, words)."""
    if not WORD_TAG.search(body):
        return body.strip(), None
    words: List[LyricWord] = []
    pos = 0
    pending_time = line_start
    for m in WORD_TAG.finditer(body):
        chunk = body[pos:m.start()].strip()
        if chunk:
            words.append(LyricWord(pending_time, chunk))
        pending_time = _tag_seconds(m)
        pos = m.end()
    tail = body[pos:].strip()
    if tail:
        words.append(LyricWord(pending_time, tail))
    text = " ".join(w.text for w in words)
    return text, (words or None)


def parse_lrc(text: str, duration: float = 0.0) -> List[LyricLine]:
    """Parse LRC text into timed lines. Supports multiple tags per line and word tags."""
    entries: List[Tuple[float, str, Optional[List[LyricWord]]]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or META_TAG.match(raw):
            continue
        # Consume every leading timestamp tag.
        tags = []
        pos = 0
        while True:
            m = LRC_TAG.match(raw, pos)
            if not m:
                break
            tags.append(m)
            pos = m.end()
        if not tags:
            continue
        body = raw[pos:]
        for tag in tags:
            start = _tag_seconds(tag)
            line_text, words = _parse_words(body, start)
            # Word timings are only meaningful for a single-timestamp line.
            entries.append((start, line_text, words if len(tags) == 1 else None))

    entries.sort(key=lambda e: e[0])
    lines: List[LyricLine] = []
    for i, (start, line_text, words) in enumerate(entries):
        if i + 1 < len(entries):
            end = entries[i + 1][0]
        else:
            end = duration if duration > start else start + MAX_LINE_HOLD
        if line_text:
            end = min(end, start + MAX_LINE_HOLD)
        lines.append(LyricLine(start=start, end=max(start, end), text=line_text, words=words))
    return lines


def plain_to_lines(text: str, duration: float) -> List[LyricLine]:
    """Spread untimed lyric lines evenly between an assumed intro and outro."""
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if not rows or duration <= 0:
        return []
    start, end = duration * 0.08, duration * 0.94
    step = (end - start) / len(rows)
    return [LyricLine(start + i * step, start + (i + 1) * step, row) for i, row in enumerate(rows)]


def apply_offset(lyrics: Lyrics, offset_ms: int) -> Lyrics:
    """Shift every timestamp by offset_ms (positive = later)."""
    if not offset_ms:
        return lyrics
    d = offset_ms / 1000.0
    for line in lyrics.lines:
        line.start = max(0.0, line.start + d)
        line.end = max(line.start, line.end + d)
        if line.words:
            for w in line.words:
                w.time = max(0.0, w.time + d)
    return lyrics


# --------------------------------------------------------------------------- providers
def _sidecar(info: SongInfo) -> Optional[Tuple[str, bool, str]]:
    """Return (text, synced, source) from a .lrc/.txt next to the audio file."""
    lrc = info.path.with_suffix(".lrc")
    if lrc.exists():
        return lrc.read_text(encoding="utf-8", errors="ignore"), True, "sidecar"
    txt = info.path.with_suffix(".txt")
    if txt.exists():
        from .chords.sheet import looks_like_chord_sheet

        content = txt.read_text(encoding="utf-8", errors="ignore")
        if not looks_like_chord_sheet(content):
            synced = bool(LRC_TAG.search(content))
            return content, synced, "sidecar"
    return None


def _pick_lrclib(data: dict) -> Optional[Tuple[str, bool]]:
    if data.get("instrumental"):
        return "", True
    if data.get("syncedLyrics"):
        return data["syncedLyrics"], True
    if data.get("plainLyrics"):
        return data["plainLyrics"], False
    return None


def fetch_lrclib(info: SongInfo) -> Optional[Tuple[str, bool]]:
    """Query lrclib.net. Exact match first (artist+title+duration), then search."""
    if not info.title:
        return None
    try:
        params = {"artist_name": info.artist, "track_name": info.title}
        if info.duration > 0:
            params["duration"] = int(round(info.duration))
        r = requests.get(f"{LRCLIB_BASE}/get", params=params, headers=HTTP_HEADERS, timeout=15)
        if r.status_code == 200:
            hit = _pick_lrclib(r.json())
            if hit:
                return hit

        r = requests.get(f"{LRCLIB_BASE}/search", params={"track_name": info.title,
                         "artist_name": info.artist}, headers=HTTP_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        results = r.json() or []
        if not results and info.artist:
            r = requests.get(f"{LRCLIB_BASE}/search", params={"q": f"{info.artist} {info.title}"},
                             headers=HTTP_HEADERS, timeout=15)
            results = r.json() if r.status_code == 200 else []

        def score(item: dict) -> float:
            s = 2.0 if item.get("syncedLyrics") else (1.0 if item.get("plainLyrics") else 0.0)
            if info.duration and item.get("duration"):
                s -= min(1.0, abs(float(item["duration"]) - info.duration) / 30.0)
            return s

        for item in sorted(results, key=score, reverse=True):
            hit = _pick_lrclib(item)
            if hit:
                return hit
    except (requests.RequestException, ValueError) as exc:
        log.warning("lrclib request failed: %s", exc)
    return None


def fetch_syncedlyrics(info: SongInfo) -> Optional[Tuple[str, bool]]:
    """Fallback aggregator across several lyric providers."""
    try:
        import syncedlyrics  # type: ignore
    except ImportError:
        return None
    term = f"{info.artist} {info.title}".strip()
    try:
        try:
            lrc = syncedlyrics.search(term, allow_plain_format=True)
        except TypeError:  # older API
            lrc = syncedlyrics.search(term)
    except Exception as exc:
        log.warning("syncedlyrics failed: %s", exc)
        return None
    if not lrc:
        return None
    return lrc, bool(LRC_TAG.search(lrc))


# --------------------------------------------------------------------------- entry point
def fetch_lyrics(info: SongInfo, settings: Settings) -> Lyrics:
    """Return the best available Lyrics for a song (possibly empty)."""
    attempts = [("sidecar", lambda: _sidecar(info))]
    attempts.append(("lrclib", lambda: fetch_lrclib(info)))
    attempts.append(("syncedlyrics", lambda: fetch_syncedlyrics(info)))

    instrumental_source = ""
    for name, fn in attempts:
        result = fn()
        if not result:
            continue
        text, synced = result[0], result[1]
        source = result[2] if len(result) > 2 else name
        if synced and not text.strip():
            # Community "instrumental" flags are not always right; keep trying other
            # providers and only trust the flag if nobody has lyrics.
            log.info("Track is flagged instrumental by %s; checking other providers", source)
            instrumental_source = instrumental_source or source
            continue
        lines = parse_lrc(text, info.duration) if synced else plain_to_lines(text, info.duration)
        if not lines:
            continue
        log.info("Lyrics: %d lines from %s (%s)", len(lines), source,
                 "synced" if synced else "plain, evenly spaced")
        return Lyrics(lines=lines, synced=synced, source=source)

    if instrumental_source:
        log.info("Treating track as instrumental (per %s)", instrumental_source)
        return Lyrics(lines=[], synced=True, source=instrumental_source)
    log.warning("No lyrics found for '%s' - '%s'", info.display_artist, info.title)
    return Lyrics()
