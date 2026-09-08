"""
Chord-sheet parsing and lyric alignment.

Two text formats are understood:

  Ultimate-Guitar style (chords on the line above the lyric):
        Am        F
      Hello darkness, my old friend

  ChordPro style (chords inline in brackets):
      [Am]Hello darkness, [F]my old friend

Once parsed, chords are placed on the timeline by fuzzy-matching each sheet line
to a time-stamped lyric line, and interpolating chords that sit between them
(intros, instrumental breaks, outros).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from ...models import ChordEvent, LyricLine, Lyrics
from ...utils.text import normalize
from .theory import is_chord_token, parse_label

SECTION_RE = re.compile(
    r"^\s*[\[\(]?\s*(verse|chorus|bridge|intro|outro|pre[- ]?chorus|solo|instrumental|"
    r"interlude|hook|refrain|tag|coda|ending|break|riff|post[- ]?chorus)\b[^\]\)]*[\]\)]?\s*:?\s*$",
    re.IGNORECASE,
)
DIRECTIVE_RE = re.compile(r"^\s*\{[^}]*\}\s*$")            # ChordPro {title: ...}
INLINE_RE = re.compile(r"\[([^\[\]\s]{1,12})\]")           # [Am] inline tokens
UG_CH_RE = re.compile(r"\[ch\](.*?)\[/ch\]", re.IGNORECASE)
UG_TAB_RE = re.compile(r"\[/?tab\]", re.IGNORECASE)
_IGNORED_TOKENS = {"|", "||", "-", "--", "x2", "x3", "x4", "x8", "n.c.", "nc", "(x2)", "(x3)", "(x4)"}
MATCH_THRESHOLD = 0.6


@dataclass
class SheetLine:
    text: str                                        # "" for chord-only lines
    chords: List[Tuple[int, str]] = field(default_factory=list)  # (char index, label)


def strip_ug_markup(text: str) -> str:
    text = UG_TAB_RE.sub("", text)
    return UG_CH_RE.sub(lambda m: m.group(1), text).replace("\r\n", "\n").replace("\r", "\n")


def _chord_tokens(line: str) -> Optional[List[Tuple[int, str]]]:
    """(column, label) for every chord on a chord-only line, or None if any token is
    not a chord (then the line is lyrics). One tokenizer serves both questions so the
    classification and the extracted positions can never disagree."""
    out: List[Tuple[int, str]] = []
    for m in re.finditer(r"\S+", line):
        raw = m.group(0)
        tok = raw.strip("()[]|,")
        if not tok or raw.lower() in _IGNORED_TOKENS or tok.lower() in _IGNORED_TOKENS:
            continue
        if not is_chord_token(tok):
            return None
        out.append((m.start(), tok))
    return out or None


def _is_chord_line(line: str) -> bool:
    return _chord_tokens(line) is not None


def _chord_positions(line: str) -> List[Tuple[int, str]]:
    return _chord_tokens(line) or []


def _parse_inline(line: str) -> Optional[SheetLine]:
    """Return a SheetLine for a ChordPro line, or None if the line has no inline chords."""
    matches = [m for m in INLINE_RE.finditer(line) if parse_label(m.group(1)) is not None]
    if not matches:
        return None
    text, chords, pos = "", [], 0
    for m in matches:
        text += line[pos:m.start()]
        chords.append((len(text), m.group(1)))
        pos = m.end()
    text += line[pos:]
    return SheetLine(text.strip(), [(min(p, len(text.strip())), c) for p, c in chords])


def parse_chord_sheet(text: str) -> List[SheetLine]:
    """Parse UG-style or ChordPro text into SheetLines (chords attached to lyric lines)."""
    text = strip_ug_markup(text)
    out: List[SheetLine] = []
    pending: Optional[List[Tuple[int, str]]] = None

    def flush_pending():
        nonlocal pending
        if pending:
            out.append(SheetLine("", pending))
        pending = None

    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            flush_pending()
            continue
        if DIRECTIVE_RE.match(line) or SECTION_RE.match(line):
            flush_pending()
            continue
        inline = _parse_inline(line)
        if inline is not None:
            flush_pending()
            out.append(inline)
            continue
        if _is_chord_line(line):
            flush_pending()
            pending = _chord_positions(line)
            continue
        # Plain lyric line: attach the chord line above it, if any.
        lyric = line.strip()
        offset = len(line) - len(line.lstrip())
        chords = []
        if pending:
            chords = [(max(0, min(p - offset, len(lyric))), c) for p, c in pending]
        out.append(SheetLine(lyric, chords))
        pending = None
    flush_pending()
    return out


def looks_like_chord_sheet(text: str) -> bool:
    text = strip_ug_markup(text)
    chord_lines = sum(1 for l in text.split("\n") if l.strip() and _is_chord_line(l))
    inline = sum(1 for m in INLINE_RE.finditer(text) if parse_label(m.group(1)) is not None)
    return chord_lines >= 3 or inline >= 3


# --------------------------------------------------------------------------- alignment
def _time_at(line: LyricLine, char_index: int) -> float:
    """Absolute time at which `char_index` of the line is sung."""
    n = max(1, len(line.text))
    if line.words and len(line.words) > 1:
        count = 0
        for w in line.words:
            if char_index <= count + len(w.text):
                return w.time
            count += len(w.text) + 1
        return line.words[-1].time
    return line.start + (char_index / n) * line.duration


def align_sheet_to_lyrics(sheet: List[SheetLine], lyrics: Lyrics, duration: float) -> List[ChordEvent]:
    """Place sheet chords on the timeline using time-stamped lyric lines as anchors."""
    lyric_lines = [l for l in lyrics.lines if l.text.strip()]
    sheet_lyric_idx = [i for i, s in enumerate(sheet) if s.text.strip()]
    if not lyric_lines or not sheet_lyric_idx:
        return []
    # One matcher per sheet line: SequenceMatcher indexes its second sequence once and
    # set_seq1() is cheap, so this avoids rebuilding that index for every lyric line.
    matchers: Dict[int, SequenceMatcher] = {i: SequenceMatcher(None, "", normalize(sheet[i].text))
                                            for i in sheet_lyric_idx}

    timed: List[Tuple[float, str]] = []
    anchors: List[Tuple[int, LyricLine]] = []     # (sheet index, matched lyric line)
    last = 0
    for L in lyric_lines:
        nl = normalize(L.text)
        best, best_score = None, 0.0
        for i in sheet_lyric_idx:
            sm = matchers[i]
            sm.set_seq1(nl)
            if sm.real_quick_ratio() < best_score or sm.quick_ratio() < best_score:
                continue
            score = sm.ratio()
            closer = best is not None and abs(i - last) < abs(best - last)
            if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and closer):
                best, best_score = i, score
        if best is None or best_score < MATCH_THRESHOLD:
            continue
        last = best
        anchors.append((best, L))
        for pos, label in sheet[best].chords:
            timed.append((_time_at(L, pos), label))

    if not anchors:
        return []

    # Chord-only lines (intro / breaks / outro) get spread between neighbouring anchors.
    def spread(sheet_from: int, sheet_to: int, t0: float, t1: float) -> None:
        block = [s for s in sheet[sheet_from:sheet_to] if not s.text.strip() and s.chords]
        chords = [c for s in block for _, c in s.chords]
        if not chords or t1 <= t0:
            return
        step = (t1 - t0) / len(chords)
        for k, label in enumerate(chords):
            timed.append((t0 + k * step, label))

    first_idx, first_line = anchors[0]
    spread(0, first_idx, 0.0, first_line.start)
    for (ia, la), (ib, lb) in zip(anchors, anchors[1:]):
        if ib > ia + 1:
            spread(ia + 1, ib, la.end, lb.start)
    last_idx, last_line = anchors[-1]
    spread(last_idx + 1, len(sheet), last_line.end, duration)

    timed.sort(key=lambda x: x[0])
    events: List[ChordEvent] = []
    for t, label in timed:
        if events and events[-1].label == label:
            continue
        if events:
            events[-1].end = t
        events.append(ChordEvent(t, t, label))
    if events:
        events[-1].end = max(duration, events[-1].start)
        if events[0].start < 1.0:
            events[0].start = 0.0
    return [e for e in events if e.end > e.start]
