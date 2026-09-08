"""
Lyrics retrieval and LRC parsing.

Provider order:
  1. Sidecar file next to the audio: "<name>.lrc" (synced) or "<name>.txt" (plain)
  2. lrclib.net  - free, no API key, returns synced + plain lyrics
  3. syncedlyrics - aggregates Musixmatch / NetEase / Megalobiz / lrclib

Synced lyrics are only correct for the *edition* they were timed against, so lrclib
results are gathered for every known title variant and ranked by how closely their
reference duration matches this file. When only a mismatched edition exists we still
use it, but flag the mismatch so the video and the log say so.

Plain-text lyrics (no timestamps) are spread evenly across the song so the video
still shows something sensible; the `synced` flag lets the renderer say so.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import requests

from ..config import Settings
from ..models import Lyrics, LyricLine, LyricWord, SongInfo
from ..utils.text import format_time

log = logging.getLogger("lyricchord")

LRC_TAG = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
WORD_TAG = re.compile(r"<(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?>")
META_TAG = re.compile(r"^\[([a-zA-Z]+):([^\]]*)\]$")

MAX_LINE_HOLD = 10.0   # seconds a lyric line stays on screen without a following line
HTTP_HEADERS = {"User-Agent": "LyricChord/1.0 (https://github.com/MasstarVT/Lyric-Chord)"}
LRCLIB_BASE = "https://lrclib.net/api"
DURATION_TOLERANCE = 8.0   # seconds; beyond this the lyrics are for a different edition

# A provider hit: (lyrics text, synced?, reference duration in seconds or 0)
Hit = Tuple[str, bool, float]


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
    """Parse LRC text into timed lines.

    Supports several timestamps per line, enhanced word tags, and the standard
    `[offset:+/-ms]` header (positive values make the lyrics appear earlier).
    """
    entries: List[Tuple[float, str, Optional[List[LyricWord]]]] = []
    offset = 0.0
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        meta = META_TAG.match(raw)
        if meta:
            if meta.group(1).lower() == "offset":
                try:
                    offset = int(meta.group(2).strip().replace("+", "")) / 1000.0
                except ValueError:
                    pass
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
    if offset:
        _shift(lines, -offset)
    return lines


def plain_to_lines(text: str, duration: float) -> List[LyricLine]:
    """Spread untimed lyric lines evenly between an assumed intro and outro."""
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if not rows or duration <= 0:
        return []
    start, end = duration * 0.08, duration * 0.94
    step = (end - start) / len(rows)
    return [LyricLine(start + i * step, start + (i + 1) * step, row) for i, row in enumerate(rows)]


def _shift(lines: List[LyricLine], seconds: float) -> None:
    for line in lines:
        line.start = max(0.0, line.start + seconds)
        line.end = max(line.start, line.end + seconds)
        if line.words:
            for w in line.words:
                w.time = max(0.0, w.time + seconds)


def apply_offset(lyrics: Lyrics, offset_ms: int) -> Lyrics:
    """Shift every timestamp by offset_ms (positive = later)."""
    if offset_ms:
        _shift(lyrics.lines, offset_ms / 1000.0)
    return lyrics


# --------------------------------------------------------------------------- providers
def _sidecar(info: SongInfo) -> Optional[Tuple[str, bool, float, str]]:
    """Return (text, synced, ref_duration, source) from a .lrc/.txt next to the audio file."""
    lrc = info.path.with_suffix(".lrc")
    if lrc.exists():
        return lrc.read_text(encoding="utf-8", errors="ignore"), True, info.duration, "sidecar"
    txt = info.path.with_suffix(".txt")
    if txt.exists():
        from .chords.sheet import looks_like_chord_sheet

        content = txt.read_text(encoding="utf-8", errors="ignore")
        if not looks_like_chord_sheet(content):
            synced = bool(LRC_TAG.search(content))
            return content, synced, info.duration, "sidecar"
    return None


def score_lyrics_candidate(item: dict, file_duration: float) -> float:
    """Rank an lrclib record: synced beats plain, and closer reference duration beats farther.

    Scores stay ordered so any synced result outranks any plain one; within a class
    the duration difference decides. Instrumental-only records score lowest.
    """
    if item.get("syncedLyrics"):
        base = 2.0
    elif item.get("plainLyrics"):
        base = 1.0
    else:
        return -1.0
    ref = float(item.get("duration") or 0)
    if file_duration and ref:
        delta = abs(ref - file_duration)
        base -= min(0.9, delta / 120.0) if delta > 2.0 else 0.0
    return base


def _pick_lrclib(item: dict) -> Optional[Hit]:
    ref = float(item.get("duration") or 0)
    if item.get("syncedLyrics"):
        return item["syncedLyrics"], True, ref
    if item.get("plainLyrics"):
        return item["plainLyrics"], False, ref
    if item.get("instrumental"):
        return "", True, ref
    return None


def _lrclib_get(params: dict) -> Optional[dict]:
    r = requests.get(f"{LRCLIB_BASE}/get", params=params, headers=HTTP_HEADERS, timeout=15)
    return r.json() if r.status_code == 200 and isinstance(r.json(), dict) else None


def _lrclib_search(params: dict) -> List[dict]:
    r = requests.get(f"{LRCLIB_BASE}/search", params=params, headers=HTTP_HEADERS, timeout=15)
    if r.status_code != 200:
        return []
    data = r.json()
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def fetch_lrclib(info: SongInfo) -> Optional[Hit]:
    """Query lrclib.net: exact match per title variant first, then a ranked search."""
    titles = info.search_titles
    if not titles:
        return None
    try:
        # 1) The /get endpoint only answers when artist, title AND duration (+-2 s) match.
        if info.artist and info.duration > 0:
            for title in titles:
                data = _lrclib_get({"artist_name": info.artist, "track_name": title,
                                    "duration": int(round(info.duration))})
                hit = _pick_lrclib(data) if data else None
                if hit and hit[0]:
                    return hit

        # 2) Gather search results for every title variant and rank by edition match.
        candidates: Dict[object, dict] = {}
        for title in titles:
            queries = [{"track_name": title, "artist_name": info.artist}] if info.artist else [{"track_name": title}]
            if info.artist:
                queries.append({"q": f"{info.artist} {title}"})
            for params in queries:
                for item in _lrclib_search(params):
                    candidates.setdefault(item.get("id", id(item)), item)
        ranked = sorted(candidates.values(), key=lambda it: score_lyrics_candidate(it, info.duration), reverse=True)
        for item in ranked:
            hit = _pick_lrclib(item)
            if hit:
                return hit
    except (requests.RequestException, ValueError) as exc:
        log.warning("lrclib request failed: %s", exc)
    return None


def fetch_syncedlyrics(info: SongInfo) -> Optional[Hit]:
    """Fallback aggregator across several lyric providers (no duration information)."""
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
    return lrc, bool(LRC_TAG.search(lrc)), 0.0


# --------------------------------------------------------------------------- entry point
def fetch_lyrics(info: SongInfo, settings: Settings) -> Lyrics:
    """Return the best available Lyrics for a song (possibly empty)."""
    attempts = [("sidecar", lambda: _sidecar(info)),
                ("lrclib", lambda: fetch_lrclib(info)),
                ("syncedlyrics", lambda: fetch_syncedlyrics(info))]

    instrumental_source = ""
    for name, fn in attempts:
        result = fn()
        if not result:
            continue
        text, synced, ref = result[0], result[1], float(result[2] or 0.0)
        source = result[3] if len(result) > 3 else name
        if synced and not text.strip():
            # Community "instrumental" flags are not always right; keep trying other
            # providers and only trust the flag if nobody has lyrics.
            log.info("Track is flagged instrumental by %s; checking other providers", source)
            instrumental_source = instrumental_source or source
            continue
        lines = parse_lrc(text, info.duration) if synced else plain_to_lines(text, info.duration)
        if not lines:
            continue
        lyrics = Lyrics(lines=lines, synced=synced, source=source, ref_duration=ref)
        log.info("Lyrics: %d lines from %s (%s)", len(lines), source,
                 "synced" if synced else "plain, evenly spaced")
        mismatch = lyrics.duration_mismatch(info.duration, DURATION_TOLERANCE)
        if mismatch:
            log.warning("Lyrics were timed for a %s recording but this file is %s; timing will be off. "
                        "Add a sidecar .lrc or set a lyrics offset.",
                        format_time(ref), format_time(info.duration))
        return lyrics

    if instrumental_source:
        log.info("Treating track as instrumental (per %s)", instrumental_source)
        return Lyrics(lines=[], synced=True, source=instrumental_source)
    log.warning("No lyrics found for '%s' - '%s'", info.display_artist, info.title)
    return Lyrics()
