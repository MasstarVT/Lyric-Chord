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
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

from ..models import Lyrics, LyricLine, LyricWord, SongInfo
from ..utils.net import HTTP_HEADERS
from ..utils.text import artist_key, format_time, normalize

log = logging.getLogger("lyricchord")

LRC_TAG = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
WORD_TAG = re.compile(r"<(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?>")
META_TAG = re.compile(r"^\[([a-zA-Z]+):([^\]]*)\]$")

MAX_LINE_HOLD = 10.0   # seconds a lyric line stays on screen without a following line
LRCLIB_BASE = "https://lrclib.net/api"
LRCLIB_PARALLELISM = 6     # concurrent lrclib queries per song
DURATION_TOLERANCE = 8.0   # seconds; beyond this the lyrics are for a different edition

# A provider hit: (lyrics text, synced?, reference duration in seconds or 0).
# Empty text with synced=True means "the provider says this track is instrumental".
Hit = Tuple[str, bool, float]


def _num(value) -> float:
    """Tolerant float for JSON fields that are occasionally null, strings or junk."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


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
def has_lyrics_sidecar(path: Path) -> bool:
    """True if a .lrc or .txt sits next to the audio (read fresh every run, never cached)."""
    return path.with_suffix(".lrc").exists() or path.with_suffix(".txt").exists()


def _sidecar(info: SongInfo) -> Optional[Hit]:
    """Lyrics from a .lrc/.txt next to the audio file. A blank file is ignored, not
    mistaken for an instrumental declaration."""
    lrc = info.path.with_suffix(".lrc")
    if lrc.exists():
        content = lrc.read_text(encoding="utf-8", errors="ignore")
        return (content, True, info.duration) if content.strip() else None
    txt = info.path.with_suffix(".txt")
    if txt.exists():
        from .chords.sheet import looks_like_chord_sheet

        content = txt.read_text(encoding="utf-8", errors="ignore")
        if content.strip() and not looks_like_chord_sheet(content):
            return content, bool(LRC_TAG.search(content)), info.duration
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
    ref = _num(item.get("duration"))
    if file_duration and ref:
        delta = abs(ref - file_duration)
        base -= min(0.9, delta / 120.0) if delta > 2.0 else 0.0
    return base


CLUSTER_TOLERANCE = 6.0      # seconds; first-line times this close count as the same timing
FOREIGN_EDITION_GAP = 20.0   # a record whose length differs this much belongs to another edition


class _Timeline:
    """Parsed view of one synced lrclib record, used for consensus voting."""

    __slots__ = ("item", "ref", "first", "last", "n_lines")

    def __init__(self, item: dict, ref: float, first: float, last: float, n_lines: int):
        self.item, self.ref, self.first, self.last, self.n_lines = item, ref, first, last, n_lines


def _timeline(item: dict) -> Optional[_Timeline]:
    if not isinstance(item.get("syncedLyrics"), str) or not item["syncedLyrics"]:
        return None
    ref = _num(item.get("duration"))
    lines = [l for l in parse_lrc(item["syncedLyrics"], ref) if l.text.strip()]
    if not lines:
        return None
    return _Timeline(item, ref, lines[0].start, lines[-1].start, len(lines))


def _same_start(a: _Timeline, b: _Timeline) -> bool:
    """Records vote on where singing starts; a truncated upload still agrees on that."""
    return abs(a.first - b.first) <= CLUSTER_TOLERANCE


def _same_timeline(a: _Timeline, b: _Timeline) -> bool:
    """Strict match on both ends, used to recognise a timeline copied from another edition."""
    return abs(a.first - b.first) <= CLUSTER_TOLERANCE and abs(a.last - b.last) <= 2 * CLUSTER_TOLERANCE


def _copied_from_other_edition(t: _Timeline, matching: List[_Timeline], foreign: List[_Timeline]) -> bool:
    """True if this timeline is carried by at least as many records of clearly different
    length as records of matching length.

    A popular single's timing gets pasted under the album edit by several uploaders, but
    the single itself has many more records, so the foreign sharers outnumber the
    matching ones. The reverse (a correct album timing also present on one or two
    oddly-labelled uploads) leaves the matching sharers in the majority.
    """
    foreign_sharers = sum(1 for f in foreign if _same_timeline(f, t))
    matching_sharers = sum(1 for m in matching if _same_timeline(m, t))
    return foreign_sharers >= max(2, matching_sharers)


# Given candidate first-line times, return a "voice enters here" score for each (see vocal.py).
OnsetScorer = Callable[[List[float]], List[float]]
ONSET_DISAGREEMENT = 0.5    # seconds; starts closer than this are the same answer (imperceptible)


def choose_lyrics_candidate(items: List[dict], file_duration: float,
                            tolerance: float = DURATION_TOLERANCE,
                            onset_scorer: Optional[OnsetScorer] = None) -> Tuple[Optional[dict], str]:
    """Pick the lrclib record whose timeline most plausibly belongs to this file's edition.

    Community uploads often paste one edition's timings under another edition's track
    (the radio single's timing filed under the album edit with a long intro). Duration
    metadata alone cannot catch that, so among records whose length matches the file:
      * records that agree on first/last line times form clusters (bigger is better);
      * a timeline that also appears under a record of clearly different length was
        copied from another edition and is heavily penalised;
      * timelines whose last line sits far from the end of the file are mildly penalised;
      * within a cluster, prefer the most complete record nearest the median start;
      * if the strongest records still disagree about the first line by more than
        ONSET_DISAGREEMENT seconds (different masters, different lead-in), `onset_scorer`
        is asked which candidate start the audio supports.
    Returns (record, human note) or (None, note) if nothing usable exists.
    """
    timelines = [t for t in (_timeline(it) for it in items if isinstance(it, dict)) if t]
    if file_duration > 0:
        matching = [t for t in timelines if abs(t.ref - file_duration) <= tolerance]
        foreign = [t for t in timelines if abs(t.ref - file_duration) > FOREIGN_EDITION_GAP]
    else:
        matching, foreign = timelines, []

    if matching:
        copied_flags = [_copied_from_other_edition(t, matching, foreign) for t in matching]
        if all(copied_flags):
            # Every candidate shares a timeline with another edition (e.g. only the outro
            # differs between single and album). Then it is simply the right timing.
            copied_flags = [False] * len(matching)
        best, best_score, best_note = None, float("-inf"), ""
        scored: List[Tuple[float, _Timeline]] = []
        for t, copied in zip(matching, copied_flags):
            cluster = [o for o in matching if _same_start(o, t)]
            median_first = sorted(o.first for o in cluster)[len(cluster) // 2]
            score = float(len(cluster))
            if copied:
                score -= 10.0
            tail = (file_duration - t.last) / file_duration if file_duration > 0 else 0.0
            score -= 3.0 * max(0.0, tail - 0.3)
            score += 0.02 * t.n_lines
            score -= 0.1 * abs(t.first - median_first)
            log.debug("lrclib candidate #%s: len %.0fs first %.1fs last %.0fs lines %d cluster %d copied %s -> %.2f",
                      t.item.get("id", "?"), t.ref, t.first, t.last, t.n_lines, len(cluster), copied, score)
            scored.append((score, t))
            if score > best_score:
                best, best_score = t, score
                best_note = (f"{len(cluster)} of {len(matching)} matching-length records agree on this start"
                             + ("; timing copied from another edition" if copied else ""))
        assert best is not None

        # Different masters of one edition differ in lead-in silence by a second or two,
        # and the records reflect that. Let the audio arbitrate among the strong records.
        if onset_scorer is not None:
            peers = sorted(((s, t) for s, t in scored if _same_start(t, best) and s >= best_score - 1.5),
                           key=lambda st: st[1].first)
            # Starts within ONSET_DISAGREEMENT of each other are the same answer; group them
            # so the audio compares genuinely different starts and completeness still
            # decides within a group.
            groups: List[List[Tuple[float, _Timeline]]] = []
            for s, t in peers:
                if groups and t.first - groups[-1][0][1].first <= ONSET_DISAGREEMENT:
                    groups[-1].append((s, t))
                else:
                    groups.append([(s, t)])
            if len(groups) > 1:
                rep_times = [sorted(t.first for _, t in g)[len(g) // 2] for g in groups]
                try:
                    rises = onset_scorer(rep_times)
                except Exception as exc:  # audio problems must never sink the lyrics stage
                    log.debug("vocal onset check failed: %s", exc)
                    rises = []
                if len(rises) == len(groups) and all(math.isfinite(r) for r in rises):
                    own = next(i for i, g in enumerate(groups) if any(t is best for _, t in g))
                    idx = max(range(len(groups)), key=lambda i: (round(rises[i], 3), i == own))
                    chosen = max(groups[idx], key=lambda st: st[0])[1]
                    if chosen is not best:
                        best_note += f"; audio places the first line at {chosen.first:.1f}s rather than {best.first:.1f}s"
                        best = chosen
        return best.item, best_note

    # No record matches this edition: fall back to the closest length, synced before plain.
    ranked = sorted((it for it in items if isinstance(it, dict)),
                    key=lambda it: score_lyrics_candidate(it, file_duration), reverse=True)
    if ranked and score_lyrics_candidate(ranked[0], file_duration) >= 0:
        return ranked[0], "no record matches this file's length; using the closest edition"
    return None, "no usable records"


def _pick_lrclib(item: dict) -> Optional[Hit]:
    ref = _num(item.get("duration"))
    if item.get("syncedLyrics"):
        return item["syncedLyrics"], True, ref
    if item.get("plainLyrics"):
        return item["plainLyrics"], False, ref
    if item.get("instrumental"):
        return "", True, ref
    return None


_TITLE_SEPARATOR = re.compile(r"\s*[/|&+,]\s*|\s+-\s+")


def title_variants(title: str, limit: int = 4) -> List[str]:
    """Spellings uploaders use for the same track: 'Sirius / Eye in the Sky' is also filed
    as 'Sirius/Eye in the Sky', 'Sirius - Eye In The Sky', 'Sirius, Eye in the Sky'...
    Returns the title first, then punctuation-free and re-joined forms."""
    out: List[str] = []

    def add(t: str) -> None:
        t = t.strip()
        if t and t.lower() not in {o.lower() for o in out}:
            out.append(t)

    add(title)
    parts = [p for p in _TITLE_SEPARATOR.split(title) if p.strip()]
    if len(parts) > 1:
        add(" ".join(parts))
        add("/".join(parts))
        add(" - ".join(parts))
    add(normalize(title))
    return out[:limit]


def artist_matches(record_artist: str, wanted: str) -> bool:
    """Lenient artist comparison: 'Alan Parsons Project' ~ 'The Alan Parsons Project',
    'Simon & Garfunkel' ~ 'Simon and Garfunkel' (see utils.text.artist_key)."""
    if not wanted:
        return True
    a, b = artist_key(record_artist or ""), artist_key(wanted)
    if not a or not b:
        return False
    return a == b or a in b or b in a


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
    """Query lrclib.net and choose among all records by edition consensus.

    The /get endpoint's single answer is deliberately not trusted on its own: it is
    just one more candidate for `choose_lyrics_candidate`.
    """
    titles = info.search_titles
    if not titles:
        return None

    # Several angles on the same song: lrclib's search is fuzzy and its result sets vary
    # between calls, so more queries (and the spellings uploaders use) mean a steadier vote.
    jobs: List[Tuple[Callable[[dict], object], dict]] = []
    if info.artist and info.duration > 0:
        for title in titles:
            jobs.append((_lrclib_get, {"artist_name": info.artist, "track_name": title,
                                       "duration": int(round(info.duration))}))
    queries: List[dict] = []
    for title in titles:
        for k, variant in enumerate(title_variants(title)):
            if k == 0:
                queries.append({"track_name": variant})
            queries.append({"q": variant})
            if info.artist:
                if k == 0:
                    queries.append({"track_name": variant, "artist_name": info.artist})
                queries.append({"q": f"{info.artist} {variant}"})
    seen_queries = set()
    for params in queries[:16]:
        key = tuple(sorted(params.items()))
        if key not in seen_queries:
            seen_queries.add(key)
            jobs.append((_lrclib_search, params))

    def run(job: Tuple[Callable[[dict], object], dict]) -> Tuple[dict, object, Optional[Exception]]:
        fn, params = job
        try:
            return params, fn(params), None
        except (requests.RequestException, ValueError) as exc:
            return params, None, exc

    # Independent requests, so run them together; one slow or failed query neither
    # stalls nor discards the others.
    candidates: Dict[object, dict] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=LRCLIB_PARALLELISM) as pool:
        for params, result, exc in pool.map(run, jobs):
            if exc is not None:
                failures += 1
                log.warning("lrclib request failed (%s): %s", params, exc)
                continue
            for item in (result if isinstance(result, list) else [result] if result else []):
                if isinstance(item, dict):
                    candidates.setdefault(item.get("id", id(item)), item)

    if info.artist:
        # Covers of the same length must not vote on the timing of this recording - but
        # the artist may itself be a guess, so an empty result keeps the unfiltered pool.
        by_artist = {k: v for k, v in candidates.items()
                     if artist_matches(str(v.get("artistName") or ""), info.artist)}
        log.debug("lrclib: %d of %d records are by '%s'", len(by_artist), len(candidates), info.artist)
        if by_artist:
            candidates = by_artist
        elif candidates:
            log.info("lrclib: no records credited to '%s'; considering all %d records", info.artist, len(candidates))
    log.debug("lrclib: %d candidate records gathered (%d failed queries)", len(candidates), failures)
    if not candidates:
        return None

    scorer: Optional[OnsetScorer] = None
    if info.path.is_file():
        from .vocal import vocal_onset_rise

        scorer = lambda times: vocal_onset_rise(info.path, times)  # noqa: E731
    chosen, note = choose_lyrics_candidate(list(candidates.values()), info.duration, onset_scorer=scorer)
    if chosen is not None:
        log.info("lrclib: record #%s chosen (%s)", chosen.get("id", "?"), note)
        return _pick_lrclib(chosen)
    # Nothing with lyrics; honour an instrumental flag only from a length-matched record
    # (or any record when the file length is unknown).
    for item in candidates.values():
        ref = _num(item.get("duration"))
        if item.get("instrumental") and (not info.duration or abs(ref - info.duration) <= DURATION_TOLERANCE):
            return "", True, ref
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
def fetch_lyrics(info: SongInfo) -> Lyrics:
    """Return the best available Lyrics for a song (possibly empty).

    The user's global lyric offset is applied by the caller after caching, so this
    function is deliberately independent of Settings.
    """
    attempts: List[Tuple[str, Callable[[], Optional[Hit]]]] = [
        ("sidecar", lambda: _sidecar(info)),
        ("lrclib", lambda: fetch_lrclib(info)),
        ("syncedlyrics", lambda: fetch_syncedlyrics(info)),
    ]

    instrumental_source = ""
    for source, fn in attempts:
        result = fn()
        if not result:
            continue
        text, synced, ref = result
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
