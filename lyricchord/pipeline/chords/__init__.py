"""
Chord acquisition front door.

Order of preference:
  1. Sidecar chord sheet next to the audio ("<name>.chords.txt", ".cho", ".crd", ".txt")
  2. Ultimate Guitar chord sheet (only if enabled in settings; best effort)
  3. Local audio analysis with librosa (always available)

Sheet-based sources need time-stamped lyrics to be placed on the timeline; if the
lyrics are missing, unsynced, or the alignment finds nothing, we fall back to analysis.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from ...config import Settings
from ...models import ChordTrack, Lyrics, SongInfo
from . import theory
from .local import detect_chords, estimate_tempo
from .online import fetch_ultimate_guitar_sheet
from .sheet import align_sheet_to_lyrics, looks_like_chord_sheet, parse_chord_sheet

log = logging.getLogger("lyricchord")

SIDECAR_SUFFIXES = [".chords.txt", ".cho", ".chopro", ".crd", ".txt"]


def chord_settings_fingerprint(settings: Settings) -> str:
    """Hash of every setting that changes the chord result (used as a cache key)."""
    parts = (settings.use_online_chords, settings.snap_chords_to_key, settings.prefer_flats,
             settings.min_chord_seconds, settings.include_seventh_chords)
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:10]


def lyrics_fingerprint(lyrics: Lyrics) -> str:
    """Hash of the lyric timeline a sheet would be aligned to (part of the chord cache key
    whenever a sheet source is possible, so shifted or replaced lyrics re-align)."""
    lines = lyrics.lines
    parts = (lyrics.source, lyrics.synced, len(lines),
             round(lines[0].start, 2) if lines else 0, round(lines[-1].end, 2) if lines else 0)
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:10]


def sidecar_sheet_path(path: Path) -> Optional[Path]:
    for suffix in SIDECAR_SUFFIXES:
        cand = path.with_suffix(suffix)
        if cand.exists() and cand != path:
            if suffix != ".txt" or looks_like_chord_sheet(cand.read_text(encoding="utf-8", errors="ignore")):
                return cand
    return None


def sheet_source_possible(path: Path, settings: Settings) -> bool:
    """True if chords for this song might come from a sheet rather than audio analysis."""
    return settings.use_online_chords or sidecar_sheet_path(path) is not None


def track_from_sheet(text: str, lyrics: Lyrics, info: SongInfo, source: str,
                     settings: Settings) -> ChordTrack:
    """Parse a chord sheet and align it to the lyrics timeline.

    Returns an empty track (caller falls back to analysis) unless the lyrics carry real
    timestamps: aligning to evenly spread plain lyrics would only manufacture wrong times.
    """
    if not lyrics.available or not lyrics.synced:
        log.warning("Chord sheet from %s needs synced lyrics to be timed; analysing audio instead", source)
        return ChordTrack(source=source)
    sheet = parse_chord_sheet(text)
    events = align_sheet_to_lyrics(sheet, lyrics, info.duration)
    if not events:
        log.warning("Could not align the %s chord sheet to the lyrics; analysing audio instead", source)
        return ChordTrack(source=source)

    parsed = [(theory.parse_label(e.label), e.duration) for e in events]
    tonic, mode = theory.infer_key_from_chords([(c, w) for c, w in parsed if c])
    # With prefer_flats off, keep each label's own spelling (None = infer from the text).
    use_flats = theory.key_uses_flats(tonic, mode) if settings.prefer_flats else None
    for e in events:
        e.label = theory.canonical_label(e.label, use_flats)

    return ChordTrack(events=events, key=theory.key_name(tonic, mode, settings.prefer_flats),
                      bpm=estimate_tempo(info.path), source=source)


def get_chords(path: Path, info: SongInfo, lyrics: Lyrics, settings: Settings,
               on_progress: Optional[Callable[[float], None]] = None) -> ChordTrack:
    """Return the best available ChordTrack for the song."""
    sidecar = sidecar_sheet_path(path)
    if sidecar is not None:
        log.info("Using chord sheet %s", sidecar.name)
        track = track_from_sheet(sidecar.read_text(encoding="utf-8", errors="ignore"), lyrics, info,
                                 "sheet", settings)
        if track.available:
            log.info("Chords: %d changes aligned from sidecar sheet", len(track.events))
            return track

    if settings.use_online_chords:
        if lyrics.available and lyrics.synced:
            text = fetch_ultimate_guitar_sheet(info.artist, info.title)
            if text:
                track = track_from_sheet(text, lyrics, info, "ultimate-guitar", settings)
                if track.available:
                    log.info("Chords: %d changes aligned from Ultimate Guitar", len(track.events))
                    return track
        else:
            log.info("Online chords need synced lyrics; falling back to audio analysis")

    return detect_chords(path, settings, on_progress)
