"""
Chord acquisition front door.

Order of preference:
  1. Sidecar chord sheet next to the audio ("<name>.chords.txt", ".cho", ".crd", ".txt")
  2. Ultimate Guitar chord sheet (only if enabled in settings; best effort)
  3. Local audio analysis with librosa (always available)

Sheet-based sources need time-stamped lyrics to be placed on the timeline; if the
lyrics are missing or the alignment finds nothing, we fall back to analysis.
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


def _sidecar_sheet(path: Path) -> Optional[str]:
    for suffix in SIDECAR_SUFFIXES:
        cand = path.with_suffix(suffix)
        if cand.exists() and cand != path:
            text = cand.read_text(encoding="utf-8", errors="ignore")
            if suffix != ".txt" or looks_like_chord_sheet(text):
                log.info("Using chord sheet %s", cand.name)
                return text
    return None


def track_from_sheet(text: str, lyrics: Lyrics, info: SongInfo, source: str,
                     settings: Settings, with_tempo: bool = True) -> ChordTrack:
    """Parse a chord sheet and align it to the lyrics timeline."""
    sheet = parse_chord_sheet(text)
    events = align_sheet_to_lyrics(sheet, lyrics, info.duration)
    if not events:
        return ChordTrack(source=source)

    parsed = [(theory.parse_label(e.label), e.duration) for e in events]
    tonic, mode = theory.infer_key_from_chords([(c, w) for c, w in parsed if c])
    use_flats = settings.prefer_flats and theory.key_uses_flats(tonic, mode)
    for e in events:
        e.label = theory.canonical_label(e.label, use_flats if settings.prefer_flats else None)

    bpm = estimate_tempo(info.path) if with_tempo else 0.0
    return ChordTrack(events=events, key=theory.key_name(tonic, mode, settings.prefer_flats),
                      bpm=bpm, source=source)


def get_chords(path: Path, info: SongInfo, lyrics: Lyrics, settings: Settings,
               on_progress: Optional[Callable[[float], None]] = None) -> ChordTrack:
    """Return the best available ChordTrack for the song."""
    # 1) Sidecar sheet
    text = _sidecar_sheet(path)
    if text:
        if lyrics.available:
            track = track_from_sheet(text, lyrics, info, "sheet", settings)
            if track.available:
                log.info("Chords: %d changes aligned from sidecar sheet", len(track.events))
                return track
            log.warning("Could not align the sidecar chord sheet to the lyrics; analysing audio")
        else:
            log.warning("Sidecar chord sheet found but no lyrics to align it to; analysing audio")

    # 2) Online sheet (opt-in)
    if settings.use_online_chords and lyrics.available and lyrics.synced:
        text = fetch_ultimate_guitar_sheet(info.artist, info.title)
        if text:
            track = track_from_sheet(text, lyrics, info, "ultimate-guitar", settings)
            if track.available:
                log.info("Chords: %d changes aligned from Ultimate Guitar", len(track.events))
                return track
            log.warning("Online chord sheet did not align with the lyrics; analysing audio")
    elif settings.use_online_chords:
        log.info("Online chords need synced lyrics; falling back to audio analysis")

    # 3) Local analysis
    return detect_chords(path, settings, on_progress)
