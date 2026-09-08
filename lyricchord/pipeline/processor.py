"""
Per-song orchestration and the batch worker thread used by the GUI.

process_song(): metadata -> lyrics -> chords -> render, with caching and cancel checks.
BatchRunner:    runs process_song() for each file in a background thread and posts
                events to a queue that the GUI drains on the Tk main loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from ..config import Settings
from ..models import ChordTrack, Lyrics, SongData, SongInfo
from ..utils.audio import AUDIO_EXTENSIONS
from ..utils.text import safe_filename
from .cache import SongCache
from .chords import chord_settings_fingerprint, get_chords
from .lyrics import apply_offset, fetch_lyrics
from .metadata import extract_metadata

log = logging.getLogger("lyricchord")

# progress(stage, fraction_of_this_song, human_message)
ProgressFn = Callable[[str, float, str], None]

# Bump when lyric selection logic changes so stale cached choices are not reused.
LYRICS_CACHE_KEY = "lyrics:v3"


class Cancelled(Exception):
    """Raised inside the pipeline when the user pressed Cancel."""


def find_audio_files(paths: Iterable[str], recursive: bool = True) -> List[Path]:
    """Expand files / folders into a sorted, de-duplicated list of audio files."""
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found += [f for f in it if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        elif p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(p)
    seen, out = set(), []
    for f in sorted(found, key=lambda x: str(x).lower()):
        key = str(f.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def output_path_for(info: SongInfo, settings: Settings) -> Path:
    name = f"{info.display_artist} - {info.display_title}" if info.artist else info.display_title
    return Path(settings.output_dir) / f"{safe_filename(name)}.mp4"


def _check(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def process_song(path: Path, settings: Settings, progress: Optional[ProgressFn] = None,
                 cancel: Optional[threading.Event] = None) -> Path:
    """Run the whole pipeline for one file and return the written MP4 path."""
    from ..render.renderer import render_video  # lazy: Pillow/ffmpeg only needed here

    def report(stage: str, frac: float, msg: str) -> None:
        if progress:
            progress(stage, frac, msg)

    cache = SongCache(Path(settings.output_dir), settings.use_cache)
    t0 = time.time()

    report("metadata", 0.0, "Reading metadata")
    info = extract_metadata(path, settings)
    if info.duration <= 0:
        raise RuntimeError("Could not determine audio duration (is the file valid?)")
    out_path = output_path_for(info, settings)
    if settings.skip_existing and out_path.exists():
        log.info("Skipping %s (output already exists)", out_path.name)
        report("done", 1.0, "Skipped (exists)")
        return out_path
    _check(cancel)

    report("lyrics", 0.08, "Fetching lyrics")
    cached = cache.get(path, LYRICS_CACHE_KEY)
    if cached:
        lyrics = Lyrics.from_dict(cached)
        log.info("Lyrics: %d lines from cache (%s)", len(lyrics.lines), lyrics.source)
    else:
        lyrics = fetch_lyrics(info, settings)
        cache.put(path, LYRICS_CACHE_KEY, lyrics.to_dict())
    lyrics = apply_offset(lyrics, settings.lyrics_offset_ms)
    _check(cancel)

    report("chords", 0.2, "Detecting chords")
    key = f"chords:{chord_settings_fingerprint(settings)}"
    cached = cache.get(path, key)
    if cached:
        chords = ChordTrack.from_dict(cached)
        log.info("Chords: %d segments from cache (%s)", len(chords.events), chords.source)
    else:
        chords = get_chords(path, info, lyrics, settings,
                            on_progress=lambda f: report("chords", 0.2 + 0.3 * f, "Analysing audio"))
        cache.put(path, key, chords.to_dict())
    _check(cancel)

    report("render", 0.5, "Rendering video")
    song = SongData(info=info, lyrics=lyrics, chords=chords)
    render_video(song, settings, out_path,
                 progress=lambda f: report("render", 0.5 + 0.5 * f, f"Rendering {int(f * 100)}%"),
                 cancel=cancel)
    log.info("Finished %s in %.0fs", out_path.name, time.time() - t0)
    report("done", 1.0, "Done")
    return out_path


class BatchRunner(threading.Thread):
    """Background thread that processes a list of files and reports via a queue.

    Events posted (tuples, first element is the kind):
      ("file_start", index, total, path)
      ("progress", overall_fraction, message)
      ("file_done", index, total, path, output_path)
      ("file_error", index, total, path, error_message)
      ("batch_done", n_ok, n_failed, cancelled)
    """

    def __init__(self, files: List[Path], settings: Settings,
                 events: "queue.Queue", cancel: threading.Event):
        super().__init__(daemon=True, name="lyricchord-batch")
        self.files = files
        self.settings = settings
        self.events = events
        self.cancel = cancel

    def run(self) -> None:
        n = len(self.files)
        ok = failed = 0
        cancelled = False
        for i, path in enumerate(self.files):
            if self.cancel.is_set():
                cancelled = True
                break
            self.events.put(("file_start", i, n, path))
            log.info("Processing %d of %d: %s", i + 1, n, path.name)

            def progress(stage: str, frac: float, msg: str, _i=i) -> None:
                overall = (_i + frac) / n
                self.events.put(("progress", overall, f"[{_i + 1}/{n}] {path.name}: {msg}"))

            try:
                out = process_song(path, self.settings, progress, self.cancel)
                ok += 1
                self.events.put(("file_done", i, n, path, out))
            except Cancelled:
                cancelled = True
                log.warning("Cancelled while processing %s", path.name)
                break
            except Exception as exc:  # keep the batch going
                failed += 1
                log.error("Failed %s: %s", path.name, exc, exc_info=log.isEnabledFor(logging.DEBUG))
                self.events.put(("file_error", i, n, path, str(exc)))
        self.events.put(("batch_done", ok, failed, cancelled))
