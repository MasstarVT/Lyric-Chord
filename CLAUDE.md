# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LyricChord is a Python desktop app (CustomTkinter) that turns a batch of audio files into
play-along MP4 videos: title, karaoke-highlighted lyrics, and the current/next chords on a
scrolling timeline. See README.md for user-facing setup and the settings reference.

## Workflow (standing instruction from the repo owner)

After any change to the code: update README.md and this file so they still describe the
behaviour accurately, run the test suite, then commit and push to `origin`. Do not leave
work uncommitted at the end of a task.

## Commands

Always use the project virtualenv interpreter; the global Python does not have the deps.

```bash
.venv\Scripts\python.exe main.py                          # desktop GUI
.venv\Scripts\python.exe main.py --cli <files/folders> -o <out> --resolution 720p   # headless batch
.venv\Scripts\python.exe -m pytest tests -q               # full suite (~15 s, renders real MP4s)
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q                        # fast, no audio/ffmpeg
.venv\Scripts\python.exe -m pytest tests/test_audio_and_render.py -k loop -q        # single test by keyword
```

Setup from scratch: `python -m venv .venv` then `pip install -r requirements-dev.txt`.
No system FFmpeg is required; `imageio-ffmpeg` bundles one (see `utils/ffmpeg.py` for the
lookup order: `LYRICCHORD_FFMPEG` env var, PATH, bundled). There is no linter configured.

## Architecture

Data flows through immutable-ish dataclasses in `lyricchord/models.py`
(`SongInfo -> Lyrics -> ChordTrack -> SongData`) and is orchestrated by
`pipeline/processor.py::process_song`, which is the one place to read to understand the
whole per-song flow: metadata -> lyrics -> chords -> render, with cache reads/writes and
cancel checks between stages.

**Chord source priority** (`pipeline/chords/__init__.py::get_chords`): sidecar chord sheet
next to the audio -> Ultimate Guitar sheet (opt-in, scraping) -> local librosa detection.
Sheet-based sources produce chords with no timing of their own; `chords/sheet.py` places
them by fuzzy-matching sheet lines to time-stamped lyric lines, so they require synced
lyrics and fall back to audio analysis otherwise. `chords/local.py` is chroma template
matching plus Viterbi smoothing; its tunables are module constants at the top of the file.
`chords/theory.py` owns chord parsing/spelling and key logic and is used by all three.

**Identification** (`pipeline/metadata.py`): tags -> filename -> title-only lookups ->
AcoustID. Title-only lookups exist because untagged rips are common: lrclib's search
results vote on the artist (`artist_consensus`), then a MusicBrainz recording search
filtered by the file's duration picks the exact recording (`rank_musicbrainz` weights
duration closeness and release count, since MusicBrainz text scores rank every cover
equally). The original filename title is kept in `SongInfo.alt_titles` for lyric searches.
MusicBrainz allows ~1 request/s and returns 503 when busy; treat it as optional.

**Lyrics** (`pipeline/lyrics.py`): sidecar `.lrc`/`.txt` -> lrclib -> syncedlyrics. lrclib
results are gathered for every title variant (the `/get` answer is only one more
candidate, never trusted alone) and `choose_lyrics_candidate` picks by consensus:
records whose length matches the file are clustered by first-line time (truncated uploads
still vote for the right start), a timeline carried by at least as many clearly
different-length records as matching ones is treated as copied from that edition and
heavily penalised (a plain "also exists elsewhere" test fails because junk uploads with
absurd lengths share the correct timeline too), and the most complete record nearest the
cluster median wins. Run the selection with DEBUG logging to see every candidate's score;
`tests/eye in the sky.mp3` (git-ignored, 6:31 album edit) is the reference case. `score_lyrics_candidate` (synced > plain, then closest length) is only the
fallback when no record matches the file's length. `Lyrics.ref_duration` records the
chosen edition's length and `duration_mismatch()` drives both the log warning and the
on-screen note. `LYRICS_CACHE_KEY` in `processor.py` must be bumped whenever selection
logic changes, otherwise cached bad choices survive. Plain lyrics
are spread evenly and flagged `synced=False`. An "instrumental" flag from one provider
does not stop the search. `parse_lrc` honours the `[offset:]` header.

**Rendering** deliberately avoids MoviePy. `render/frames.py::FrameComposer` builds a static
layer once (background, panels, header) and draws only time-dependent parts per frame,
caching rasterised text by (text, font, colour). `render/renderer.py` pipes raw frames into
a single FFmpeg process on stdin. Two modes: solid/gradient backgrounds are baked into RGB
frames; a loop video background sends RGBA overlay frames and lets FFmpeg's `overlay`
filter composite them. Output is written to `<name>.part.mp4` and renamed on success.
All layout sizes are computed from a 1080-pixel short side (`compute_layout`), so font
sizes in settings are "1080p units".

**GUI threading** (`gui/app.py`): all pipeline work runs in `BatchRunner` (a thread in
`processor.py`). It never touches Tk; it posts tuples to an event queue, and logging goes
through a `QueueHandler` (`utils/logging_utils.py`). `App._poll` drains both queues every
100 ms on the Tk thread. Keep it that way: any Tk call from a worker thread is a bug
(the FFmpeg version probe also reports back via an attribute read in `_poll`).

**Settings** (`config.py::Settings`) is a flat dataclass persisted as JSON in
`~/.lyricchord/settings.json`. `gui/settings_panel.py` binds one Tk variable per field
name and converts back using `get_type_hints(Settings)`, so adding a setting means adding
a dataclass field plus one `_option/_check/_entry/_slider/_color` call in `_build`.
Settings that change chord results must be included in `chord_settings_fingerprint`,
which is part of the cache key.

**Cache** (`pipeline/cache.py`): JSON under `<output_dir>/.lyricchord_cache/`, keyed by
file path+size+mtime. Lyrics are cached before the user offset is applied; chords are
cached per settings fingerprint.

## Gotchas

- librosa 1.0 is installed: every librosa call must use keyword arguments.
- CustomTkinter 6 and tkinterdnd2: drag-and-drop works by mixing `TkinterDnD.DnDWrapper`
  into the root class and calling `TkinterDnD._require(self)`; tkinterdnd2 patches
  `drop_target_register`/`dnd_bind` onto all widgets at import time.
- `errors.Cancelled` lives in its own module because both the renderer and the processor
  raise it; `processor.py` imports the renderer lazily to avoid pulling Pillow into every
  import.
- `utils/audio.py::decode_audio` decodes via FFmpeg, not `librosa.load`, so MP3/M4A work
  regardless of libsndfile codec support.
- Tests synthesise audio (sine-wave triads) rather than shipping fixtures; the chord
  detector must recover C-G-Am-F in order from `tests/test_audio_and_render.py`. Real
  audio the owner drops into `tests/` is git-ignored; use it for manual checks only, and
  keep network lookups out of the automated tests (rank/score functions are pure for
  that reason).
- The Ultimate Guitar fetcher (`chords/online.py`) parses page markup and cannot be tested
  offline; treat it as best-effort and never let its failure propagate.
