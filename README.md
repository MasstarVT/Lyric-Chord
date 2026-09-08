# LyricChord

Batch generator of **play-along lyric + chord videos** from MP3 (or FLAC/M4A/WAV/OGG) files.
Drop a folder of songs on the window, press *Start batch*, and get one MP4 per song showing
the title, the current lyric line with karaoke-style highlighting, and the chords playing
now and next on a scrolling timeline that stays in sync with the audio.

```
+----------------------------------------------------------+
|                Song Title                  Key: G - 120  |
|                Artist                                    |
|  +----------------------------------------------------+  |
|  |             previous line (dim)                    |  |
|  |        >> CURRENT LINE (highlighted) <<            |  |
|  |             next line (dim)                        |  |
|  +----------------------------------------------------+  |
|  +--------+ +------+ +-----------------------------+     |
|  |  NOW   | | NEXT | | |Am  |F     |C   |G    ...  |     |
|  |   Am   | |  F   | |  scrolling chord timeline   |     |
|  +--------+ +------+ +-----------------------------+     |
|  ==== progress ===================================  1:23 |
+----------------------------------------------------------+
```

## Features

- **Modern desktop GUI** (CustomTkinter) with drag-and-drop of files *or folders*, a queue
  with per-file status, a progress bar, a colour-coded log console, and a *Preview layout*
  button that renders a sample frame with your current settings.
- **Automatic metadata**: ID3/Vorbis/MP4 tags, then filename parsing
  (`Artist - Title.mp3`, `01. Artist - Title (Official Video).mp3`), then optional
  AcoustID fingerprinting.
- **Lyrics**: sidecar `.lrc`/`.txt` files, then [lrclib.net](https://lrclib.net)
  (synced + plain), then the `syncedlyrics` aggregator (Musixmatch, NetEase, Megalobiz).
  Enhanced LRC word timings are used for word-accurate highlighting when available.
  A provider's "instrumental" flag is only trusted if no provider has lyrics for the track.
- **Chords**, three sources in order of preference:
  1. A chord sheet next to the song (`Song.chords.txt`, `.cho`, `.crd`, ChordPro or
     Ultimate-Guitar style) aligned to the synced lyrics.
  2. Ultimate Guitar chord sheets (opt-in, best effort, see *Limitations*).
  3. **Local audio analysis with librosa**: harmonic separation, CQT chroma, beat
     tracking, template matching against major/minor (optionally 7th) chords, key
     estimation, and Viterbi smoothing. Works fully offline.
- **Fast renderer**: frames are drawn with Pillow and piped straight into FFmpeg
  (no MoviePy, no temp images). Hardware encoders (NVENC / QSV / AMF) are selectable.
- **Backgrounds**: solid colour, two-colour gradient, or a looping video file with an
  adjustable darkening layer.
- **Output presets**: 720p, 1080p, 1440p, 4K, and vertical 1080x1920 for phones.
- Results are cached, so re-rendering a batch with a new look does not re-download or
  re-analyse anything.
- Headless CLI for scripting.

## Requirements

- Python 3.9 - 3.12 (3.12 recommended; librosa/numba wheels exist for these)
- FFmpeg (see below - a binary is downloaded automatically by `imageio-ffmpeg`)
- Windows, macOS or Linux. Tkinter must be available (it is in the python.org
  installers; on Debian/Ubuntu run `sudo apt install python3-tk`).

## Setup

```bash
git clone <this repo>
cd Lyric+Chord
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies and launch:

```bash
pip install -r requirements.txt
python main.py
```

### FFmpeg

The renderer needs the `ffmpeg` executable. It is found in this order:

1. The `LYRICCHORD_FFMPEG` environment variable pointing to the executable.
2. `ffmpeg` on your `PATH`.
3. The static build bundled by the `imageio-ffmpeg` package (installed by
   `requirements.txt`). This is the zero-configuration default and includes `libx264`.

If you would rather use a system install (for example to get NVENC/QSV/AMF hardware
encoders or `libx265`):

- **Windows**: download a build from <https://www.gyan.dev/ffmpeg/builds/> or
  <https://github.com/BtbN/FFmpeg-Builds/releases>, unzip, and add the `bin` folder to
  your `PATH` (or `winget install Gyan.FFmpeg`).
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg`.

The header of the app window shows which FFmpeg version was found.

### Optional: AcoustID fingerprinting

Only needed for files with no usable tags *and* an unhelpful filename.

```bash
pip install pyacoustid
```

Install the `fpcalc` binary from <https://acoustid.org/chromaprint> (put it on your
`PATH`), get a free API key at <https://acoustid.org/> and paste it into
*Settings > Lyrics and chords > AcoustID API key*.

## Using the app

1. **Add songs**: drag MP3 files or whole folders onto the window (or use *Add files* /
   *Add folder*). Sub-folders are scanned by default.
2. **Choose settings** in the right-hand panel. Press *Preview layout* at any time to see
   a sample frame with the current fonts, colours and background.
3. **Start batch**. The log shows each stage (`Processing 3 of 10: Song.mp3`, lyrics
   source, detected key/BPM, render progress). *Cancel* stops after the current step.
4. Finished videos are written to the output folder as `Artist - Title.mp4`.
   *Open output folder* opens it in Explorer/Finder.

### Sidecar files (recommended for best results)

Put these next to the audio file with the same base name:

| File | Purpose |
|------|---------|
| `Song.lrc` | Synced lyrics. Enhanced LRC (`<mm:ss.xx>` word tags) enables word-level highlighting. |
| `Song.txt` | Plain lyrics (spread evenly over the song) - or a chord sheet, detected automatically. |
| `Song.chords.txt` / `Song.cho` / `Song.crd` | Chord sheet in Ultimate-Guitar style (chords above lyrics) or ChordPro (`[Am]inline`). Aligned to the synced lyrics, so you get exact human-verified chords instead of detected ones. |

Example `Song.chords.txt`:

```
[Intro]
C  G  Am  F

[Verse 1]
C            G
Hello darkness my old friend
Am                  F
I've come to talk with you again
```

### Settings reference

| Group | Setting | Notes |
|-------|---------|-------|
| Output | Resolution, frame rate, encoder, preset, CRF | `libx264` + `medium` + CRF 20 is a good default. Use `veryfast` or a hardware encoder for speed. |
| Background | Style, theme preset, colours, gradient direction, loop video, darken | Loop mode composites your text over any video file that FFmpeg can read. |
| Typography | Font (curated system fonts or *Browse* for any TTF/OTF), sizes, accent/text/panel colours, panel opacity | Sizes are relative to 1080p and scale with resolution. |
| Layout | Chord timeline, progress bar, key/BPM badge, karaoke fill, timeline look-ahead | |
| Lyrics and chords | Lyrics offset, online chords, snap-to-key, flats, 7th chords, minimum chord length, AcoustID key | *Snap to key* strongly improves detection on real recordings. |

Settings are saved to `~/.lyricchord/settings.json`.

## Command line

```bash
python main.py --cli "C:\Music\Album" extra_song.mp3 --output "C:\Videos\out" --resolution 1080p --encoder h264_nvenc
```

Run `python main.py --cli --help` for all options (`--fps`, `--background`, `--loop-video`,
`--preset`, `--online-chords`, `--no-cache`, `--skip-existing`, `--lyrics-offset`, `-v`).

## How it works

```
audio file
  |-- metadata.py   tags -> filename -> AcoustID            => SongInfo
  |-- lyrics.py     sidecar -> lrclib -> syncedlyrics       => Lyrics (timed lines)
  |-- chords/       sidecar sheet -> Ultimate Guitar -> librosa
  |                   sheet.py  parses + aligns chord sheets to lyric timestamps
  |                   local.py  chroma + beats + templates + Viterbi
  |-- render/frames.py   Pillow draws one frame per timestamp (static layer cached)
  '-- render/renderer.py raw frames -> FFmpeg stdin -> H.264 + AAC MP4
```

Project layout:

```
main.py                     entry point (GUI, or --cli)
lyricchord/
  config.py                 Settings dataclass + JSON persistence + presets
  models.py                 SongInfo, Lyrics, ChordTrack, SongData
  gui/app.py                main window, drag & drop, batch thread wiring
  gui/settings_panel.py     scrollable settings form bound to Settings
  gui/log_console.py        colour-coded log widget
  pipeline/metadata.py      tag reading, filename parsing, AcoustID
  pipeline/lyrics.py        LRC parsing, lrclib + syncedlyrics providers
  pipeline/chords/theory.py chord templates, key finding, chord spelling
  pipeline/chords/local.py  librosa chord detection
  pipeline/chords/sheet.py  chord sheet parsing + lyric alignment
  pipeline/chords/online.py Ultimate Guitar fetcher (opt-in)
  pipeline/processor.py     per-song orchestration + BatchRunner thread
  pipeline/cache.py         JSON cache of fetched/analysed data
  render/frames.py          frame composition (layout, text, timeline)
  render/renderer.py        FFmpeg command building and piping
  utils/                    ffmpeg discovery, audio decoding, fonts, text helpers
tests/                      pytest suite (parsers, theory, synthetic-audio detection, render)
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite synthesises a C-G-Am-F progression, checks that the detector recovers it in
order, and renders short MP4s (including loop-background mode) with the bundled FFmpeg.

## Performance tips

- Rendering speed is dominated by the encoder. 1080p30 with `libx264 medium` runs at
  roughly real time or faster on a modern CPU; `veryfast` or `h264_nvenc` is several
  times faster.
- Chord analysis takes about 10-30 s per song and is cached.
- 4K output is 4x the pixels of 1080p; expect proportionally longer renders.

## Limitations and notes

- **Chord detection is an estimate.** Template matching on chroma is good at triads in
  pop/rock/folk and weaker on dense jazz voicings, heavy distortion or inversions.
  Turn on *7th chords* only if the material really uses them. For perfect results,
  supply a chord sheet sidecar.
- **Plain (unsynced) lyrics** are spread evenly across the song; the video notes that
  timing is approximate. Synced lyrics are available for most popular songs via lrclib.
- **Ultimate Guitar** has no public API. The optional fetcher parses their page
  markup and may stop working or be blocked at any time; it is off by default, falls
  back to local analysis on any failure, and you are responsible for complying with the
  site's terms of use.
- Lyrics and chord sheets are copyrighted works. Use the generated videos for personal
  practice or with the appropriate licences.

## License

MIT - see `LICENSE`.
