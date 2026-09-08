"""
Headless command-line interface (useful for scripting and servers without a display).

    python main.py --cli "C:/Music/Album" song.mp3 --output "C:/Videos/out" --resolution 720p
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import List, Optional

from .config import RESOLUTIONS, Settings
from .errors import Cancelled
from .pipeline.processor import find_audio_files, process_song


def _resolution_choice(text: str) -> str:
    for name in RESOLUTIONS:
        if text.lower() in name.lower():
            return name
    raise argparse.ArgumentTypeError(f"unknown resolution '{text}'. Choose from: {', '.join(RESOLUTIONS)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lyricchord", description="Generate play-along lyric + chord videos.")
    p.add_argument("inputs", nargs="+", help="audio files and/or folders")
    p.add_argument("-o", "--output", help="output folder (default: saved setting)")
    p.add_argument("-r", "--resolution", type=_resolution_choice, help="720p | 1080p | 1440p | 4K | Vertical")
    p.add_argument("--fps", type=int, choices=[24, 30, 60])
    p.add_argument("--background", choices=["solid", "gradient", "loop"])
    p.add_argument("--loop-video", help="video file to loop behind the text (implies --background loop)")
    p.add_argument("--encoder", help="libx264 (default), h264_nvenc, h264_qsv, h264_amf, libx265")
    p.add_argument("--preset", help="x264 preset, e.g. veryfast / medium / slow")
    p.add_argument("--online-chords", action="store_true", help="try Ultimate Guitar chord sheets first")
    p.add_argument("--no-cache", action="store_true", help="ignore cached lyrics / chords")
    p.add_argument("--skip-existing", action="store_true", help="skip songs whose MP4 already exists")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    p.add_argument("--lyrics-offset", type=int, metavar="MS", help="shift lyrics by milliseconds")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("numba", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("lyricchord")

    settings = Settings.load()
    if args.output:
        settings.output_dir = args.output
    if args.resolution:
        settings.resolution = args.resolution
    if args.fps:
        settings.fps = args.fps
    if args.background:
        settings.background_style = args.background
    if args.loop_video:
        settings.loop_video_path = args.loop_video
        settings.background_style = "loop"
    if args.encoder:
        settings.encoder = args.encoder
    if args.preset:
        settings.preset = args.preset
    if args.online_chords:
        settings.use_online_chords = True
    if args.no_cache:
        settings.use_cache = False
    if args.skip_existing:
        settings.skip_existing = True
    if args.lyrics_offset is not None:
        settings.lyrics_offset_ms = args.lyrics_offset

    files = find_audio_files(args.inputs, recursive=not args.no_recursive)
    if not files:
        log.error("No audio files found in: %s", ", ".join(args.inputs))
        return 2
    log.info("%d file(s) queued -> %s", len(files), settings.output_dir)

    cancel = threading.Event()
    ok = failed = 0
    last_stage = {"name": ""}

    def progress(stage: str, frac: float, msg: str) -> None:
        if stage != last_stage["name"]:
            last_stage["name"] = stage
            log.info("  %s", msg)

    try:
        for i, f in enumerate(files, 1):
            log.info("Processing %d of %d: %s", i, len(files), f.name)
            try:
                out = process_song(f, settings, progress, cancel)
                ok += 1
                log.info("  -> %s", out)
            except Cancelled:
                break
            except Exception as exc:
                failed += 1
                log.error("  failed: %s", exc, exc_info=args.verbose)
    except KeyboardInterrupt:
        cancel.set()
        log.warning("Interrupted")
    log.info("Done: %d succeeded, %d failed", ok, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
