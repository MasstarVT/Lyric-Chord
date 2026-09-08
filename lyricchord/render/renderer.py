"""
Video encoding.

Frames are drawn with Pillow and piped as raw pixels into a single FFmpeg process
that encodes H.264 and muxes the original audio. This is several times faster than
MoviePy for text-heavy videos and needs no temporary image files.

Two modes:
  * solid / gradient background -> frames are RGB, background baked in
  * loop background             -> frames are RGBA overlays; FFmpeg loops the user's
                                   video underneath and composites with `overlay`
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional

from ..config import Settings
from ..errors import Cancelled
from ..models import SongData
from ..utils.ffmpeg import encoder_available, find_ffmpeg, no_window_flag
from .frames import FrameComposer

log = logging.getLogger("lyricchord")

ProgressFn = Optional[Callable[[float], None]]


def video_codec_args(settings: Settings) -> List[str]:
    enc = settings.encoder
    if enc != "libx264" and not encoder_available(enc):
        log.warning("Encoder %s is not available in this FFmpeg build; using libx264", enc)
        enc = "libx264"
    q = str(int(settings.crf))
    if enc == "libx265":
        return ["-c:v", "libx265", "-preset", settings.preset, "-crf", q, "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", q, "-pix_fmt", "nv12"]
    if enc == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", q, "-qp_p", q, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", settings.preset, "-crf", q, "-profile:v", "high", "-pix_fmt", "yuv420p"]


def build_command(settings: Settings, width: int, height: int, fps: int, audio: Path,
                  out: Path, loop_video: Optional[Path]) -> List[str]:
    """Assemble the FFmpeg command line. Frames arrive on stdin."""
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if loop_video is not None:
        cmd += ["-stream_loop", "-1", "-i", str(loop_video)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps),
                "-thread_queue_size", "1024", "-i", "pipe:0"]
        cmd += ["-i", str(audio)]
        graph = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                 f"crop={width}:{height},setsar=1,fps={fps}[bg];"
                 f"[bg][1:v]overlay=0:0:shortest=1:format=auto[v]")
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "2:a:0"]
    else:
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                "-thread_queue_size", "1024", "-i", "pipe:0"]
        cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
    cmd += video_codec_args(settings)
    cmd += ["-c:a", "aac", "-b:a", settings.audio_bitrate, "-ar", "48000"]
    cmd += ["-r", str(fps), "-shortest", "-movflags", "+faststart", str(out)]
    return cmd


def render_video(song: SongData, settings: Settings, out_path: Path,
                 progress: ProgressFn = None, cancel: Optional[threading.Event] = None) -> Path:
    """Render the whole song to `out_path`. Raises Cancelled / RuntimeError."""
    width, height = settings.size
    fps = int(settings.fps)
    duration = song.info.duration
    total = max(1, math.ceil(duration * fps))

    loop_video: Optional[Path] = None
    if settings.background_style == "loop":
        cand = Path(settings.loop_video_path) if settings.loop_video_path else None
        if cand and cand.is_file():
            loop_video = cand
        else:
            log.warning("Loop background video not found; falling back to gradient")

    composer = FrameComposer(song, settings, width, height, transparent=loop_video is not None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    cmd = build_command(settings, width, height, fps, song.info.path, tmp, loop_video)
    log.debug("ffmpeg: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, creationflags=no_window_flag())
    assert proc.stdin is not None and proc.stderr is not None
    err_chunks: List[bytes] = []

    def drain() -> None:  # keep stderr from filling up and deadlocking ffmpeg
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            err_chunks.append(chunk)

    threading.Thread(target=drain, daemon=True).start()

    report_every = max(1, total // 200)
    try:
        for i in range(total):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            proc.stdin.write(composer.render(i / fps))
            if i % report_every == 0 and progress:
                progress(i / total)
        proc.stdin.close()
        rc = proc.wait()
    except Cancelled:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    except (BrokenPipeError, OSError):
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg stopped accepting frames: {_tail(err_chunks)}")

    if rc != 0:
        tmp.unlink(missing_ok=True)
        hint = " (try the libx264 encoder)" if settings.encoder != "libx264" else ""
        raise RuntimeError(f"FFmpeg exited with code {rc}{hint}: {_tail(err_chunks)}")

    tmp.replace(out_path)
    if progress:
        progress(1.0)
    log.info("Wrote %s (%dx%d @ %dfps, %d frames)", out_path.name, width, height, fps, total)
    return out_path


def _tail(chunks: List[bytes], n: int = 600) -> str:
    return b"".join(chunks).decode(errors="ignore").strip()[-n:] or "(no error output)"
