"""
Audio decoding via FFmpeg.

We decode with FFmpeg rather than `librosa.load` so MP3/M4A/OGG all work
without relying on libsndfile's codec support or the deprecated audioread path.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .ffmpeg import find_ffmpeg, no_window_flag

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus", ".aac", ".wma"}
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def decode_audio(path: Path, sr: int = 22050, start: Optional[float] = None,
                 duration: Optional[float] = None) -> np.ndarray:
    """Decode an audio file (or a window of it) to mono float32 at sample rate `sr`.

    `start`/`duration` are seconds; FFmpeg seeks and stops itself, so a 10 s window of
    a 6-minute file costs a fraction of a full decode.
    """
    cmd = [find_ffmpeg(), "-v", "error"]
    if start:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{max(0.0, duration):.3f}"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=no_window_flag())
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")[:400]
        raise RuntimeError(f"FFmpeg failed to decode {path.name}: {err}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def probe_duration(path: Path) -> float:
    """Duration in seconds from FFmpeg's input banner (a header read, not a decode)."""
    # With no output specified FFmpeg prints the banner and exits non-zero immediately.
    cmd = [find_ffmpeg(), "-hide_banner", "-i", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=no_window_flag())
    m = _DURATION_RE.search(proc.stderr or "")
    if not m:
        return 0.0
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)
