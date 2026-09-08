"""
Audio decoding via FFmpeg.

We decode with FFmpeg rather than `librosa.load` so MP3/M4A/OGG all work
without relying on libsndfile's codec support or the deprecated audioread path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .ffmpeg import find_ffmpeg, no_window_flag

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus", ".aac", ".wma"}


def decode_audio(path: Path, sr: int = 22050) -> np.ndarray:
    """Decode an audio file to a mono float32 numpy array at sample rate `sr`."""
    cmd = [
        find_ffmpeg(), "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, creationflags=no_window_flag())
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")[:400]
        raise RuntimeError(f"FFmpeg failed to decode {path.name}: {err}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def probe_duration(path: Path) -> float:
    """Duration in seconds parsed from FFmpeg's banner (fallback when tags are missing)."""
    cmd = [find_ffmpeg(), "-i", str(path), "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=no_window_flag())
    # FFmpeg prints "Duration: 00:03:45.12, start: ..." to stderr.
    for line in proc.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            hms = line.split()[1].rstrip(",")
            try:
                h, m, s = hms.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                return 0.0
    return 0.0
