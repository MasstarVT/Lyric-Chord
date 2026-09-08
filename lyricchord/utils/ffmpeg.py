"""
FFmpeg discovery.

Resolution order:
  1. LYRICCHORD_FFMPEG environment variable (explicit path)
  2. `ffmpeg` on PATH
  3. Binary bundled with the `imageio-ffmpeg` package (installed via requirements)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Optional


class FFmpegNotFound(RuntimeError):
    pass


def no_window_flag() -> int:
    """On Windows, prevent a console window from flashing for each subprocess."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    env = os.environ.get("LYRICCHORD_FFMPEG")
    if env and os.path.isfile(env):
        return env

    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # ImportError or download failure
        pass

    raise FFmpegNotFound(
        "FFmpeg was not found. Install it and add it to PATH, set LYRICCHORD_FFMPEG "
        "to the executable, or `pip install imageio-ffmpeg`."
    )


def ffmpeg_version() -> Optional[str]:
    try:
        out = subprocess.run(
            [find_ffmpeg(), "-version"], capture_output=True, text=True, timeout=10,
            creationflags=no_window_flag(),
        )
        return out.stdout.splitlines()[0] if out.stdout else None
    except Exception:
        return None


@lru_cache(maxsize=None)
def encoder_available(name: str) -> bool:
    """Return True if `ffmpeg -encoders` lists the given encoder."""
    try:
        out = subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True,
            timeout=15, creationflags=no_window_flag(),
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == name:
                return True
        return False
    except Exception:
        return False
