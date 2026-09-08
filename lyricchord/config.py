"""
User settings.

`Settings` is a plain dataclass with sensible defaults. It is persisted as JSON in
the user's home directory so the GUI remembers choices between runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger("lyricchord")

CONFIG_DIR = Path.home() / ".lyricchord"
CONFIG_FILE = CONFIG_DIR / "settings.json"

# Resolution presets: name -> (width, height)
RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
    "1440p (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
    "Vertical 1080x1920": (1080, 1920),
}

BACKGROUND_STYLES = ["solid", "gradient", "loop"]
GRADIENT_DIRECTIONS = ["vertical", "horizontal", "diagonal"]
ENCODERS = ["libx264", "h264_nvenc", "h264_qsv", "h264_amf", "libx265"]
FPS_OPTIONS = [24, 30, 60]

# Colour presets applied by the GUI "Theme" dropdown.
THEME_PRESETS: Dict[str, Dict[str, str]] = {
    "Midnight": {
        "bg_color": "#0f172a", "gradient_start": "#0f172a", "gradient_end": "#1e3a8a",
        "accent_color": "#38bdf8", "text_color": "#f8fafc", "dim_text_color": "#94a3b8",
        "panel_color": "#0b1220",
    },
    "Sunset": {
        "bg_color": "#2a0a1e", "gradient_start": "#3b0d3a", "gradient_end": "#c2410c",
        "accent_color": "#fbbf24", "text_color": "#fff7ed", "dim_text_color": "#fdba74",
        "panel_color": "#1c0a14",
    },
    "Forest": {
        "bg_color": "#052e16", "gradient_start": "#052e16", "gradient_end": "#166534",
        "accent_color": "#a3e635", "text_color": "#f0fdf4", "dim_text_color": "#86efac",
        "panel_color": "#02170b",
    },
    "Mono": {
        "bg_color": "#111111", "gradient_start": "#000000", "gradient_end": "#333333",
        "accent_color": "#ffffff", "text_color": "#ffffff", "dim_text_color": "#8a8a8a",
        "panel_color": "#000000",
    },
}


@dataclass
class Settings:
    # --- Output -----------------------------------------------------------
    output_dir: str = str(Path.home() / "Videos" / "LyricChord")
    resolution: str = "1080p (1920x1080)"
    fps: int = 30
    encoder: str = "libx264"
    crf: int = 20                       # quality for software encoders (lower = better)
    preset: str = "medium"              # x264/x265 speed preset
    audio_bitrate: str = "192k"

    # --- Background -------------------------------------------------------
    background_style: str = "gradient"  # solid | gradient | loop
    bg_color: str = "#0f172a"
    gradient_start: str = "#0f172a"
    gradient_end: str = "#1e3a8a"
    gradient_direction: str = "diagonal"
    loop_video_path: str = ""           # used when background_style == "loop"
    loop_dim: float = 0.45              # darken the loop video so text stays legible

    # --- Typography / colours --------------------------------------------
    font_path: str = ""                 # "" = auto-detect a system font (bold variant is inferred)
    title_size: int = 56
    lyric_size: int = 64
    context_lyric_size: int = 42
    chord_size: int = 120
    accent_color: str = "#38bdf8"
    text_color: str = "#f8fafc"
    dim_text_color: str = "#94a3b8"
    panel_color: str = "#0b1220"
    panel_alpha: int = 150              # 0..255 opacity of the lyric/chord panels

    # --- Layout toggles ---------------------------------------------------
    show_chord_timeline: bool = True
    show_progress_bar: bool = True
    show_key_bpm: bool = True
    karaoke_fill: bool = True           # progressive colour fill of the active line
    timeline_window_sec: float = 8.0    # seconds of upcoming chords visible in the lane

    # --- Pipeline ---------------------------------------------------------
    lyrics_offset_ms: int = 0           # global shift applied to lyric timestamps
    use_online_chords: bool = False     # try Ultimate Guitar chord sheets first
    snap_chords_to_key: bool = True     # bias local detection toward diatonic chords
    prefer_flats: bool = True           # spell chords with flats in flat keys
    min_chord_seconds: float = 0.5      # merge shorter detected segments
    include_seventh_chords: bool = False
    acoustid_api_key: str = ""          # optional; enables fingerprinting fallback
    scan_subfolders: bool = True
    use_cache: bool = True              # reuse fetched lyrics/chords between runs
    skip_existing: bool = False         # skip songs whose MP4 already exists

    # --- Window -----------------------------------------------------------
    appearance_mode: str = "dark"       # customtkinter appearance

    # ------------------------------------------------------------------ API
    @property
    def size(self) -> Tuple[int, int]:
        return RESOLUTIONS.get(self.resolution, (1920, 1080))

    def loop_video(self) -> Optional[Path]:
        """The loop background file if that mode is active and the file exists, else None.

        Shared by the renderer and the layout preview so both agree on which mode runs.
        """
        if self.background_style != "loop" or not self.loop_video_path:
            return None
        p = Path(self.loop_video_path)
        return p if p.is_file() else None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Settings":
        valid = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in d.items() if k in valid})

    def save(self, path: Path = CONFIG_FILE) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues
            log.warning("Could not save settings: %s", exc)

    @staticmethod
    def load(path: Path = CONFIG_FILE) -> "Settings":
        if path.exists():
            try:
                return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as exc:
                log.warning("Could not read settings (%s); using defaults", exc)
        return Settings()
