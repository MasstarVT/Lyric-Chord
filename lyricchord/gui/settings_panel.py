"""
Settings panel: a scrollable frame of grouped controls bound to a `Settings` object.

Every control writes into a Tk variable keyed by the Settings field name, so
`apply_to()` can copy values back generically using the dataclass type hints.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog
from typing import Callable, Dict, Optional, get_type_hints

import customtkinter as ctk

from ..config import (BACKGROUND_STYLES, ENCODERS, FPS_OPTIONS, GRADIENT_DIRECTIONS, RESOLUTIONS,
                      THEME_PRESETS, Settings)
from ..utils.fonts import find_named_fonts

X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
AUTO_FONT = "Auto (system default)"
# file stem -> display name, for the curated font dropdown
POPULAR_FONTS = {
    "segoeui": "Segoe UI", "arial": "Arial", "calibri": "Calibri", "verdana": "Verdana",
    "tahoma": "Tahoma", "georgia": "Georgia", "impact": "Impact", "trebuc": "Trebuchet MS",
    "bahnschrift": "Bahnschrift", "consola": "Consolas", "Roboto-Regular": "Roboto",
    "Montserrat-Regular": "Montserrat", "OpenSans-Regular": "Open Sans", "Helvetica": "Helvetica",
    "Arial": "Arial", "DejaVuSans": "DejaVu Sans", "LiberationSans-Regular": "Liberation Sans",
    "NotoSans-Regular": "Noto Sans",
}


def _contrast_text(hex_color: str) -> str:
    v = hex_color.lstrip("#")
    try:
        r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except (ValueError, IndexError):
        return "#ffffff"
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


class ColorButton(ctk.CTkButton):
    """Button that shows a colour swatch and opens the system colour chooser."""

    def __init__(self, master, variable: tk.StringVar, **kwargs):
        super().__init__(master, text="", width=120, command=self._pick, **kwargs)
        self.var = variable
        self.var.trace_add("write", lambda *_: self._refresh())
        self._refresh()

    def _refresh(self) -> None:
        color = self.var.get().strip() or "#000000"
        try:
            self.configure(fg_color=color, hover_color=color, text=color, text_color=_contrast_text(color))
        except tk.TclError:
            pass

    def _pick(self) -> None:
        result = colorchooser.askcolor(color=self.var.get() or None, parent=self)
        if result and result[1]:
            self.var.set(result[1])


class SettingsPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, settings: Settings, **kwargs):
        super().__init__(master, label_text="Settings", **kwargs)
        self.settings = settings
        self.vars: Dict[str, tk.Variable] = {}
        self._types = get_type_hints(Settings)
        self._font_choices: Dict[str, str] = {}
        self._row = 0
        self.grid_columnconfigure(1, weight=1)
        self._build()
        self.load_from(settings)

    # ------------------------------------------------------------------ builders
    def _var(self, name: str, kind) -> tk.Variable:
        v = kind()
        self.vars[name] = v
        return v

    def _section(self, title: str) -> None:
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=self._row, column=0, columnspan=3, sticky="w", padx=6, pady=(14, 4))
        self._row += 1

    def _add(self, label: str, widget, extra=None) -> None:
        ctk.CTkLabel(self, text=label, anchor="w").grid(row=self._row, column=0, sticky="w", padx=(6, 10), pady=3)
        widget.grid(row=self._row, column=1, sticky="ew", pady=3)
        if extra is not None:
            extra.grid(row=self._row, column=2, padx=(6, 6))
        self._row += 1

    def _option(self, name: str, label: str, values, command=None) -> ctk.CTkOptionMenu:
        var = self._var(name, tk.StringVar)
        menu = ctk.CTkOptionMenu(self, values=[str(v) for v in values], variable=var, command=command)
        self._add(label, menu)
        return menu

    def _check(self, name: str, label: str) -> None:
        var = self._var(name, tk.BooleanVar)
        ctk.CTkCheckBox(self, text=label, variable=var).grid(
            row=self._row, column=0, columnspan=3, sticky="w", padx=6, pady=3)
        self._row += 1

    def _entry(self, name: str, label: str, browse: Optional[Callable[[], None]] = None,
               secret: bool = False) -> None:
        var = self._var(name, tk.StringVar)
        entry = ctk.CTkEntry(self, textvariable=var, show="*" if secret else "")
        extra = ctk.CTkButton(self, text="Browse...", width=80, command=browse) if browse else None
        self._add(label, entry, extra)

    def _slider(self, name: str, label: str, lo: float, hi: float, steps: int,
                fmt: Callable[[float], str]) -> None:
        var = self._var(name, tk.DoubleVar)
        frame = ctk.CTkFrame(self, fg_color="transparent")
        slider = ctk.CTkSlider(frame, from_=lo, to=hi, number_of_steps=steps, variable=var)
        slider.pack(side="left", fill="x", expand=True)
        value = ctk.CTkLabel(frame, text="", width=56, anchor="e")
        value.pack(side="left", padx=(8, 0))
        var.trace_add("write", lambda *_: value.configure(text=fmt(var.get())))
        self._add(label, frame)

    def _color(self, name: str, label: str) -> None:
        var = self._var(name, tk.StringVar)
        self._add(label, ColorButton(self, var))

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self._section("Output")
        self._entry("output_dir", "Output folder", browse=self._browse_output)
        self._option("resolution", "Resolution", list(RESOLUTIONS))
        self._option("fps", "Frame rate", FPS_OPTIONS)
        self._option("encoder", "Encoder", ENCODERS)
        self._option("preset", "x264/x265 preset", X264_PRESETS)
        self._slider("crf", "Quality (CRF, lower = better)", 14, 32, 18, lambda v: f"{int(v)}")
        self._check("skip_existing", "Skip songs that already have a video")
        self._check("use_cache", "Cache lyrics and chords between runs")

        self._section("Background")
        self._option("background_style", "Style", BACKGROUND_STYLES)
        self.theme_var = tk.StringVar(value="Midnight")
        theme = ctk.CTkOptionMenu(self, values=list(THEME_PRESETS), variable=self.theme_var,
                                  command=self._apply_theme)
        self._add("Colour theme preset", theme)
        self._color("bg_color", "Solid colour")
        self._color("gradient_start", "Gradient start")
        self._color("gradient_end", "Gradient end")
        self._option("gradient_direction", "Gradient direction", GRADIENT_DIRECTIONS)
        self._entry("loop_video_path", "Loop video (mp4/mov)", browse=self._browse_loop)
        self._slider("loop_dim", "Darken loop video", 0.0, 0.9, 18, lambda v: f"{int(v * 100)}%")

        self._section("Typography and colours")
        self.font_var = tk.StringVar(value=AUTO_FONT)
        self.vars["font_path"] = tk.StringVar()
        fonts = find_named_fonts(POPULAR_FONTS)
        self._font_choices = {label: fonts[stem] for stem, label in POPULAR_FONTS.items() if stem in fonts}
        menu = ctk.CTkOptionMenu(self, values=[AUTO_FONT] + sorted(self._font_choices),
                                 variable=self.font_var, command=self._font_selected)
        self._add("Font", menu, ctk.CTkButton(self, text="Browse...", width=80, command=self._browse_font))
        self._slider("title_size", "Title size", 30, 90, 60, lambda v: f"{int(v)}")
        self._slider("lyric_size", "Lyric size", 36, 100, 64, lambda v: f"{int(v)}")
        self._slider("context_lyric_size", "Context lyric size", 24, 64, 40, lambda v: f"{int(v)}")
        self._slider("chord_size", "Chord size", 60, 200, 140, lambda v: f"{int(v)}")
        self._color("accent_color", "Accent (chords / highlight)")
        self._color("text_color", "Text")
        self._color("dim_text_color", "Dim text")
        self._color("panel_color", "Panels")
        self._slider("panel_alpha", "Panel opacity", 0, 255, 51, lambda v: f"{int(v / 255 * 100)}%")
        self._check("show_chord_timeline", "Show scrolling chord timeline")
        self._check("show_progress_bar", "Show progress bar")
        self._check("show_key_bpm", "Show key and BPM")
        self._check("karaoke_fill", "Karaoke-style fill on the active lyric line")
        self._slider("timeline_window_sec", "Timeline look-ahead", 4, 16, 12, lambda v: f"{v:.0f}s")

        self._section("Lyrics and chords")
        self._entry("lyrics_offset_ms", "Lyrics offset (ms, + = later)")
        self._check("use_online_chords", "Try Ultimate Guitar chord sheets first (experimental)")
        self._check("snap_chords_to_key", "Bias detected chords toward the song key")
        self._check("prefer_flats", "Use flats in flat keys (Bb instead of A#)")
        self._check("include_seventh_chords", "Detect 7th chords (7, m7, maj7)")
        self._slider("min_chord_seconds", "Minimum chord length", 0.2, 2.0, 18, lambda v: f"{v:.1f}s")
        self._entry("acoustid_api_key", "AcoustID API key (optional)", secret=True)
        self._check("scan_subfolders", "Include sub-folders when a folder is dropped")

    # ------------------------------------------------------------------ callbacks
    def _browse_output(self) -> None:
        d = filedialog.askdirectory(parent=self, initialdir=self.vars["output_dir"].get() or None)
        if d:
            self.vars["output_dir"].set(d)

    def _browse_loop(self) -> None:
        f = filedialog.askopenfilename(parent=self, title="Choose a background video",
                                       filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.avi"), ("All files", "*.*")])
        if f:
            self.vars["loop_video_path"].set(f)
            self.vars["background_style"].set("loop")

    def _browse_font(self) -> None:
        f = filedialog.askopenfilename(parent=self, title="Choose a font",
                                       filetypes=[("Fonts", "*.ttf *.otf *.ttc"), ("All files", "*.*")])
        if f:
            self.vars["font_path"].set(f)
            self.font_var.set(f"Custom: {Path(f).name}")

    def _font_selected(self, label: str) -> None:
        self.vars["font_path"].set(self._font_choices.get(label, ""))

    def _apply_theme(self, name: str) -> None:
        for key, value in THEME_PRESETS.get(name, {}).items():
            if key in self.vars:
                self.vars[key].set(value)

    # ------------------------------------------------------------------ data binding
    def load_from(self, settings: Settings) -> None:
        for name, var in self.vars.items():
            value = getattr(settings, name, None)
            if value is None:
                continue
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, tk.DoubleVar):
                var.set(float(value))
            else:
                var.set(str(value))
        # Font dropdown label.
        path = settings.font_path
        label = next((k for k, v in self._font_choices.items() if v == path), None)
        self.font_var.set(label or (f"Custom: {Path(path).name}" if path else AUTO_FONT))

    def get_bool(self, name: str) -> bool:
        return bool(self.vars[name].get())

    def apply_to(self, settings: Settings) -> Settings:
        """Copy widget values into `settings`, converting to each field's declared type."""
        for name, var in self.vars.items():
            raw = var.get()
            kind = self._types.get(name, str)
            try:
                if kind is bool:
                    value = bool(raw)
                elif kind is int:
                    value = int(round(float(raw)))
                elif kind is float:
                    value = float(raw)
                else:
                    value = str(raw).strip()
            except (TypeError, ValueError):
                continue  # keep the previous value on bad input
            setattr(settings, name, value)
        return settings
