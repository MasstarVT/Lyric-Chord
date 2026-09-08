"""
Font discovery for Pillow text rendering.

Pillow needs an actual TrueType/OpenType file path. We look for a user-chosen
font first, then a list of common families on Windows / macOS / Linux.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import ImageFont

# (regular, bold) filename candidates, in order of preference.
_CANDIDATES = [
    ("segoeui.ttf", "segoeuib.ttf"),
    ("arial.ttf", "arialbd.ttf"),
    ("calibri.ttf", "calibrib.ttf"),
    ("Helvetica.ttc", "Helvetica.ttc"),
    ("Arial.ttf", "Arial Bold.ttf"),
    ("SFNS.ttf", "SFNS.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
]


def font_dirs() -> List[Path]:
    dirs: List[Path] = []
    if sys.platform.startswith("win"):
        dirs.append(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
        dirs.append(Path.home() / "AppData/Local/Microsoft/Windows/Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
                 Path.home() / ".local/share/fonts"]
    return [d for d in dirs if d.is_dir()]


def find_named_fonts(stems: Iterable[str]) -> Dict[str, str]:
    """Map stem -> path for the given font files, probing exact paths only.

    Deliberately no directory walk: this runs on the GUI thread at startup and the
    Windows font folder holds thousands of files.
    """
    found: Dict[str, str] = {}
    dirs = font_dirs()
    for stem in stems:
        for d in dirs:
            for ext in (".ttf", ".otf", ".ttc"):
                p = d / f"{stem}{ext}"
                if p.exists():
                    found[stem] = str(p)
                    break
            if stem in found:
                break
    return found


def _find_file(name: str) -> Optional[str]:
    for d in font_dirs():
        p = d / name
        if p.exists():
            return str(p)
        # Linux fonts are usually nested in sub-directories.
        for hit in d.rglob(name):
            return str(hit)
    return None


@lru_cache(maxsize=1)
def default_font_paths() -> Tuple[Optional[str], Optional[str]]:
    """Return (regular_path, bold_path), or (None, None) if nothing was found."""
    for reg, bold in _CANDIDATES:
        r = _find_file(reg)
        if r:
            return r, (_find_file(bold) or r)
    return None, None


def guess_bold_variant(path: str) -> str:
    """Given 'foo.ttf', return 'foobd.ttf' / 'foo-Bold.ttf' etc. if it exists."""
    if not path:
        return ""
    p = Path(path)
    for suffix in ("bd", "b", "-Bold", "Bold", "_Bold", " Bold"):
        cand = p.with_name(p.stem + suffix + p.suffix)
        if cand.exists():
            return str(cand)
    return ""


def load_font(path: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font at `size`. Falls back to system defaults, then Pillow's bitmap font."""
    candidates = [path] if path else []
    reg, bld = default_font_paths()
    candidates.append(bld if bold else reg)
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    # Pillow >= 10.1 can scale its built-in font; older versions ignore size.
    try:
        return ImageFont.load_default(size=size)  # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()  # type: ignore[return-value]
