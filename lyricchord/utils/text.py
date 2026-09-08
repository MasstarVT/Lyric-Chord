"""Small text helpers used by several pipeline stages."""

from __future__ import annotations

import re
import unicodedata

# Bracketed noise commonly found in downloaded filenames / YouTube rips.
_PAREN_NOISE = re.compile(
    r"[\(\[\{]\s*(official|lyrics?|lyric video|audio|video|hd|hq|remaster(ed)?( \d{4})?|"
    r"live|explicit|clean|mono|stereo|visuali[sz]er|4k|1080p|from .*)[^\)\]\}]*[\)\]\}]",
    re.IGNORECASE,
)
_TRACK_PREFIX = re.compile(r"^\s*\d{1,3}\s*[-._)\]]?\s+")


def clean_title(text: str) -> str:
    """Remove '(Official Video)'-style noise and leading track numbers."""
    text = _PAREN_NOISE.sub("", text)
    text = _TRACK_PREFIX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -_.")
    return text


def normalize(text: str) -> str:
    """Lower-case, strip accents and punctuation. Used for fuzzy matching."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "vs"}


def smart_title_case(text: str) -> str:
    """'eye in the sky' -> 'Eye in the Sky'. Leaves mixed-case input untouched."""
    if text != text.lower():
        return text
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if 0 < i < len(words) - 1 and w in _SMALL_WORDS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def safe_filename(name: str, max_len: int = 120) -> str:
    """Make a string safe to use as a filename on Windows/macOS/Linux."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return (name or "untitled")[:max_len]
