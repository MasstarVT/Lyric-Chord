"""
Song identification.

Resolution order:
  1. Embedded tags (ID3 for MP3, Vorbis comments for FLAC/OGG, iTunes atoms for M4A)
  2. Filename parsing ("Artist - Title.mp3", "01. Artist - Title.mp3", ...)
  3. AcoustID fingerprint lookup (optional; needs an API key and the fpcalc binary)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from ..config import Settings
from ..models import SongInfo
from ..utils.audio import probe_duration
from ..utils.text import clean_title

log = logging.getLogger("lyricchord")

# Separators that commonly divide artist from title in filenames.
_SEPARATORS = [" - ", " – ", " — ", "_-_", " -- ", " _ "]


def parse_filename(stem: str) -> Tuple[str, str]:
    """Guess (artist, title) from a filename stem. Artist is '' when unknown."""
    text = clean_title(stem)
    for sep in _SEPARATORS:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) >= 2:
                artist = parts[0]
                title = clean_title(" - ".join(parts[1:]))
                return artist, title
    return "", text.replace("_", " ").strip()


def read_tags(path: Path) -> Tuple[str, str, str, float]:
    """Return (title, artist, album, duration_seconds) from embedded tags."""
    try:
        from mutagen import File as MutagenFile  # type: ignore
    except ImportError:  # pragma: no cover
        log.warning("mutagen is not installed; tag reading disabled")
        return "", "", "", 0.0

    try:
        audio = MutagenFile(str(path), easy=True)
    except Exception as exc:  # corrupt file, unsupported container...
        log.debug("mutagen could not open %s: %s", path.name, exc)
        return "", "", "", 0.0
    if audio is None:
        return "", "", "", 0.0

    def first(key: str) -> str:
        if not audio.tags:
            return ""
        value = audio.tags.get(key)
        if isinstance(value, list):
            value = value[0] if value else ""
        return str(value or "").strip()

    duration = float(getattr(audio.info, "length", 0.0) or 0.0)
    return first("title"), first("artist"), first("album"), duration


def identify_acoustid(path: Path, api_key: str) -> Optional[Tuple[str, str]]:
    """Look the song up by audio fingerprint. Returns (artist, title) or None."""
    try:
        import acoustid  # type: ignore
    except ImportError:
        log.info("pyacoustid not installed; skipping fingerprint lookup")
        return None
    try:
        for score, _rid, title, artist in acoustid.match(api_key, str(path)):
            if title and artist and score >= 0.5:
                return artist, title
    except acoustid.FingerprintGenerationError:
        log.warning("AcoustID: fpcalc (Chromaprint) not found; cannot fingerprint")
    except acoustid.WebServiceError as exc:
        log.warning("AcoustID web service error: %s", exc)
    except Exception as exc:  # pragma: no cover
        log.warning("AcoustID lookup failed: %s", exc)
    return None


def extract_metadata(path: Path, settings: Settings) -> SongInfo:
    """Resolve title/artist/duration for one audio file."""
    title, artist, album, duration = read_tags(path)
    source = "id3" if (title and artist) else ""

    if not title or not artist:
        f_artist, f_title = parse_filename(path.stem)
        title = title or f_title
        artist = artist or f_artist
        source = source or "filename"

    if (not artist or not title) and settings.acoustid_api_key:
        hit = identify_acoustid(path, settings.acoustid_api_key)
        if hit:
            artist, title = hit
            source = "acoustid"

    if duration <= 0:
        duration = probe_duration(path)

    title = clean_title(title) or path.stem
    info = SongInfo(path=path, title=title, artist=artist.strip(), album=album,
                    duration=duration, source=source or "filename")
    log.info("Identified '%s' by '%s' (%s, %.0fs)", info.title, info.display_artist,
             info.source, info.duration)
    return info
