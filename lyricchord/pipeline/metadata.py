"""
Song identification.

Resolution order:
  1. Embedded tags (ID3 for MP3, Vorbis comments for FLAC/OGG, iTunes atoms for M4A)
  2. Filename parsing ("Artist - Title.mp3", "01. Artist - Title.mp3", ...)
  3. Title-only lookups when the artist is still unknown:
       a. lrclib search: the artist most of the lyric entries for this title agree on
       b. MusicBrainz recording search filtered by the file's duration, which also
          yields the exact recording title (e.g. "Sirius / Eye in the Sky" for the
          album edit that has an instrumental intro)
  4. AcoustID fingerprint lookup (optional; needs an API key and the fpcalc binary)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from ..config import Settings
from ..models import SongInfo
from ..utils.audio import probe_duration
from ..utils.net import HTTP_HEADERS
from ..utils.text import artist_key, clean_title, normalize, smart_title_case

log = logging.getLogger("lyricchord")

# Separators that commonly divide artist from title in filenames.
_SEPARATORS = [" - ", " – ", " — ", "_-_", " -- ", " _ "]

LRCLIB_SEARCH = "https://lrclib.net/api/search"
MUSICBRAINZ_RECORDING = "https://musicbrainz.org/ws/2/recording/"
MB_TIMEOUT = 12          # MusicBrainz can be slow or answer 503 when busy; never block a batch on it


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


# --------------------------------------------------------------------------- title-only lookups
def artist_consensus(results: List[dict], min_votes: int = 2, min_share: float = 0.4) -> Optional[str]:
    """Pick the artist most lrclib entries for a title agree on (pure function, testable)."""
    votes: Counter = Counter()
    display: Dict[str, Counter] = defaultdict(Counter)
    for item in results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("artistName") or "").strip()
        if not name or not (item.get("syncedLyrics") or item.get("plainLyrics")):
            continue
        key = artist_key(name)
        if not key:
            continue
        weight = 2 if item.get("syncedLyrics") else 1
        votes[key] += weight
        display[key][name] += 1
    if not votes:
        return None
    key, best = votes.most_common(1)[0]
    if best < min_votes or best / sum(votes.values()) < min_share:
        return None
    return display[key].most_common(1)[0][0]


def lrclib_artist_for_title(title: str) -> Optional[str]:
    try:
        r = requests.get(LRCLIB_SEARCH, params={"track_name": title}, headers=HTTP_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        return artist_consensus(data) if isinstance(data, list) else None
    except (requests.RequestException, ValueError) as exc:
        log.debug("lrclib artist lookup failed: %s", exc)
        return None


def _artist_credit(rec: dict) -> str:
    parts = []
    for credit in rec.get("artist-credit", []) or []:
        if isinstance(credit, dict):
            parts.append(str(credit.get("name", "")) + str(credit.get("joinphrase", "") or ""))
        elif isinstance(credit, str):
            parts.append(credit)
    return "".join(parts).strip()


def rank_musicbrainz(recordings: List[dict], duration: float, artist_hint: str = "") -> Optional[Tuple[str, str]]:
    """Choose (artist, title) from MusicBrainz search results (pure function, testable).

    Text score alone is useless for popular songs (every cover scores 100), so it is
    halved and combined with: a penalty for duration differences beyond the 2 s that
    different masters of the same recording normally show, a bonus for how many
    releases carry the recording (popularity), and a large bonus for agreeing with an
    artist hint from another source (lrclib consensus), which is the strongest signal.
    """
    hint = normalize(artist_hint) if artist_hint else ""
    best, best_score = None, float("-inf")
    for rec in recordings:
        length = (rec.get("length") or 0) / 1000.0
        if not length:
            continue
        delta = abs(length - duration)
        artist = _artist_credit(rec)
        score = (0.5 * float(rec.get("score") or 0)
                 - 4.0 * max(0.0, delta - 2.0)
                 + 6.0 * min(len(rec.get("releases") or []), 8))
        if hint and (hint in normalize(artist) or normalize(artist) in hint):
            score += 40.0
        if score > best_score and artist:
            best, best_score = (artist, str(rec.get("title") or "")), score
    return best


def musicbrainz_lookup(title: str, duration: float, artist_hint: str = "") -> Optional[Tuple[str, str]]:
    """Find the recording matching `title` and `duration` (within a small window)."""
    if not title or duration <= 0:
        return None
    window = max(6.0, duration * 0.02)
    lo, hi = int((duration - window) * 1000), int((duration + window) * 1000)
    q = f'recording:"{title.replace(chr(34), " ")}" AND dur:[{lo} TO {hi}]'
    try:
        r = requests.get(MUSICBRAINZ_RECORDING, params={"query": q, "fmt": "json", "limit": 15},
                         headers=HTTP_HEADERS, timeout=MB_TIMEOUT)
        if r.status_code != 200:
            log.info("MusicBrainz answered HTTP %s; skipping", r.status_code)
            return None
        return rank_musicbrainz(r.json().get("recordings", []) or [], duration, artist_hint)
    except (requests.RequestException, ValueError) as exc:
        log.info("MusicBrainz lookup failed: %s", exc)
        return None


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


# --------------------------------------------------------------------------- entry point
def extract_metadata(path: Path, settings: Settings) -> SongInfo:
    """Resolve title/artist/duration for one audio file."""
    title, artist, album, duration = read_tags(path)
    source = "id3" if (title and artist) else ""
    alt_titles: List[str] = []

    if not title or not artist:
        f_artist, f_title = parse_filename(path.stem)
        title = title or f_title
        artist = artist or f_artist
        source = source or "filename"
    if duration <= 0:
        duration = probe_duration(path)
    title = clean_title(title) or path.stem

    # Fingerprinting is authoritative, so when it is configured it goes before the
    # fuzzy title-only guesses below.
    if not artist and settings.acoustid_api_key:
        hit = identify_acoustid(path, settings.acoustid_api_key)
        if hit:
            artist, title = hit
            source = "acoustid"

    if not artist and title:
        log.info("No artist in tags or filename; looking '%s' up by title and length", title)
        hint = lrclib_artist_for_title(title)
        hit = musicbrainz_lookup(title, duration, artist_hint=hint or "")
        if hit:
            artist, mb_title = hit
            if mb_title and normalize(mb_title) != normalize(title):
                alt_titles.append(title)       # keep the filename title for lyric searches
                title = mb_title
            source = "musicbrainz"
        elif hint:
            artist, source = hint, "lrclib"

    # Titles from lookups can carry "(Remastered)"-style noise too.
    title = clean_title(title) or path.stem
    if source == "filename":
        title = smart_title_case(title)
    info = SongInfo(path=path, title=title, artist=artist.strip(), album=album,
                    duration=duration, source=source or "filename", alt_titles=alt_titles)
    log.info("Identified '%s' by '%s' (%s, %.0fs)", info.title, info.display_artist,
             info.source, info.duration)
    return info
