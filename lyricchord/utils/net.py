"""Shared HTTP identity for the public services we call (lrclib, MusicBrainz).

MusicBrainz rate-limits and blocks per User-Agent, so every module must present the
same, descriptive one.
"""

USER_AGENT = "LyricChord/1.0 (https://github.com/MasstarVT/Lyric-Chord)"
HTTP_HEADERS = {"User-Agent": USER_AGENT}
