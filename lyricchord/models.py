"""
Data models shared across the pipeline.

Everything here is a plain dataclass so it can be serialised to JSON for the
on-disk cache (see pipeline.cache) and passed safely between worker threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class SongInfo:
    """Identity of a song, resolved from tags / filename / fingerprint."""

    path: Path
    title: str
    artist: str
    album: str = ""
    duration: float = 0.0          # seconds
    source: str = "unknown"        # "id3" | "filename" | "musicbrainz" | "lrclib" | "acoustid"
    alt_titles: List[str] = field(default_factory=list)  # other names to try when searching lyrics

    @property
    def search_titles(self) -> List[str]:
        seen, out = set(), []
        for t in [self.title, *self.alt_titles]:
            key = t.strip().lower()
            if t.strip() and key not in seen:
                seen.add(key)
                out.append(t.strip())
        return out

    @property
    def display_title(self) -> str:
        return self.title or self.path.stem

    @property
    def display_artist(self) -> str:
        return self.artist or "Unknown Artist"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @staticmethod
    def from_dict(d: dict) -> "SongInfo":
        d = dict(d)
        d["path"] = Path(d["path"])
        return SongInfo(**d)


@dataclass
class LyricWord:
    """A single word with its start time (only present for enhanced LRC)."""

    time: float
    text: str


@dataclass
class LyricLine:
    """One displayed line of lyrics with absolute start/end times in seconds."""

    start: float
    end: float
    text: str
    words: Optional[List[LyricWord]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def progress_at(self, t: float) -> float:
        """Fraction (0..1) of the line that has been sung at time t.

        Uses word timings when available (enhanced LRC), otherwise linear
        interpolation across the line duration.
        """
        if t <= self.start:
            return 0.0
        if t >= self.end:
            return 1.0
        if self.words and len(self.words) > 1:
            # Count characters sung so far, interpolating inside the current word.
            total_chars = max(1, len(self.text))
            sung = 0
            for i, w in enumerate(self.words):
                nxt = self.words[i + 1].time if i + 1 < len(self.words) else self.end
                if t >= nxt:
                    sung += len(w.text) + 1
                else:
                    frac = (t - w.time) / max(1e-6, nxt - w.time) if t > w.time else 0.0
                    sung += frac * (len(w.text) + 1)
                    break
            return min(1.0, sung / total_chars)
        return (t - self.start) / max(1e-6, self.duration)


@dataclass
class Lyrics:
    """A full set of lyric lines for a song."""

    lines: List[LyricLine] = field(default_factory=list)
    synced: bool = False        # True if timestamps came from an LRC source
    source: str = "none"        # "lrclib" | "syncedlyrics" | "sidecar" | "none"
    ref_duration: float = 0.0   # length of the recording the lyrics were timed for (0 = unknown)

    @property
    def available(self) -> bool:
        return bool(self.lines)

    def duration_mismatch(self, file_duration: float, tolerance: float = 8.0) -> float:
        """Seconds of difference between the lyrics' reference recording and this file (0 if fine)."""
        if not self.ref_duration or not file_duration or not self.synced:
            return 0.0
        diff = abs(self.ref_duration - file_duration)
        return diff if diff > tolerance else 0.0

    def to_dict(self) -> dict:
        return {
            "synced": self.synced,
            "source": self.source,
            "ref_duration": self.ref_duration,
            "lines": [
                {
                    "start": l.start,
                    "end": l.end,
                    "text": l.text,
                    "words": [[w.time, w.text] for w in l.words] if l.words else None,
                }
                for l in self.lines
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Lyrics":
        lines = []
        for l in d.get("lines", []):
            words = [LyricWord(t, w) for t, w in l["words"]] if l.get("words") else None
            lines.append(LyricLine(l["start"], l["end"], l["text"], words))
        return Lyrics(lines=lines, synced=d.get("synced", False), source=d.get("source", "none"),
                      ref_duration=float(d.get("ref_duration", 0.0) or 0.0))


@dataclass
class ChordEvent:
    """A chord held from `start` to `end` (seconds). Label like 'Am', 'F#', 'N'."""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ChordTrack:
    """Timeline of chords plus global musical info."""

    events: List[ChordEvent] = field(default_factory=list)
    key: str = ""                  # e.g. "G major"
    bpm: float = 0.0
    source: str = "none"           # "librosa" | "sheet" | "ultimate-guitar" | "none"

    @property
    def available(self) -> bool:
        return bool(self.events)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "bpm": self.bpm,
            "source": self.source,
            "events": [[e.start, e.end, e.label] for e in self.events],
        }

    @staticmethod
    def from_dict(d: dict) -> "ChordTrack":
        return ChordTrack(
            events=[ChordEvent(s, e, l) for s, e, l in d.get("events", [])],
            key=d.get("key", ""),
            bpm=d.get("bpm", 0.0),
            source=d.get("source", "none"),
        )


@dataclass
class SongData:
    """Everything the renderer needs for one song."""

    info: SongInfo
    lyrics: Lyrics
    chords: ChordTrack
