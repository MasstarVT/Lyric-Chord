"""
Music-theory helpers: pitch classes, chord templates, key estimation, spelling.

Chord "qualities" are kept deliberately small (triads plus optional sevenths) so
the on-screen chords stay playable for a strumming guitarist / pianist.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOTE_TO_PC: Dict[str, int] = {n: i for i, n in enumerate(NOTES_SHARP)}
NOTE_TO_PC.update({n: i for i, n in enumerate(NOTES_FLAT)})
NOTE_TO_PC.update({"Cb": 11, "Fb": 4, "E#": 5, "B#": 0})

# quality -> semitone intervals from the root
QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "min7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}
QUALITY_SUFFIX: Dict[str, str] = {
    "maj": "", "min": "m", "7": "7", "min7": "m7", "maj7": "maj7",
    "dim": "dim", "aug": "aug", "sus2": "sus2", "sus4": "sus4",
}

TRIADS = ["maj", "min"]
SEVENTHS = ["7", "min7", "maj7"]

# Krumhansl-Schmuckler key profiles.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Keys conventionally written with flats (major tonics; minors via relative major).
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1, 6}  # F Bb Eb Ab Db Gb

Chord = Tuple[int, str]  # (root pitch class, quality)


# --------------------------------------------------------------------------- templates
def build_templates(qualities: Sequence[str]) -> Tuple[List[Chord], np.ndarray]:
    """Return chord list and a (n_chords, 12) matrix of unit-norm binary templates."""
    chords: List[Chord] = []
    rows = []
    for q in qualities:
        for root in range(12):
            vec = np.zeros(12)
            for iv in QUALITY_INTERVALS[q]:
                vec[(root + iv) % 12] = 1.0
            vec[root] = 1.15  # slight emphasis on the root
            rows.append(vec / np.linalg.norm(vec))
            chords.append((root, q))
    return chords, np.vstack(rows)


# --------------------------------------------------------------------------- keys
def estimate_key(chroma_mean: np.ndarray) -> Tuple[int, str, float]:
    """Krumhansl-Schmuckler key finding. Returns (tonic_pc, 'major'|'minor', confidence)."""
    c = np.asarray(chroma_mean, dtype=float)
    if c.sum() <= 0:
        return 0, "major", 0.0
    best = (0, "major", -2.0)
    for tonic in range(12):
        for mode, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
            score = float(np.corrcoef(c, np.roll(profile, tonic))[0, 1])
            if score > best[2]:
                best = (tonic, mode, score)
    return best


def diatonic_chords(tonic: int, mode: str, include_sevenths: bool = False) -> Set[Chord]:
    """Chords built on the scale degrees of the key (harmonic-minor V included)."""
    if mode == "major":
        degrees = [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (11, "dim")]
        sevenths = [(0, "maj7"), (2, "min7"), (4, "min7"), (5, "maj7"), (7, "7"), (9, "min7")]
    else:
        degrees = [(0, "min"), (2, "dim"), (3, "maj"), (5, "min"), (7, "min"), (7, "maj"),
                   (8, "maj"), (10, "maj")]
        sevenths = [(0, "min7"), (3, "maj7"), (5, "min7"), (7, "7"), (8, "maj7"), (10, "7")]
    out = {((tonic + d) % 12, q) for d, q in degrees}
    if include_sevenths:
        out |= {((tonic + d) % 12, q) for d, q in sevenths}
    return out


def key_uses_flats(tonic: int, mode: str) -> bool:
    rel_major = tonic if mode == "major" else (tonic + 3) % 12
    return rel_major in _FLAT_MAJOR_TONICS


def key_name(tonic: int, mode: str, prefer_flats: bool = True) -> str:
    names = NOTES_FLAT if (prefer_flats and key_uses_flats(tonic, mode)) else NOTES_SHARP
    return f"{names[tonic]} {mode}"


def infer_key_from_chords(chords: Iterable[Tuple[Chord, float]]) -> Tuple[int, str]:
    """Pick the key whose diatonic set covers the most (duration-weighted) chords."""
    weighted = list(chords)
    best, best_score = (0, "major"), -1.0
    for tonic in range(12):
        for mode in ("major", "minor"):
            dia = diatonic_chords(tonic, mode, include_sevenths=True)
            score = sum(w for (c, w) in weighted if (c[0], _base_quality(c[1])) in dia or c in dia)
            # Tonic bonus: songs usually start/end on the tonic.
            if weighted and weighted[0][0][0] == tonic:
                score += 0.5
            if score > best_score:
                best, best_score = (tonic, mode), score
    return best


def _base_quality(q: str) -> str:
    return {"min7": "min", "maj7": "maj", "7": "maj"}.get(q, q)


# --------------------------------------------------------------------------- spelling / parsing
def spell(root: int, quality: str, use_flats: bool) -> str:
    names = NOTES_FLAT if use_flats else NOTES_SHARP
    return names[root % 12] + QUALITY_SUFFIX.get(quality, quality)


_CHORD_RE = re.compile(
    r"^(?P<root>[A-G])(?P<acc>[#b♯♭]?)(?P<qual>[^/\s]*?)(?:/(?P<bass>[A-G][#b]?))?$"
)
# Every quality suffix we accept, mapped to its canonical quality. The suffix must match
# in full (plus an optional parenthesised alteration such as "(b5)"), so ordinary words
# that start with a note letter - "Amen", "Come", "Go", "Do" - are not read as chords.
_QUALITY_CANON: Dict[str, str] = {
    "": "maj", "maj": "maj", "M": "maj", "6": "maj", "69": "maj", "6/9": "maj", "5": "maj",
    "2": "maj", "4": "maj", "add9": "maj", "add2": "maj", "add4": "maj", "add11": "maj",
    "maj7": "maj7", "maj9": "maj7", "maj13": "maj7", "M7": "maj7", "Δ": "maj7", "Δ7": "maj7",
    "m": "min", "min": "min", "-": "min", "m6": "min",
    "m7": "min7", "min7": "min7", "-7": "min7", "m9": "min7", "m11": "min7", "m13": "min7",
    "7": "7", "9": "7", "11": "7", "13": "7", "dom7": "7", "7sus4": "7", "7sus2": "7",
    "7#9": "7", "7b9": "7", "7#5": "7", "7b5": "7",
    "dim": "dim", "dim7": "dim", "°": "dim", "°7": "dim", "o7": "dim", "ø": "dim", "ø7": "dim", "m7b5": "dim",
    "aug": "aug", "+": "aug",
    "sus2": "sus2", "sus4": "sus4", "sus": "sus4",
}
_ALTERATION_RE = re.compile(r"\([^)]*\)$")


def parse_label(label: str) -> Optional[Chord]:
    """Parse 'F#m7', 'Bb', 'Gsus4', 'C/E', 'Am7(b5)' -> (root_pc, quality). None if not a chord."""
    label = label.strip()
    if not label or label.upper() in ("N", "NC", "N.C.", "N.C"):
        return None
    m = _CHORD_RE.match(label)
    if not m:
        return None
    root = m.group("root") + m.group("acc").replace("♯", "#").replace("♭", "b")
    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return None
    qual = _ALTERATION_RE.sub("", m.group("qual") or "")
    canon = _QUALITY_CANON.get(qual)
    return (pc, canon) if canon else None


def is_chord_token(token: str) -> bool:
    return parse_label(token) is not None


def canonical_label(label: str, use_flats: Optional[bool] = None) -> str:
    """Re-spell a chord label in our canonical form. Unknown labels are returned as-is."""
    parsed = parse_label(label)
    if parsed is None:
        return label
    root, qual = parsed
    if use_flats is None:
        use_flats = "b" in label[1:2]
    return spell(root, qual, use_flats)
