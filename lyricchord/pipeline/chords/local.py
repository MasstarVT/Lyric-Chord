"""
Local chord detection with librosa.

Pipeline:
  audio -> harmonic/percussive separation -> CQT chroma -> beat-synchronous chroma
        -> cosine similarity against chord templates -> (optional) key prior
        -> Viterbi decoding with a "sticky" transition matrix -> merged segments

It is not as accurate as a trained model, but it runs offline in ~10-30 s per song
and produces clean, playable progressions for most pop/rock/folk material.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ...config import Settings
from ...models import ChordEvent, ChordTrack
from ...utils.audio import decode_audio
from . import theory

log = logging.getLogger("lyricchord")

SR = 22050
HOP = 512
SOFTMAX_TEMPERATURE = 0.04     # lower = trust the template match more
STAY_PROBABILITY = 0.85        # Viterbi self-transition probability per beat
NON_DIATONIC_PENALTY = 0.75    # multiplies similarity for chords outside the key
SILENCE_RATIO = 0.03           # segments below this fraction of peak RMS become "N"

ProgressFn = Optional[Callable[[float], None]]


def _report(cb: ProgressFn, value: float) -> None:
    if cb:
        cb(value)


def _segments(boundaries: np.ndarray) -> List[slice]:
    return [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]


def fold_tempo(bpm: float) -> float:
    """Fold octave errors of the beat tracker into the common 60-190 BPM range."""
    if bpm <= 0:
        return 0.0
    while bpm < 60:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return bpm


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def detect_chords(path: Path, settings: Settings, on_progress: ProgressFn = None) -> ChordTrack:
    """Analyse an audio file and return a ChordTrack (source='librosa')."""
    import librosa  # imported lazily: slow to import, only needed here

    y = decode_audio(path, SR)
    if y.size < SR:  # < 1 s of audio
        return ChordTrack(source="librosa")
    _report(on_progress, 0.1)

    # 1) Tonal content only: percussive hits smear chroma.
    y_harm = librosa.effects.harmonic(y=y, margin=3.0)
    _report(on_progress, 0.4)

    # 2) Chroma from a constant-Q transform (better low-frequency resolution than STFT).
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=SR, hop_length=HOP)
    n_frames = chroma.shape[1]
    _report(on_progress, 0.6)

    # 3) Beat tracking on the full mix; chords are assumed to change on beats.
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="frames")
        tempo = fold_tempo(float(np.atleast_1d(tempo)[0]))
    except Exception as exc:  # pragma: no cover - librosa edge cases
        log.debug("beat tracking failed (%s); using fixed windows", exc)
        tempo, beats = 0.0, np.array([], dtype=int)
    if len(beats) < 8:
        beats = np.arange(0, n_frames, max(1, int(0.5 * SR / HOP)))
    boundaries = np.unique(np.concatenate([[0], beats, [n_frames]]).astype(int))
    segs = _segments(boundaries)

    seg_chroma = librosa.util.sync(chroma, segs, aggregate=np.median, pad=False)
    seg_chroma = seg_chroma / (np.linalg.norm(seg_chroma, axis=0, keepdims=True) + 1e-9)

    # 4) Template matching.
    qualities = theory.TRIADS + (theory.SEVENTHS if settings.include_seventh_chords else [])
    chords, templates = theory.build_templates(qualities)
    sim = templates @ seg_chroma                      # (n_chords, n_segments) in [0, 1]

    tonic, mode, key_conf = theory.estimate_key(chroma.mean(axis=1))
    if settings.snap_chords_to_key:
        diatonic = theory.diatonic_chords(tonic, mode, include_sevenths=True)
        prior = np.array([1.0 if c in diatonic else NON_DIATONIC_PENALTY for c in chords])
        sim = sim * prior[:, None]

    # 5) Viterbi smoothing: chords tend to persist for several beats.
    prob = _softmax(sim / SOFTMAX_TEMPERATURE, axis=0)
    transition = librosa.sequence.transition_loop(len(chords), STAY_PROBABILITY)
    states = librosa.sequence.viterbi(prob, transition)
    _report(on_progress, 0.9)

    # 6) Silence detection -> "N" (no chord).
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms = rms[:n_frames] if rms.size >= n_frames else np.pad(rms, (0, n_frames - rms.size))
    seg_rms = np.array([rms[s].mean() if s.stop > s.start else 0.0 for s in segs])
    silent = seg_rms < max(1e-4, SILENCE_RATIO * seg_rms.max())

    use_flats = settings.prefer_flats and theory.key_uses_flats(tonic, mode)
    times = librosa.frames_to_time(boundaries, sr=SR, hop_length=HOP)
    events: List[ChordEvent] = []
    for i, state in enumerate(states):
        label = "N" if silent[i] else theory.spell(chords[state][0], chords[state][1], use_flats)
        start, end = float(times[i]), float(times[i + 1])
        if events and events[-1].label == label:
            events[-1].end = end
        else:
            events.append(ChordEvent(start, end, label))

    events = merge_short_events(events, settings.min_chord_seconds)
    duration = y.size / SR
    if events:
        events[-1].end = max(events[-1].end, duration)

    track = ChordTrack(events=events, key=theory.key_name(tonic, mode, settings.prefer_flats),
                       bpm=round(tempo, 1), source="librosa")
    log.info("Chords: %d segments, key %s (conf %.2f), %.0f BPM", len(events), track.key,
             key_conf, tempo)
    _report(on_progress, 1.0)
    return track


def merge_short_events(events: List[ChordEvent], min_seconds: float) -> List[ChordEvent]:
    """Absorb segments shorter than `min_seconds` into their neighbours."""
    if min_seconds <= 0 or len(events) < 2:
        return events
    out: List[ChordEvent] = []
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.duration < min_seconds and (out or i + 1 < len(events)):
            if out:
                out[-1].end = ev.end          # extend previous
            else:
                events[i + 1].start = ev.start  # give to next
            i += 1
            continue
        if out and out[-1].label == ev.label:
            out[-1].end = ev.end
        else:
            out.append(ChordEvent(ev.start, ev.end, ev.label))
        i += 1
    return out


def estimate_tempo(path: Path) -> float:
    """Cheap BPM estimate, used when chords come from a sheet instead of analysis."""
    try:
        import librosa

        y = decode_audio(path, SR)
        tempo, _ = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
        return round(fold_tempo(float(np.atleast_1d(tempo)[0])), 1)
    except Exception as exc:
        log.debug("tempo estimation failed: %s", exc)
        return 0.0
