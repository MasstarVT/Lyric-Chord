"""
Narrow-window vocal-entry check.

A global "where do the vocals start" detector is unreliable on a full mix, but a much
smaller question is tractable: given a few candidate times a few seconds apart (lyric
records that disagree about the first line), at which one does singing actually begin?
The true entry shows a sustained rise in harmonic energy in the vocal band; guitar
fills and drum hits at the wrong candidates usually do not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from ..utils.audio import decode_audio

log = logging.getLogger("lyricchord")

SR = 22050
HOP = 256
N_FFT = 2048
VOCAL_BAND = (300.0, 3500.0)   # Hz; fundamentals and lower formants of sung voice


def vocal_onset_rise(path: Path, times: List[float], before: float = 1.5, after: float = 1.5) -> List[float]:
    """For each candidate time, return (mean vocal-band energy just after) - (just before).

    Larger is more consistent with a voice entering at that moment. Energies are
    normalised to the window's peak so the values are comparable between candidates.
    """
    if not times:
        return []
    import librosa

    y = decode_audio(path, SR)
    lo = max(0.0, min(times) - before - 2.0)
    hi = max(times) + after + 2.0
    seg = y[int(lo * SR):int(hi * SR)]
    if seg.size < SR:
        return [0.0] * len(times)

    harmonic = librosa.effects.harmonic(y=seg, margin=3.0)
    spec = np.abs(librosa.stft(harmonic, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    energy = spec[(freqs >= VOCAL_BAND[0]) & (freqs <= VOCAL_BAND[1])].sum(axis=0)
    energy = energy / (energy.max() + 1e-9)
    fps = SR / HOP

    def mean(a: float, b: float) -> float:
        i, j = int(max(0.0, (a - lo) * fps)), int(max(0.0, (b - lo) * fps))
        return float(energy[i:j].mean()) if j > i else 0.0

    rises = [mean(t, t + after) - mean(t - before, t) for t in times]
    log.debug("vocal onset check: %s", ", ".join(f"{t:.1f}s -> {r:+.2f}" for t, r in zip(times, rises)))
    return rises
