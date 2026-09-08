"""Integration tests: synthetic audio -> chord detection, and a short video render."""

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from lyricchord.config import Settings
from lyricchord.models import ChordEvent, ChordTrack, Lyrics, LyricLine, SongData, SongInfo
from lyricchord.pipeline.chords.local import detect_chords, merge_short_events
from lyricchord.render.frames import FrameComposer, compute_layout, render_preview
from lyricchord.render.renderer import render_video
from lyricchord.utils.ffmpeg import find_ffmpeg

SR = 22050
NOTE_HZ = {"C": 261.63, "E": 329.63, "G": 392.00, "A": 220.00, "F": 174.61, "B": 246.94, "D": 293.66}


def _triad(freqs, seconds):
    t = np.arange(int(SR * seconds)) / SR
    sig = np.zeros_like(t)
    for f in freqs:
        for harmonic, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
            sig += amp * np.sin(2 * np.pi * f * harmonic * t)
    # simple strum envelope every 0.5 s so the beat tracker has something to lock to
    env = 0.4 + 0.6 * np.abs(np.cos(np.pi * t / 0.5)) ** 4
    return sig * env / (3 * len(freqs))


def _write_wav(path: Path, audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


@pytest.fixture(scope="module")
def progression_wav(tmp_path_factory) -> Path:
    # C - G - Am - F, 3 seconds each (a classic I-V-vi-IV in C major)
    chords = [("C", ["C", "E", "G"]), ("G", ["G", "B", "D"]),
              ("Am", ["A", "C", "E"]), ("F", ["F", "A", "C"])]
    audio = np.concatenate([_triad([NOTE_HZ[n] for n in notes], 3.0) for _, notes in chords])
    p = tmp_path_factory.mktemp("audio") / "progression.wav"
    _write_wav(p, audio)
    return p


def test_ffmpeg_is_available():
    assert Path(find_ffmpeg()).exists()


def test_detect_chords_on_synthetic_progression(progression_wav: Path):
    settings = Settings(min_chord_seconds=0.5, snap_chords_to_key=True)
    track = detect_chords(progression_wav, settings)
    assert track.source == "librosa"
    assert track.key.startswith("C major") or track.key.startswith("A minor")
    labels = [e.label for e in track.events if e.label != "N"]
    # Collapse repeats and check the expected progression appears in order.
    collapsed = [l for i, l in enumerate(labels) if i == 0 or l != labels[i - 1]]
    assert collapsed == ["C", "G", "Am", "F"], collapsed
    for ev, expected_start in zip(track.events, (0, 3, 6, 9)):
        assert abs(ev.start - expected_start) < 0.75, (ev, expected_start)


def test_merge_short_events():
    ev = [ChordEvent(0, 2, "C"), ChordEvent(2, 2.2, "G"), ChordEvent(2.2, 4, "C"), ChordEvent(4, 6, "F")]
    merged = merge_short_events(ev, 0.5)
    assert [e.label for e in merged] == ["C", "F"]
    assert merged[0].start == 0 and merged[0].end == 4


def test_layout_and_preview_frame():
    L = compute_layout(1920, 1080)
    assert L.lane_box[2] <= L.chord_box[2] and L.now_box[0] >= L.chord_box[0]
    img = render_preview(Settings(), size=(640, 360))
    assert img.size == (640, 360)
    # Portrait layout must also compose without error.
    img = render_preview(Settings(resolution="Vertical 1080x1920", background_style="solid"), size=(270, 480))
    assert img.size == (270, 480)


def test_render_short_video(progression_wav: Path, tmp_path: Path):
    info = SongInfo(progression_wav, "Test Song", "Test Artist", duration=3.0)
    lyrics = Lyrics([LyricLine(0.5, 1.5, "Hello world"), LyricLine(1.5, 3.0, "Second line here")], True, "test")
    chords = ChordTrack([ChordEvent(0, 1.5, "C"), ChordEvent(1.5, 3.0, "G")], "C major", 120.0, "test")
    song = SongData(info, lyrics, chords)
    settings = Settings(resolution="720p (1280x720)", fps=24, output_dir=str(tmp_path), preset="ultrafast")
    out = tmp_path / "out.mp4"
    seen = []
    render_video(song, settings, out, progress=seen.append)
    assert out.exists() and out.stat().st_size > 10_000
    assert seen and seen[-1] == 1.0
    probe = subprocess.run([find_ffmpeg(), "-i", str(out)], capture_output=True, text=True).stderr
    assert "Video: h264" in probe and "Audio: aac" in probe


def test_render_with_loop_background(progression_wav: Path, tmp_path: Path):
    bg = tmp_path / "bg.mp4"
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=24", "-t", "1", "-pix_fmt", "yuv420p", str(bg)], check=True)
    info = SongInfo(progression_wav, "Loop Song", "Artist", duration=2.0)
    song = SongData(info, Lyrics([LyricLine(0.2, 1.8, "Over a video")], True, "t"),
                    ChordTrack([ChordEvent(0, 2, "Am")], "A minor", 0, "t"))
    settings = Settings(resolution="720p (1280x720)", fps=24, background_style="loop",
                        loop_video_path=str(bg), preset="ultrafast")
    out = tmp_path / "loop.mp4"
    render_video(song, settings, out)
    probe = subprocess.run([find_ffmpeg(), "-i", str(out)], capture_output=True, text=True).stderr
    assert out.stat().st_size > 5_000 and "1280x720" in probe and "Audio: aac" in probe


def test_probe_duration_and_windowed_decode(progression_wav: Path):
    from lyricchord.utils.audio import decode_audio, probe_duration
    assert abs(probe_duration(progression_wav) - 12.0) < 0.1
    window = decode_audio(progression_wav, SR, start=3.0, duration=2.0)
    assert abs(window.size - 2 * SR) < SR * 0.05


def test_vocal_onset_rise_never_returns_nan(progression_wav: Path):
    from lyricchord.pipeline.vocal import vocal_onset_rise
    rises = vocal_onset_rise(progression_wav, [3.0, 40.0, 90.0])   # last two lie past the 12 s of audio
    assert len(rises) == 3 and all(np.isfinite(r) for r in rises)
    assert rises[1] == 0.0 and rises[2] == 0.0


def test_render_drawing_error_kills_ffmpeg(progression_wav: Path, tmp_path: Path, monkeypatch):
    import lyricchord.render.renderer as renderer

    class Broken(renderer.FrameComposer):
        def render(self, t):
            if t > 0.5:
                raise OSError("simulated glyph failure")
            return super().render(t)

    monkeypatch.setattr(renderer, "FrameComposer", Broken)
    song = SongData(SongInfo(progression_wav, "T", "A", duration=3.0), Lyrics(), ChordTrack())
    settings = Settings(resolution="720p (1280x720)", fps=24, preset="ultrafast")
    out = tmp_path / "broken.mp4"
    with pytest.raises(OSError):
        renderer.render_video(song, settings, out)      # must not hang waiting on FFmpeg
    assert not out.exists() and not out.with_name("broken.part.mp4").exists()


def test_empty_lyrics_are_not_cached(progression_wav: Path, tmp_path: Path, monkeypatch):
    import lyricchord.pipeline.processor as processor
    calls = []

    def fake_fetch(info):
        calls.append(info.title)
        return Lyrics()

    monkeypatch.setattr(processor, "fetch_lyrics", fake_fetch)
    monkeypatch.setattr(processor, "render_video", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(processor, "get_chords", lambda *a, **k: ChordTrack(events=[ChordEvent(0, 12, "C")], source="t"))
    import lyricchord.render.renderer as renderer
    monkeypatch.setattr(renderer, "render_video", lambda *a, **k: None)
    settings = Settings(output_dir=str(tmp_path), use_cache=True)
    processor.process_song(progression_wav, settings)
    processor.process_song(progression_wav, settings)
    assert calls == ["progression", "progression"]    # second run fetched again, not served from cache


def test_frame_bytes_size():
    song = SongData(SongInfo(Path("x.mp3"), "T", "A", duration=10.0), Lyrics(), ChordTrack())
    c = FrameComposer(song, Settings(), 320, 180)
    assert len(c.render(1.0)) == 320 * 180 * 3
    c = FrameComposer(song, Settings(), 320, 180, transparent=True)
    assert len(c.render(1.0)) == 320 * 180 * 4
