"""Unit tests for the data pipeline (no network, no GUI)."""

from pathlib import Path

import numpy as np

from lyricchord.config import Settings
from lyricchord.models import Lyrics, LyricLine
from lyricchord.pipeline.chords import theory
from lyricchord.pipeline.chords.sheet import (align_sheet_to_lyrics, looks_like_chord_sheet,
                                              parse_chord_sheet)
from lyricchord.pipeline.lyrics import (artist_matches, choose_lyrics_candidate, parse_lrc,
                                        plain_to_lines, score_lyrics_candidate, title_variants)
from lyricchord.pipeline.metadata import artist_consensus, parse_filename, rank_musicbrainz
from lyricchord.utils.text import clean_title, safe_filename, smart_title_case


# --------------------------------------------------------------------------- metadata
def test_parse_filename_variants():
    assert parse_filename("Radiohead - Creep") == ("Radiohead", "Creep")
    assert parse_filename("01 - Radiohead - Creep (Official Video)") == ("Radiohead", "Creep")
    assert parse_filename("03. Oasis - Wonderwall [Lyrics]") == ("Oasis", "Wonderwall")
    assert parse_filename("Wonderwall") == ("", "Wonderwall")
    assert parse_filename("Artist_-_Song_Name")[0] == "Artist"


def test_clean_title_and_safe_filename():
    assert clean_title("Hey Jude (Remastered 2015)") == "Hey Jude"
    assert safe_filename('A/B:C*D?"E') == "A_B_C_D__E"
    assert smart_title_case("eye in the sky") == "Eye in the Sky"
    assert smart_title_case("Already Cased") == "Already Cased"


def test_artist_consensus_from_lrclib_results():
    results = [
        {"artistName": "The Alan Parsons Project", "syncedLyrics": "x"},
        {"artistName": "Alan Parsons Project", "syncedLyrics": "x"},
        {"artistName": "The Alan Parsons Project", "plainLyrics": "x"},
        {"artistName": "Some Cover Band", "syncedLyrics": "x"},
        {"artistName": "Instrumental Guy", "instrumental": True},   # no lyrics: ignored
    ]
    assert artist_consensus(results) == "The Alan Parsons Project"
    assert artist_consensus([{"artistName": "A", "plainLyrics": "x"}]) is None   # too few votes
    assert artist_consensus([]) is None


def test_rank_musicbrainz_prefers_duration_and_popularity():
    recordings = [
        {"score": 100, "title": "Eye in the Sky", "length": 389000, "releases": [{}, {}],
         "artist-credit": [{"name": "Centory"}]},
        {"score": 84, "title": "Sirius / Eye in the Sky", "length": 391000, "releases": [{}] * 7,
         "artist-credit": [{"name": "The Alan Parsons Project"}]},
        {"score": 91, "title": "Sirius/Eye in the Sky", "length": 398000, "releases": [{}],
         "artist-credit": [{"name": "Zombi"}]},
    ]
    assert rank_musicbrainz(recordings, 391.3) == ("The Alan Parsons Project", "Sirius / Eye in the Sky")
    # An artist hint from another source outweighs a slightly better text score.
    assert rank_musicbrainz(recordings[:1] + recordings[2:], 391.3, artist_hint="Zombi")[0] == "Zombi"
    assert rank_musicbrainz([], 100.0) is None


def _lrc(first: float, last: float, n: int) -> str:
    step = (last - first) / max(1, n - 1)
    return "\n".join("[%02d:%05.2f]line %d" % ((first + i * step) // 60, (first + i * step) % 60, i) for i in range(n))


def _rec(rid: int, duration: float, first: float, last: float, n: int = 60) -> dict:
    return {"id": rid, "duration": duration, "syncedLyrics": _lrc(first, last, n), "plainLyrics": "x"}


def test_choose_candidate_rejects_timing_copied_from_another_edition():
    # Mirrors the real "Sirius / Eye in the Sky" case: the 6:31 album edit vs the 4:36 single.
    file_len = 391.3
    single = [_rec(i, 276 + i % 2, 19, 235) for i in range(1, 8)]           # the popular 4:36 edition
    copied = [_rec(10 + i, 386 + i, 19, 235) for i in range(5)]             # single's timing, album length
    right = [_rec(20, 390.3, 127, 343), _rec(21, 390, 127, 343), _rec(22, 387, 131, 346),
             _rec(23, 392, 132, 264, n=27), _rec(24, 391, 132, 264, n=27)]
    junk = [_rec(30, 2551, 132, 347), _rec(31, 446, 132, 347)]  # odd lengths, correct timing
    chosen, note = choose_lyrics_candidate(single + copied + right + junk, file_len)
    assert chosen["id"] == 22, note            # complete record nearest the cluster's median start
    assert "5 of 10" in note and "copied" not in note
    # Without the single's records the copied timing cannot be recognised, but the
    # right cluster still wins on completeness and a plausible ending.
    chosen, _ = choose_lyrics_candidate(copied + right, file_len)
    assert chosen["id"] == 22


def test_choose_candidate_lets_audio_arbitrate_first_line():
    right = [_rec(20, 390.3, 127, 343), _rec(21, 390, 127, 343), _rec(22, 387, 131, 346),
             _rec(23, 392, 132, 264, n=27), _rec(24, 391, 132, 264, n=27),
             _rec(25, 392, 132.4, 347, n=51)]   # same start as 23/24 (within 0.75 s), more complete
    asked = []

    def scorer(times):                       # pretend the voice clearly enters at 132 s
        asked.extend(times)
        return [1.0 - abs(t - 132.0) for t in times]

    chosen, note = choose_lyrics_candidate(right, 391.3, onset_scorer=scorer)
    assert chosen["id"] == 25 and "audio places the first line at 132.4s" in note
    assert asked == [127, 131, 132]          # one representative time per distinct start
    # Agreeing records never trigger the audio check.
    asked.clear()
    agree = [_rec(1, 200, 10, 180), _rec(2, 201, 10.3, 181)]
    choose_lyrics_candidate(agree, 200.0, onset_scorer=scorer)
    assert asked == []
    # A failing scorer falls back to the consensus pick.
    chosen, _ = choose_lyrics_candidate(right, 391.3, onset_scorer=lambda ts: 1 / 0)
    assert chosen["id"] == 22


def test_choose_candidate_no_penalty_when_only_the_outro_differs():
    # Single (276 s) and album edit (330 s) share the whole vocal timeline legitimately.
    single = [_rec(1, 276, 19, 235), _rec(2, 276, 19, 235)]
    album = [_rec(3, 330, 19, 235), _rec(4, 331, 20, 236)]
    chosen, note = choose_lyrics_candidate(single + album, 330.5)
    assert chosen["id"] in (3, 4) and "copied" not in note


def test_choose_candidate_majority_and_fallbacks():
    good = [_rec(1, 200, 10, 180), _rec(2, 201, 11, 181), _rec(3, 199, 10, 179)]
    odd = [_rec(4, 200, 40, 190)]
    chosen, _ = choose_lyrics_candidate(good + odd, 200.0)
    assert chosen["id"] in (1, 2, 3)
    # No record of matching length: fall back to the closest edition rather than nothing.
    chosen, note = choose_lyrics_candidate([_rec(5, 276, 19, 235)], 391.0)
    assert chosen["id"] == 5 and "closest" in note
    # Plain-only records are still usable when nothing synced exists.
    chosen, _ = choose_lyrics_candidate([{"id": 6, "duration": 391, "plainLyrics": "words"}], 391.0)
    assert chosen["id"] == 6
    assert choose_lyrics_candidate([{"id": 7, "duration": 391, "instrumental": True}], 391.0)[0] is None


def test_title_variants_cover_uploader_spellings():
    v = title_variants("Sirius / Eye in the Sky")
    assert v[0] == "Sirius / Eye in the Sky"
    assert "Sirius Eye in the Sky" in v and "Sirius/Eye in the Sky" in v
    assert title_variants("Wonderwall") == ["Wonderwall"]      # case-insensitive dedupe
    assert len(title_variants("A - B & C")) <= 4


def test_artist_matches_is_lenient_but_rejects_covers():
    assert artist_matches("Alan Parsons Project", "The Alan Parsons Project")
    assert artist_matches("The Alan Parsons Project", "Alan Parsons Project")
    assert artist_matches("Alan Parsons Project, The", "The Alan Parsons Project")
    assert artist_matches("anything", "")                     # unknown artist: no filtering
    assert not artist_matches("Spanish Magic Guitar", "The Alan Parsons Project")
    assert not artist_matches("", "The Alan Parsons Project")


def test_score_lyrics_candidate_ordering():
    synced_match = {"syncedLyrics": "x", "duration": 391}
    synced_other_edition = {"syncedLyrics": "x", "duration": 276}
    plain_match = {"plainLyrics": "x", "duration": 391}
    instrumental = {"instrumental": True, "duration": 391}
    scores = [score_lyrics_candidate(c, 391.3) for c in (synced_match, synced_other_edition, plain_match, instrumental)]
    assert scores == sorted(scores, reverse=True)
    assert scores[1] > scores[2]        # any synced beats any plain
    assert scores[3] < 0


# --------------------------------------------------------------------------- lyrics
def test_parse_lrc_basic_and_multi_timestamp():
    lrc = "[ar:Someone]\n[00:10.00]First line\n[00:12.50][00:30.00]Repeated\n[00:15.000] \n"
    lines = parse_lrc(lrc, duration=60)
    assert [l.text for l in lines] == ["First line", "Repeated", "", "Repeated"]
    assert lines[0].start == 10.0 and lines[0].end == 12.5
    assert lines[1].start == 12.5 and lines[1].end == 15.0
    assert lines[3].start == 30.0 and lines[3].end == 40.0  # capped hold


def test_parse_enhanced_lrc_words():
    lrc = "[00:05.00] <00:05.00>Hello <00:05.50>big <00:06.00>world"
    lines = parse_lrc(lrc, duration=10)
    assert lines[0].text == "Hello big world"
    assert [w.text for w in lines[0].words] == ["Hello", "big", "world"]
    assert lines[0].progress_at(5.0) == 0.0
    assert 0.3 < lines[0].progress_at(5.6) < 0.7
    assert lines[0].progress_at(9.0) > 0.9
    assert lines[0].progress_at(10.0) == 1.0


def test_parse_lrc_offset_tag():
    lines = parse_lrc("[offset:+1500]\n[00:10.00]Early\n[00:20.00]Later", duration=60)
    assert lines[0].start == 8.5 and lines[1].start == 18.5      # positive offset = earlier
    lines = parse_lrc("[offset:-2000]\n[00:10.00]Late", duration=60)
    assert lines[0].start == 12.0


def test_plain_lyrics_spread_evenly():
    lines = plain_to_lines("a\nb\n\nc", duration=100)
    assert len(lines) == 3
    assert lines[0].start < lines[1].start < lines[2].start < 100


# --------------------------------------------------------------------------- theory
def test_parse_label():
    assert theory.parse_label("C") == (0, "maj")
    assert theory.parse_label("F#m") == (6, "min")
    assert theory.parse_label("Bbmaj7") == (10, "maj7")
    assert theory.parse_label("Am7") == (9, "min7")
    assert theory.parse_label("G7") == (7, "7")
    assert theory.parse_label("C/E") == (0, "maj")
    assert theory.parse_label("Dsus4") == (2, "sus4")
    assert theory.parse_label("Bdim") == (11, "dim")
    assert theory.parse_label("Hello") is None
    assert theory.parse_label("Am I") is None


def test_spelling_and_keys():
    assert theory.spell(10, "min", True) == "Bbm"
    assert theory.spell(10, "min", False) == "A#m"
    assert theory.key_uses_flats(5, "major")          # F major
    assert not theory.key_uses_flats(7, "major")      # G major
    assert theory.key_name(1, "major") == "Db major"
    assert (7, "maj") in theory.diatonic_chords(0, "major")
    assert (4, "maj") in theory.diatonic_chords(9, "minor")   # E major (harmonic minor V)


def test_estimate_key_from_chroma():
    chroma = np.zeros(12)
    for pc in (0, 2, 4, 5, 7, 9, 11):  # C major scale
        chroma[pc] = 1.0
    chroma[0] += 1.0
    chroma[7] += 0.5
    tonic, mode, _ = theory.estimate_key(chroma)
    assert (tonic, mode) == (0, "major")


# --------------------------------------------------------------------------- sheets
UG_SHEET = """[Intro]
C  G  Am  F

[Verse 1]
C            G
Hello darkness my old friend
Am                  F
I've come to talk with you again

[Chorus]
F        G          C
Singing in the pouring rain
"""

CHORDPRO_SHEET = """{title: Test}
[C]Hello darkness [G]my old friend
[Am]I've come to talk [F]with you again
"""


def test_parse_ug_sheet():
    sheet = parse_chord_sheet(UG_SHEET)
    assert sheet[0].text == "" and [c for _, c in sheet[0].chords] == ["C", "G", "Am", "F"]
    assert sheet[1].text == "Hello darkness my old friend"
    assert [c for _, c in sheet[1].chords] == ["C", "G"]
    assert sheet[1].chords[0][0] == 0 and sheet[1].chords[1][0] == 13
    assert looks_like_chord_sheet(UG_SHEET)


def test_parse_chordpro_sheet():
    sheet = parse_chord_sheet(CHORDPRO_SHEET)
    assert sheet[0].text == "Hello darkness my old friend"
    assert sheet[0].chords == [(0, "C"), (15, "G")]
    assert looks_like_chord_sheet(CHORDPRO_SHEET)


def test_align_sheet_to_synced_lyrics():
    lyrics = Lyrics(lines=[
        LyricLine(10.0, 14.0, "Hello darkness, my old friend"),
        LyricLine(14.0, 18.0, "I've come to talk with you again"),
        LyricLine(20.0, 24.0, "Singing in the pouring rain"),
    ], synced=True, source="test")
    events = align_sheet_to_lyrics(parse_chord_sheet(UG_SHEET), lyrics, duration=30)
    labels = [e.label for e in events]
    assert labels[:4] == ["C", "G", "Am", "F"]          # intro spread over 0..10
    assert events[0].start == 0.0
    # verse chords land inside their lines
    verse = [e for e in events if 10 <= e.start < 18]
    assert [e.label for e in verse] == ["C", "G", "Am", "F"]
    assert events[-1].end == 30
    assert all(e.end >= e.start for e in events)


def test_settings_roundtrip(tmp_path: Path):
    s = Settings(fps=60, accent_color="#ff0000")
    f = tmp_path / "s.json"
    s.save(f)
    loaded = Settings.load(f)
    assert loaded.fps == 60 and loaded.accent_color == "#ff0000"
    assert loaded.size == (1920, 1080)
