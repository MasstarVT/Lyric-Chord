"""
LyricChord - batch generator of play-along lyric + chord videos from audio files.

Package layout
--------------
lyricchord.config      Persistent user settings (resolution, colours, fonts, ...).
lyricchord.models      Plain dataclasses passed between pipeline stages.
lyricchord.pipeline    Per-song data pipeline: metadata -> lyrics -> chords.
lyricchord.render      Frame drawing (Pillow) and video encoding (FFmpeg).
lyricchord.gui         CustomTkinter desktop application.
lyricchord.utils       FFmpeg discovery, audio decoding, fonts, logging helpers.
"""

__version__ = "1.0.0"
