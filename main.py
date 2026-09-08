"""
LyricChord entry point.

    python main.py                 -> desktop GUI
    python main.py --cli <inputs>  -> headless batch (see lyricchord/cli.py)
"""

from __future__ import annotations

import sys


def _utf8_console() -> None:
    """Keep non-ASCII artist names from crashing log output on cp1252 Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:  # pragma: no cover
                pass


def main() -> int:
    _utf8_console()
    argv = sys.argv[1:]
    if argv and argv[0] in ("--cli", "cli"):
        from lyricchord.cli import main as cli_main

        return cli_main(argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    from lyricchord.gui.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
