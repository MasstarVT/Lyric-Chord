"""
Small JSON cache so re-rendering a batch with different visual settings does not
re-download lyrics or re-run chord analysis.

Keyed by (absolute path, size, mtime) so a re-tagged or replaced file invalidates
automatically. Stored under <output_dir>/.lyricchord_cache/.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("lyricchord")


def file_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class SongCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.dir = root / ".lyricchord_cache"
        self.enabled = enabled

    def _file(self, path: Path) -> Path:
        return self.dir / f"{file_key(path)}.json"

    def _read(self, path: Path) -> dict:
        f = self._file(path)
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        return {}

    def get(self, path: Path, kind: str) -> Optional[dict]:
        if not self.enabled:
            return None
        return self._read(path).get(kind)

    def put(self, path: Path, kind: str, data: dict) -> None:
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            blob = self._read(path)
            blob[kind] = data
            self._file(path).write_text(json.dumps(blob), encoding="utf-8")
        except OSError as exc:
            log.debug("cache write failed: %s", exc)
