"""
Best-effort online chord-sheet retrieval from Ultimate Guitar.

Ultimate Guitar has no public API. Their pages embed a JSON blob in a
`<div class="js-store" data-content="...">` element which we parse. This is
scraping: it may break when the site changes, be rate-limited, or be blocked by
their bot protection. It is therefore OFF by default and the pipeline falls back to
local librosa detection whenever it fails. Please respect the site's terms of use.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
from typing import Optional

import requests

from ...utils.text import normalize

log = logging.getLogger("lyricchord")

SEARCH_URL = "https://www.ultimate-guitar.com/search.php"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
_STORE_RE = re.compile(r'class="js-store"\s+data-content="([^"]+)"')


def _store_json(page: str) -> Optional[dict]:
    m = _STORE_RE.search(page)
    if not m:
        return None
    try:
        return json.loads(html.unescape(m.group(1)))
    except ValueError:
        return None


def fetch_ultimate_guitar_sheet(artist: str, title: str, timeout: int = 20) -> Optional[str]:
    """Return UG chord-sheet text (with [ch]..[/ch] markup) for the best-rated match."""
    query = f"{artist} {title}".strip()
    if not query:
        return None
    try:
        r = requests.get(SEARCH_URL, params={"search_type": "title", "value": query},
                         headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            log.info("Ultimate Guitar search returned HTTP %s", r.status_code)
            return None
        data = _store_json(r.text)
        results = (((data or {}).get("store") or {}).get("page") or {}).get("data", {}).get("results", [])
        n_artist, n_title = normalize(artist), normalize(title)

        def score(item: dict) -> float:
            if item.get("type") != "Chords" or not item.get("tab_url"):
                return -1.0
            if item.get("tab_access_type", "public") != "public":
                return -1.0
            s = float(item.get("rating", 0)) * math.log1p(float(item.get("votes", 0)))
            if n_artist and normalize(item.get("artist_name", "")) == n_artist:
                s += 10.0
            if n_title and normalize(item.get("song_name", "")) == n_title:
                s += 10.0
            return s

        ranked = sorted((x for x in results if score(x) > 0), key=score, reverse=True)
        if not ranked:
            log.info("Ultimate Guitar: no chord sheets found for '%s'", query)
            return None
        best = ranked[0]
        log.info("Ultimate Guitar: using '%s - %s' (rating %.1f, %s votes)", best.get("artist_name"),
                 best.get("song_name"), float(best.get("rating", 0)), best.get("votes", 0))

        r = requests.get(best["tab_url"], headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        data = _store_json(r.text) or {}
        content = (data.get("store", {}).get("page", {}).get("data", {})
                   .get("tab_view", {}).get("wiki_tab", {}).get("content"))
        return content or None
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        log.warning("Ultimate Guitar lookup failed: %s", exc)
        return None
