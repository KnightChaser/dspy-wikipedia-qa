# wikiqa/search.py
from __future__ import annotations
from typing import Optional
import requests

from wikiqa.datatypes import SearchHit
from wikiqa.wiki_client import LanguageCode, DEFAULT_UA


def _rest_search_base(lang: str) -> str:
    """
    Build the REST search endpoint hosted on the target Wikipedia project.
    Using site-local REST avoids API keys and works anonymously.
    Docs (sample): https://en.wikipedia.org/w/rest.php/v1/search/page?q=jupiter&limit=20
    """
    return f"https://{lang}.wikipedia.org/w/rest.php/v1/search/page"


def search_pages(
    query: str, *, lang: LanguageCode = "en", limit: int = 5, timeout: float = 10.0
) -> list[SearchHit]:
    """
    Call Wikimedia REST 'search/page' to get candidate pages for a free-form query.

    - Returns ranked hits with title/key/URL and optional description/excerpt/thumbnail.
    - 'limit' is clamped to [1, 50] to be reasonable.
    - Uses default UA string respecting Wikimedia etiquette.

    References:
      - REST search/page sample + Python snippet. See docs.
    """
    # clamp limit
    limit = max(1, min(50, limit))

    url = _rest_search_base(lang)
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "limit": str(limit),
    }

    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("pages", [])
    hits: list[SearchHit] = []

    for p in pages:
        # Fields per REST docs: title, key, excerpt, description, thumbnail{url,...}
        title = p.get("title") or p.get("key") or ""  # e.g. "Jupiter"
        key = p.get("key") or title.replace(" ", "_")  # e.g. "Jupiter"
        desc = p.get("description")  # e.g. "Fifth planet from the Sun"
        excerpt = p.get(
            "excerpt"
        )  # e.g. '<span class="searchmatch">Jupiter</span> ...'
        thumb_url: Optional[str] = None
        thumb = p.get("thumbnail") or {}
        if isinstance(thumb, dict):
            tu = thumb.get(
                "url"
            )  # e.g. "//upload.wikimedia.org/wikipedia/commons/e/e2/...jpg"
            if isinstance(tu, str):
                thumb_url = (
                    tu if tu.startswith("http") else f"https://{lang}.wikipedia.org{tu}"
                )

        full_url = f"https://{lang}.wikipedia.org/wiki/{key}"

        hits.append(
            SearchHit(
                title=title,
                key=key,
                url=full_url,
                description=desc,
                excerpt=excerpt,
                thumbnail_url=thumb_url,
            )
        )

    return hits
