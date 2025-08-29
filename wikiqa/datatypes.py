# wikiqa/types.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WikiPageData:
    """
    Data about a single Wikipedia page.
    If exists is False, the page does not exist,
    thus other fields will be None.
    """

    title: str
    exists: bool
    summary: str | None
    full_url: str | None
    canonical_url: str | None
    text: str | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """
    A single Wikipedia search result (REST /v1/search/page)
    """

    title: str
    key: str
    url: str
    description: str | None
    excerpt: str | None
    thumbnail_url: str | None


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    A semantically coherent snippet with provenance.
    """

    id_int: int
    page_title: str
    section_path: str  # e.g., "History > Early life"
    url: str  # page URL + optional #anchor
    lang: str
    text: str
    token_estimate: str  # rough token count for bookkeeping
