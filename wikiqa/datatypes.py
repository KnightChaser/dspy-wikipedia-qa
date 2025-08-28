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
