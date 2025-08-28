# wikiqa/types.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WikiPageData:
    title: str
    exists: bool
    summary: str | None
    full_url: str | None
    canonical_url: str | None
    text: str | None
