# wikiqa/wiki_client.py
from __future__ import annotations
from typing import Literal, Optional
import wikipediaapi
from wikiqa.datatypes import WikiPageData

LanguageCode = Literal["en", "ko", "es", "de", "fr", "ja", "zh"]

DEFAULT_UA = "dspy-wikipedia-qa/0.1 (contact: knightchaser@github)"


class WikiClient:
    """
    Thin, typed wrapper around wikipediaapi.
    Respects Wikimedia UA etiquette and supports both summary and full text.
    """

    def __init__(
        self,
        language: LanguageCode = "en",
        extract_format: wikipediaapi.ExtractFormat = wikipediaapi.ExtractFormat.WIKI,
    ) -> None:
        self._wiki = wikipediaapi.Wikipedia(
            user_agent=DEFAULT_UA, language=language, extract_format=extract_format
        )

    def get_page(self, title: str, *, full_text: bool = True) -> WikiPageData:
        """
        Fetch a page by exact title.
        If the page does not exist, returns WikiPageData with exists=False
        """
        page = self._wiki.page(title)
        exists = page.exists()
        if not exists:
            return WikiPageData(
                title=title,
                exists=False,  # invalid title!
                summary=None,
                full_url=None,
                canonical_url=None,
                text=None,
            )

        summary: Optional[str] = page.summary or None
        text: Optional[str] = page.text if full_text else None

        return WikiPageData(
            title=page.title,
            exists=True,
            summary=summary,
            full_url=getattr(page, "fullurl", None),
            canonical_url=getattr(page, "canonicalurl", None),
            text=text,
        )
