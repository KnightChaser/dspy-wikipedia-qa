# wikiqa/chunk.py
from __future__ import annotations
from typing import Iterable, Tuple
import re
import hashlib
import wikipediaapi

from wikiqa import config
from wikiqa.datatypes import Chunk


def _estimate_tokens(s: str) -> int:
    """
    Estimate the number of tokens in a string with a cheap heuristic:
    T ~ 0.75 * words
    """
    words = len(re.findall(r"\S+", s))
    return max(1, int(words * 0.75))


def _anchor_from_title(section_title: str) -> str:
    """
    Create a Wikipedia-style anchor from a section title.
    """
    return section_title.replace(" ", "_")


def _stable_64bit_int(s: str) -> int:
    """
    Create a stable 64-bit integer hash from a string.
    """
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False) & ((1 << 63) - 1)


def _walk_sections(page: wikipediaapi.WikipediaPage) -> Iterable[Tuple[str, str]]:
    """
    Yield (section_page, text) for the page summary and each section subtree.
    Use WikipediaAPI's built-in section representations.
    """
    # yield the summary first
    if getattr(page, "summary", None):
        yield ("Summary", page.summary)

    def rec(
        sections: list[wikipediaapi.WikipediaPageSection], path: list[str]
    ) -> Iterable[Tuple[str, str]]:
        """
        Recursively walk sections, yielding (section_path, text).
        section_path is the full path to the section, with " > " as a separator.
        """
        for section in sections:
            section_title = section.title.strip()
            this_path = path + [section_title] if section_title else path
            text = (section.text or "").strip()
            if text:
                yield (" > ".join(this_path), text)

            # recurse into subsections
            if section.sections:
                yield from rec(section.sections, this_path)

    # then, recurse into sections
    yield from rec(getattr(page, "sections", []) or [], [])


def _split_with_overlap(
    text: str,
    max_tokens: int = config.DEFAULT_CHUNK_MAX_TOKENS,
    overlap_tokens: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split by paragraphs then fall back to sentence-ish units, keeping overlap.
    Receives a block of text and splits it into chunks of at most `max_tokens` tokens,
    with `overlap_tokens` tokens of overlap between chunks.
    Returns a list of text chunks. (Actually a list of strings.)
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        """
        Flush the current chunk to the list of chunks.
        Keep overlap by retaining the last `overlap_tokens` worth of text.
        """
        nonlocal current, current_tokens
        if not current:
            return

        joined = "\n\n".join(current).strip()
        if joined:
            chunks.append(joined)

        # clear current, keeping overlap
        current = []
        current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)
        if paragraph_tokens > max_tokens:
            # The given sentence is too long, we need to split it further.
            # explode long paragraph into sentence-like pieces again
            # because we need to split it
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
            for sentence in sentences:
                sentence_tokens = _estimate_tokens(sentence)
                if current_tokens + sentence_tokens > max_tokens:
                    flush()
                    # overlap by appending tail of previous chunks
                    if overlap_tokens > 0 and chunks:
                        tail_words = " ".join(
                            chunks[-1].split()[-(overlap_tokens * 4 // 3) :]
                        )  # split by words up to estimated overlap
                        current = [tail_words] if tail_words else []
                        current_tokens = _estimate_tokens(tail_words)
                current.append(sentence)
                current_tokens += sentence_tokens
            # We complete the last chunk after processing all sentences
            flush()
        else:
            # normal paragraph within limits
            if current_tokens + paragraph_tokens > max_tokens and current:
                flush()
                if overlap_tokens > 0 and chunks:
                    tail_words = " ".join(
                        chunks[-1].split()[-(overlap_tokens * 4 // 3) :]
                    )  # split by words up to estimated overlap
                    current = [tail_words] if tail_words else []
                    current_tokens = _estimate_tokens(tail_words)
            current.append(paragraph)
            current_tokens += paragraph_tokens

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def make_chunks_from_page(
    page: wikipediaapi.WikipediaPage,
    *,
    lang: str,
    base_url: str | None = None,
    max_tokens: int = config.DEFAULT_CHUNK_MAX_TOKENS,
    overlap_tokens: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Produce section-aware chunks for a Wikipedia page.
    - Keeps headings as part of the chunk string for better retrieval.
    - Splits oversized sections with small overlap.
    """
    assert max_tokens > 50
    base_url = base_url or getattr(page, "fullurl", None) or ""

    chunks: list[Chunk] = []

    for section_path, text in _walk_sections(page):
        # Include a small header line for anchoring
        header = f"[{page.title} / {section_path}]"
        body = text.strip()
        if not body:
            continue

        pieces = _split_with_overlap(
            body,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        if not pieces:
            # If splitting failed, just use the whole body
            # (Expect the body was too small to split.)
            pieces = [body]

        # For each piece (a chunk of text), create a Chunk object
        # with the header prepended
        for idx, piece in enumerate(pieces):
            merged = f"{header}\n\n{piece}".strip()
            token = _estimate_tokens(merged)
            anchor = _anchor_from_title(section_path.split(" > ")[-1])
            url = f"{base_url}#{anchor}" if base_url and anchor else base_url
            if not url:
                url = ""

            # Create a stable 64-bit integer ID from lang, page title, section path, and index
            stable = _stable_64bit_int(f"{lang}|{page.title}|{section_path}|{idx}")

            chunks.append(
                Chunk(
                    id_int=stable,
                    page_title=page.title,
                    section_path=section_path,
                    url=url,
                    lang=lang,
                    text=merged,
                    token_estimate=str(token),
                )
            )

    return chunks
