# wikiqa/cli/generic.py
from __future__ import annotations

import json
import typer
import wikipediaapi
from dataclasses import asdict
from rich import print
from rich.panel import Panel
from rich.table import Table

from wikiqa import config
from wikiqa.datatypes import WikiPageData
from wikiqa.wiki_client import WikiClient, DEFAULT_UA
from wikiqa.search import search_pages
from wikiqa.chunk import make_chunks_from_page
from wikiqa.index_milvus import (
    get_client,
    get_openai_ef,
    ensure_collection,
    upsert_chunks,
)
from wikiqa.rag_dspy import SimpleRAG, build_rag_pipeline
from wikiqa.utils import ensure_openai_api_key

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def search(
    query: str = typer.Argument(..., help="Free-form query to search Wikipedia"),
    lang: str = typer.Option("en", help="Language code, e.g., en, ko, es"),
    k: int = typer.Option(5, "--k", help="Number of results to return (max 50)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of table"),
) -> None:
    """
    Search Wikipedia and show top-k candidates (titles, URLs, brief context).
    """
    hits = search_pages(query=query, lang=lang, limit=k)  # type: ignore[arg-type]

    # Request JSON output
    if json_out:
        print(json.dumps([asdict(hit) for hit in hits], indent=2, ensure_ascii=False))
        return

    # No result available
    if not hits:
        print(Panel.fit(f"[bold red]No results for:[/bold red] {query!r}"))
        return

    table = Table(title=f"Search results for: {query!r} [{lang}]")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Title")
    table.add_column("URL")
    table.add_column("Description")
    table.add_column("Excerpt")

    for i, h in enumerate(hits, start=1):
        table.add_row(
            str(i),
            h.title,
            h.url,
            h.description or "",
            (h.excerpt or "").replace("\n", " ")[:160],
        )

    print(table)
    print(
        "[dim]Tip: use[/dim] [bold]fetch[/bold] [dim]to load a specific hit by index.[/dim]"
    )


@app.command()
def fetch(
    query: str = typer.Argument(..., help="Search query to resolve and fetch"),
    lang: str = typer.Option("en", help="Language code, e.g., en, ko, es"),
    k: int = typer.Option(5, "--k", help="Number of candidates to search (max 50)"),
    pick: int = typer.Option(1, "--pick", help="1-based index from search results"),
    html: bool = typer.Option(False, help="Fetch HTML instead of wiki text"),
    json_out: bool = typer.Option(True, "--json/--no-json", help="Emit JSON output"),
) -> None:
    """
    End-to-end: search -> pick one hit -> hydrate via wikipedia-api (summary/urls/text).
    """
    hits = search_pages(query=query, lang=lang, limit=k)  # type: ignore[arg-type]

    # Pick the i-th hit (1-based)
    if not hits:
        raise typer.Exit(code=1)

    # Validate pick
    if pick < 1 or pick > len(hits):
        print(
            Panel.fit(
                f"[bold red]Invalid --pick {pick}[/bold red]. Range: 1..{len(hits)}"
            )
        )
        raise typer.Exit(code=1)

    chosen = hits[pick - 1]

    extract = (
        wikipediaapi.ExtractFormat.HTML if html else wikipediaapi.ExtractFormat.WIKI
    )
    client = WikiClient(
        language=lang,  # type: ignore[arg-type]
        extract_format=extract,
    )
    page = client.get_page(title=chosen.title, full_text=True)

    if json_out:
        print(json.dumps(asdict(page), indent=2, ensure_ascii=False))
    else:
        print(Panel.fit(f"[bold]Selected:[/bold] {chosen.title}\n{chosen.url}"))
        print(page)


@app.command()
def get(
    title: str = typer.Argument(..., help="Exact Wikipedia page title"),
    lang: str = typer.Option("en", help="Language code, e.g., en, ko, es"),
    html: bool = typer.Option(False, help="Fetch HTML instead of wiki text"),
):
    extract = (
        wikipediaapi.ExtractFormat.HTML if html else wikipediaapi.ExtractFormat.WIKI
    )

    client = WikiClient(
        language=lang,  # type: ignore[arg-type]
        extract_format=extract,
    )
    data = client.get_page(title=title, full_text=True)
    print(data)


@app.command()
def index_title(
    title: str = typer.Argument(..., help="Exact Wikipedia page title to index"),
    lang: str = typer.Option("en", help="Language code, e.g., en, ko, es"),
    collection: str = typer.Option(
        config.DEFAULT_COLLECTION_NAME, help="Milvus collection name"
    ),
    uri: str = typer.Option(
        config.DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"
    ),
    html: bool = typer.Option(
        False, help="Use HTML for text extraction (default WIKI)"
    ),
    max_tokens: int = typer.Option(
        config.DEFAULT_CHUNK_MAX_TOKENS, help="Chunk target size (approx tokens)"
    ),
    overlap: int = typer.Option(
        config.DEFAULT_CHUNK_OVERLAP, help="Overlap between chunks (approx tokens)"
    ),
    embed_model: str = typer.Option(
        config.DEFAULT_EMBED_MODEL_NAME, help="OpenAI embedding model"
    ),
) -> None:
    """
    Fetch -> chunk -> embed -> index a single Wikipedia page into Milvus.
    """
    extract = (
        wikipediaapi.ExtractFormat.HTML if html else wikipediaapi.ExtractFormat.WIKI
    )
    client = WikiClient(
        language=lang,  # type: ignore[arg-type]
        extract_format=extract,
    )
    page: WikiPageData = client.get_page(title=title, full_text=True)
    if not page.exists or not page.text:
        print(Panel.fit(f"[bold red]Page not found or empty:[/bold red] {title!r}"))
        raise typer.Exit(code=1)

    # Build the actual wikipediaapi page again to access sections/summary
    wiki = wikipediaapi.Wikipedia(
        language=lang,  # type: ignore[arg-type]
        extract_format=extract,
        user_agent=DEFAULT_UA,
    )
    raw_page = wiki.page(page.title)

    chunks = make_chunks_from_page(
        page=raw_page,
        lang=lang,
        base_url=page.full_url or page.canonical_url or "",
        max_tokens=max_tokens,
        overlap_tokens=overlap,
    )
    if not chunks:
        print(
            Panel.fit(f"[bold yellow]No chunkable text for[/bold yellow] {page.title}")
        )
        raise typer.Exit(code=2)

    ef = get_openai_ef(model_name=embed_model)
    db = get_client(uri=uri)
    dim = getattr(ef, "dim", config.DEFAULT_EMBED_DIM)
    ensure_collection(db, collection_name=collection, dim=int(dim))
    insert = upsert_chunks(
        db,
        collection_name=collection,
        chunks=chunks,
        ef=ef,
    )

    print(
        Panel.fit(
            f"[bold green]Indexed {insert} chunks from:[/bold green] [bold]{page.title}[/bold] "
            f"into collection [bold]{collection}[/bold] at {uri}"
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(...),
    collection: str = typer.Option(
        config.DEFAULT_COLLECTION_NAME, help="Milvus collection name"
    ),
    uri: str = typer.Option(
        config.DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"
    ),
    k: int = typer.Option(
        config.DEFAULT_TOP_K, "--k", help="Number of retrieved chunks"
    ),
    model: str = typer.Option(config.DEFAULT_CHAT_MODEL_NAME, help="OpenAI chat model"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
    min_score: float = typer.Option(
        config.DEFAULT_MIN_SCORE, help="Minimum similarity score for hits"
    ),
    min_hits: int = typer.Option(
        config.DEFAULT_MIN_HITS, help="Minimum number of hits to consider valid"
    ),
) -> None:
    ensure_openai_api_key()

    rag: SimpleRAG = build_rag_pipeline(
        uri=uri,
        collection_name=collection,
        k=k,
        model_name=model,
        min_score=min_score,
        min_hits=min_hits,
    )
    pred = rag(question)

    print(Panel.fit(f"[bold]Answer[/bold]\n{pred.answer}"))
    if show_sources and getattr(pred, "sources", None):
        print(Panel.fit("[bold]Sources[/bold]\n" + "\n".join(pred.sources)))
