# wikiqa/cli.py
from __future__ import annotations
import json
from dataclasses import asdict
from typing import Optional
import typer
import wikipediaapi
from rich import print
from rich.panel import Panel
from rich.table import Table
from wikiqa.wiki_client import WikiClient
from wikiqa.search import search_pages

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
