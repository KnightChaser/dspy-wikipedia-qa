# wikiqa/cli.py
from __future__ import annotations
import typer
import wikipediaapi
from rich import print
from wikiqa.wiki_client import WikiClient

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
