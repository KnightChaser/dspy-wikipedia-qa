# wikiqa/cli/milvus.py
from __future__ import annotations

import json
from typing import Any, Optional
import typer
from rich import print
from rich.panel import Panel
from rich.table import Table

from wikiqa.index_milvus import (
    get_client,
    DEFAULT_COLLECTION,
    DEFAULT_URI,
    DEFAULT_EMBED_DIM,
)

db_app = typer.Typer(add_completion=False, no_args_is_help=True)


@db_app.command("collections")
def db_collections(
    uri: str = typer.Option(DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of table"),
) -> None:
    """
    List all collections in the Milvus database
    """
    client = get_client(uri=uri)
    cols = client.list_collections() or []
    if json_out:
        print(
            json.dumps({"uri": uri, "collections": cols}, indent=2, ensure_ascii=False)
        )
        return

    table = Table(title=f"Milvus Collections at {uri}")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Collection Name")
    for index, column in enumerate(cols, start=1):
        table.add_row(str(index), column)
    if not cols:
        table.caption = "[bold yellow]No collections found.[/bold yellow]"
    print(table)


@db_app.command("peek")
def db_peek(
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Collection name"),
    uri: str = typer.Option(DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"),
    limit: int = typer.Option(5, help="Number of rows to display"),
    fields: str = typer.Option(
        "id,page_title,section_path,lang,url,token_estimate,text",
        help="Comma-separated field list to display (embedding is excluded)",
    ),
    where: Optional[str] = typer.Option(
        None,
        help='Optional scalar filter (e.g., \'lang == "en" and page_title like "Alan%"\')',
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of table"),
    truncate: int = typer.Option(
        160, help="Truncate long text fields to N chars (0 = no truncate)"
    ),
) -> None:
    """
    Show sample rows from a collection. Tries query(); falls back to a neutral search if needed.
    """
    client = get_client(uri=uri)
    field_list = [field.strip() for field in fields.split(",") if field.strip()]

    # never pull embeddings by mistake
    field_list = [f for f in field_list if f != "embedding"]

    rows: list[dict[str, Any]] = []

    # Try the high-level query API first
    try:
        rows = (
            client.query(
                collection_name=collection,
                filter=where or "",
                output_fielsd=field_list,
                limit=limit,
            )
            or []
        )
    except Exception:
        print(
            Panel(
                "[bold yellow]Failed to query(); falling back to search()[/bold yellow]"
            )
        )
        raise typer.Exit(code=1)

    if json_out:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    if not rows:
        print(
            Panel.fit(f"[bold red]No rows found in collection:[/bold red] {collection}")
        )
        raise typer.Exit(code=1)

    table = Table(title=f"Peek: {collection} at {uri} (limit={limit})")
    for field in field_list:
        table.add_column(field, overflow="fold")

    for row in rows:
        values: list[str] = []
        for field in field_list:
            v = row.get(field, "")
            s = str(v)
            if (
                truncate
                and field.lower() in {"text", "excerpt", "summary"}
                and len(s) > truncate
            ):
                # If truncating text fields, add ellipsis
                s = s[:truncate] + "…"
            values.append(s)
        table.add_row(*values)
    print(table)


@db_app.command("stats")
def db_stats(
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Collection name"),
    uri: str = typer.Option(DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of table"),
) -> None:
    """
    Best-effort stats & schema for a collection. Gracefully degrades if a method isn't supported.
    """
    client = get_client(uri=uri)

    info: dict[str, Any] = {"uri": uri, "collection": collection}

    # Try describe_collection (schema-ish)
    try:
        desc = client.describe_collection(collection_name=collection)
        info["describe_collection"] = desc
    except Exception:
        info["describe_collection"] = None

    # Try get_collection_stats (row count, etc.)
    row_count: Optional[int] = None
    try:
        stats = client.get_collection_stats(
            collection_name=collection
        )  # may not exist on older clients
        # try common shapes
        if isinstance(stats, dict):
            # look for common keys
            for key in ("row_count", "rowCount", "rows", "count"):
                if key in stats:
                    row_count = int(stats[key])  # type: ignore[arg-type]
                    break
        info["collection_stats"] = stats
    except Exception:
        info["collection_stats"] = None

    # As a fallback, sample one row to infer field names
    sample_fields = [
        "id",
        "page_title",
        "section_path",
        "lang",
        "url",
        "token_estimate",
        "text",
    ]
    sample: list[dict[str, Any]] = []
    try:
        sample = (
            client.query(
                collection_name=collection,
                filter="",
                output_fields=sample_fields,
                limit=1,
            )
            or []
        )
    except Exception:
        pass
    info["sample_fields"] = sample[0] if sample else None
    if row_count is not None:
        info["row_count_guess"] = row_count

    if json_out:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    # Pretty print
    table = Table(title=f"Stats: {collection} @ {uri}")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row(
        "Row count (guess)", str(row_count) if row_count is not None else "n/a"
    )
    # show a few fields from describe_collection if present
    if info["describe_collection"]:
        dc = info["describe_collection"]
        dim = None
        try:
            # common place the dimension lives (depends on client version)
            dim = dc.get("schema", {}).get("properties", {}).get("dimension") or dc.get(
                "dimension"
            )
        except Exception:
            dim = None
        table.add_row("Dimension", str(dim or f"(likely {DEFAULT_EMBED_DIM})"))
    else:
        table.add_row("Dimension", f"(likely {DEFAULT_EMBED_DIM})")

    # sample fields
    if info["sample_fields"]:
        sf = info["sample_fields"]
        field_names = ", ".join(sf.keys())
        table.add_row("Sample fields", field_names)

    print(table)


@db_app.command("titles")
def db_titles(
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Collection name"),
    uri: str = typer.Option(DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"),
    where: Optional[str] = typer.Option(
        None,
        help='Optional scalar filter, e.g., \'lang == "en" and page_title like "Alan%"\'.',
    ),
    limit: int = typer.Option(
        10000,
        help="Max rows to scan for counting (guard rail for large collections).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """
    List distinct page titles with their chunk counts.
    Best-effort: tries query(); falls back to a neutral vector search if needed.
    """
    client = get_client(uri=uri)

    rows: list[dict[str, Any]] = []
    field_list = ["page_title"]

    # Try the high-level query API
    try:
        rows = (
            client.query(
                collection_name=collection,
                filter=where or "",
                output_fields=field_list,
                limit=limit,
            )
            or []
        )
    except Exception:
        print(
            Panel(
                "[bold yellow]Failed to query(); falling back to search()[/bold yellow]"
            )
        )
        raise typer.Exit(code=1)

    # Aggregate counts by title
    counts: dict[str, int] = {}
    for row in rows:
        title = str(row.get("page_title", "")).strip()
        if not title:
            continue
        counts[title] = counts.get(title, 0) + 1

    # Sort by coutn desc, then title
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    if json_out:
        print(
            json.dumps(
                {
                    "uri": uri,
                    "collection": collection,
                    "where": where,
                    "scanned": len(rows),
                    "titles": [{"page_title": t, "chunks": c} for t, c in items],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    table = Table(
        title=f"Page titles in {collection.strip()} @ {uri} (scanned={len(rows)})"
    )
    table.add_column("#", justify="right", style="bold")
    table.add_column("Page title")
    table.add_column("Chunks", justify="right")

    for idx, (title, count) in enumerate(items, start=1):
        table.add_row(str(idx), title, str(count))

    if not items:
        table.caption = "No titles found (empty collection or filter too strict)."
    print(table)
