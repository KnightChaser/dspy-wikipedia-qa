# wikiqa/cli/__init__.py
from __future__ import annotations
from wikiqa.cli.generic import app
from wikiqa.cli.milvus import db_app

app.add_typer(db_app, name="db", help="Milvus vector database management")

# Expose the main app only
__all__ = ["app"]
