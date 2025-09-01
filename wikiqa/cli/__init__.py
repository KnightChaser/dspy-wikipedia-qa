# wikiqa/cli/__init__.py
from __future__ import annotations
from wikiqa.cli.generic import app
from wikiqa.cli.milvus import db_app
from wikiqa.cli.evaluate import eval_app

app.add_typer(db_app, name="db", help="Milvus vector database management")
app.add_typer(eval_app, name="eval", help="Evaluate RAG performance")

# Expose the main app only
__all__ = ["app"]
