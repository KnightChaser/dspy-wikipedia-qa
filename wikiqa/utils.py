# utils.py
from __future__ import annotations

import os
import typer


def ensure_openai_api_key() -> None:
    """
    Checks for OPENAI_API_KEY environment variable and exits if not found.
    This application requires OpenAI API access necessarily.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        # Use typer.echo to print and then exit, which is idiomatic for Typer
        print("Please set the OPENAI_API_KEY environment variable.")
        raise typer.Exit(code=3)
