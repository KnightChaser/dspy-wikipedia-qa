# main.py
import warnings
from wikiqa.cli import app

with warnings.catch_warnings(action="ignore"):
    app()
