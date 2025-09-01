# wikiqa/cli/evaluate.py
from __future__ import annotations

import json
import os
import re
import yaml
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import typer
from rich import print
from rich.panel import Panel
from rich.table import Table

import dspy
from wikiqa.index_milvus import (
    get_client,
    DEFAULT_COLLECTION,
    DEFAULT_URI,
)
from wikiqa.retriever_milvus import MilvusRetriever
from wikiqa.rag_dspy import SimpleRAG

eval_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Evaluate the RAG model."
)

# NOTE:
# Helpers (normalization + numbers)

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_NUM_RE = re.compile(
    r"(?<![\w.])-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?"
)


def _normalize(s: str) -> str:
    """
    Normalize a string: lowercase, trim, unify quotes/dashes, collapse whitespace.
    """
    s = s.strip().lower()
    s = (
        s.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    s = _WS_RE.sub(" ", s)
    return s


def _extract_numbers(s: str) -> list[float]:
    """
    Extract all numbers from a string, removing thousands separators.
    Valid numbers include integers and decimals, with optional leading minus sign.
    """
    nums: list[float] = []
    for m in _NUM_RE.finditer(s):
        raw = m.group(0)
        # remove thousands separators (e.g. "1,000" -> "1000", "1 000" -> "1000")
        num = float(raw.replace(",", "").replace(" ", ""))

        # Valid if float conversion worked
        try:
            nums.append(int(num))
        except ValueError:
            continue
    return nums


def _approx_numeric_present(
    expected: str, answer: str, tolerance_pcrcentage: float
) -> bool:
    """
    If expected contains any number(s), accept if the answer
    contains any number within +/- tolerance_percentage % of any expected number.
    """
    expected_nums: list[float] = _extract_numbers(expected)
    if not expected_nums:
        return False

    answer_nums: list[float] = _extract_numbers(answer)
    if not answer_nums:
        return False

    tolerance = max(0.0, tolerance_pcrcentage / 100.0)

    for en in expected_nums:
        for an in answer_nums:
            if en == 0:
                # avoid division by zero
                continue
            if abs(en - an) / abs(en) <= tolerance:
                # The answer number is within the tolerance range
                return True

    return False


# NOTE:
# Metric: must-include with any-of groups + numeric tolerance


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    must: list[Any]  # list[str | list[str]]


def _coerce_eval_case(obj: dict[str, Any]) -> EvalCase:
    """
    Coerce a dict to an EvalCase, validating the structure.
    """
    question = str(obj.get("question", "")).strip()
    must = obj.get("must", [])
    if not isinstance(must, list):
        raise ValueError("Field 'must' must be a list of strings or list of strings.")

    # Normalize structure but don't lowercase yet
    # (To match case-insensitivity parts later)
    groups: list[Any] = []
    for group in must:
        if isinstance(group, str):
            groups.append(group)
        elif isinstance(groups, list) and all(isinstance(s, str) for s in group):
            groups.append(list(group))
        else:
            raise ValueError(
                "Field 'must' must be a list of strings or list of strings."
            )

    return EvalCase(question=question, must=groups)


def _contains_phrase(answer_norm: str, needle: str) -> bool:
    """
    Check if the normalized answer contains the normalized needle as a substring.
    """
    return _normalize(needle) in answer_norm


def must_include_metric_factory(tolerance_percentage: float) -> Callable:
    """
    Returns a metric function usable by dspy.Evaluate. The metric:
        - takes an Example with fields {question, must}
        - gets a Prediction with field .answer
        - returns fraction of constraints satisfied in [0, 1]
    Each constraint is either:
        - a string: must appear (case-insensitive) OR match numerically within tolerance
        - a list[str]: at least one of the options must appear (or numeric-approx match)
    """
    # Shared list to collect per-example details
    records_lock: threading.Lock = threading.Lock()
    records: list[dict[str, Any]] = []

    def metric(
        example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
    ) -> float:
        answer = str(prediction.answer).strip() or ""
        answer_norm = _normalize(answer)
        groups: list[Any] = example.must if hasattr(example, "must") else []
        total = len(groups) if groups else 0
        satisfied = 0  # of groups satisfied
        failed_details: list[dict[str, Any]] = []
        satisfied_details: list[dict[str, Any]] = []

        for group in groups:
            ok = False
            opts = [group] if isinstance(group, str) else list(group)

            # check any-of within the group
            for opt in opts:
                if _contains_phrase(answer_norm, opt) or _approx_numeric_present(
                    opt, answer, tolerance_percentage
                ):
                    ok = True
                    satisfied_details.append({"groups": group, "matched": opt})
                    break
            if ok:
                satisfied += 1
            else:
                failed_details.append({"groups": group})

        score = (satisfied / total) if total > 0 else 0.0

        # Store a compact record for optional JSON dump
        with records_lock:
            records.append(
                {
                    "question": example.question,
                    "answer": answer,
                    "score": score,
                    "satisfied": satisfied_details,
                    "failed": failed_details,
                    "must": groups,
                    "sources": getattr(prediction, "sources", []),
                }
            )
        return score

    # expose the captured records so the caller can persist them!
    metric.records = records  # type: ignore[attr-defined]
    metric.records_lock = records_lock  # type: ignore[attr-defined]
    return metric


# NOTE:
# data loading


def _load_jsonl(path: Path) -> list[EvalCase]:
    """
    Loads a JSONL file of EvalCase items.
    Each line is a JSON object with fields {question, must}.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    items: list[EvalCase] = []
    for line in raw_lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(_coerce_eval_case(obj))
    return items


def _load_yaml(path: Path) -> list[EvalCase]:
    """
    Loads a YAML file of EvalCase items.
    Each item is a YAML object with fields {question, must}.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_list = data if isinstance(data, list) else data.get("examples", [])
    if not isinstance(raw_list, list):
        raise ValueError(
            "YAML must be a list of {question, must} or {examples: [...]}."
        )
    return [_coerce_eval_case(obj) for obj in raw_list]


def _load_cases(path: Path, data_format: str) -> list[EvalCase]:
    """
    Load EvalCase items from a file, format auto-detected or specified.
    """
    fmt = data_format.lower()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return _load_yaml(path)
        return _load_jsonl(path)
    if fmt == "yaml":
        return _load_yaml(path)
    if fmt == "jsonl":
        return _load_jsonl(path)
    raise ValueError("--data-format must be one of: auto|yaml|jsonl")


# NOTE:
# Pretty tables
def _render_summary_table(records: list[dict[str, Any]], truncate: int = 80) -> Table:
    """
    Render a rich Table summarizing the evaluation records.
    """
    table = Table(title="Evaluation Results")
    table.add_column("#", justify="right", style="bold")
    table.add_column("score", justify="right")
    table.add_column("satisfied/total", justify="right")
    table.add_column("question")
    table.add_column("failed_groups")

    for idx, row in enumerate(records, start=1):
        score = f"{row['score']:.3f}"
        satisfied = len(row.get("satisfied", []))
        total = len(row.get("must", []))
        question = str(row["question"])

        if truncate and len(question) > truncate:
            question = question[:truncate] + "…"

        # show up to 2 failed groups compactly
        failed = row.get("failed", [])
        failed_txt = "; ".join([", ".join(g.get("group", [])) for g in failed[:2]])
        if truncate and len(failed_txt) > truncate:
            failed_txt = failed_txt[:truncate] + "…"

        table.add_row(str(idx), score, f"{satisfied}/{total}", question, failed_txt)

    return table


def _render_wide_table(records: list[dict[str, Any]], truncate: int = 100) -> Table:
    """
    Render a rich Table with detailed evaluation records.
    """
    table = Table(title="Evaluation Results (Wide)")
    table.add_column("#", justify="right", style="bold")
    table.add_column("score", justify="right")
    table.add_column("satisfied/total", justify="right")
    table.add_column("question")
    table.add_column("satisfied")
    table.add_column("failed")
    table.add_column("sources")

    for idx, row in enumerate(records, start=1):
        score = f"{row['score']:.3f}"
        satisfied = len(row.get("satisfied", []))
        total = len(row.get("must", []))
        question = str(row["question"])

        if truncate and len(question) > truncate:
            question = question[:truncate] + "…"
        sat = "; ".join([x.get("matched", "") for x in row.get("satisfied", [])])
        fail = "; ".join(
            [", ".join(group.get("group", [])) for group in row.get("failed", [])]
        )
        srcs = " ".join(row.get("sources", [])[:3])
        for val_name in ("sat", "fail", "srcs"):
            val = locals()[val_name]
            if truncate and len(val) > truncate:
                locals()[val_name] = val[:truncate] + "…"

        table.add_row(
            str(idx), score, f"{satisfied}/{total}", question, sat, fail, srcs
        )

    return table


# NOTE:
# CLI commands


@eval_app.command("run")
def eval_run(
    data_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="JSONL with {question, must}"
    ),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Milvus collection"),
    uri: str = typer.Option(DEFAULT_URI, help="Milvus URI (file path = Milvus Lite)"),
    k: int = typer.Option(6, "--k", help="Top-k passages"),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI chat model"),
    min_score: float = typer.Option(0.60, help="Retriever gating: min similarity"),
    min_hits: int = typer.Option(
        2, help="Retriever gating: require N hits >= min_score"
    ),
    tolerance_percentage: float = typer.Option(
        3.0, help="Numeric tolerance percent for phrase checks"
    ),
    threads: int | None = typer.Option(None, help="Parallel threads for evaluation"),
    show_dspy_table: bool = typer.Option(
        False,
        "--show-dspy-table/--no-dspy-table",
        help="Show DSPy's built-in table (else custom table)",
    ),
    view: str = typer.Option("table", help="table|wide|json"),
    truncate: int = typer.Option(100, help="Truncate long fields (0 = no truncate)"),
    max_examples: int | None = typer.Option(None, help="Evaluate at most N examples"),
    save_json: Path | None = typer.Option(None, help="Write per-example results JSON"),
    data_format: str = typer.Option("auto", help="auto|yaml|jsonl"),
) -> None:
    """
    Evaluate the RAG pipeline with a tolerant 'must-include phrases' metric.

    YAML format (recommended):
    ---
    - question: When Avril Lavigne was born, and where?
      must: ["September 27, 1984", "Belleville", "Ontario", "Canada"]
    - question: How big is Jupiter?
      must:
        - ["equatorial diameter", "mean radius"]
        - ["142,984", "69,911"]
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(Panel.fit("[bold red]Set OPENAI_API_KEY in your environment.[/bold red]"))
        raise typer.Exit(code=3)

    # Load data
    items = _load_cases(data_path, data_format=data_format)
    if max_examples is not None and max_examples > 0:
        items = items[:max_examples]  # truncate to max_examples

    # Build DSPy program
    db = get_client(uri=uri)
    retr = MilvusRetriever(
        client=db,
        collection=collection,
        min_score=min_score,
        min_hits=min_hits,
        output_fields=("text", "url", "page_title", "section_path", "lang"),
    )
    lm = dspy.LM(model=model, temperature=1.0, max_tokens=4096)

    dspy.settings.configure(lm=lm)

    rag = SimpleRAG(retriever=retr, top_k=k)

    # Build devset for DSPy
    devset = [
        dspy.Example(question=example.question, must=example.must).with_inputs(
            "question"
        )
        for example in items
    ]

    # Metric
    metric = must_include_metric_factory(tolerance_percentage=tolerance_percentage)

    # Evaluate via DSPy
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=threads,
        display_progress=True,
        display_table=(len(devset) if show_dspy_table else 0),
    )
    result = evaluator(rag)
    avg = getattr(result, "score", None)
    print(
        Panel.fit(
            f"[bold]Average Metric:[/bold] {avg:.3f}"
            if isinstance(avg, (int, float))
            else "Done."
        )
    )

    # Our structured output...
    records: list[dict[str, Any]] = getattr(metric, "records", [])
    if view == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        table = (
            _render_wide_table(records, truncate=truncate)
            if view == "wide"
            else _render_summary_table(records, truncate=truncate)
        )
        print(table)

    # Optional details dump
    if save_json:
        with getattr(metric, "records_lock"):
            save_json.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(Panel.fit(f"[bold]Saved details to[/bold] {save_json}"))
