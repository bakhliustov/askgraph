"""Labeled retrieval eval: recall@k, MRR, hit-rate over ground-truth cases.

A case pairs a question with the files/symbols a good answer should retrieve.
Targets are written as ``path/to/file.py`` (file-level) or
``path/to/file.py::SymbolName`` (symbol-level). The metric functions are pure
and dependency-free so they're cheap to unit-test; :func:`evaluate` is the only
part that needs a live retriever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """One labeled question and the targets a good retrieval should surface."""

    question: str
    relevant: list[str] = field(default_factory=list)


def _normalize(path: str) -> str:
    return path.strip().lstrip("./").replace("\\", "/")


def hit_matches(hit: dict[str, Any], target: str) -> bool:
    """Does a retrieved hit satisfy a relevance target?

    ``file.py`` matches on file path; ``file.py::Sym`` additionally requires the
    hit's symbol to match. Path comparison is suffix-based so labels can be
    written relative to a subpackage.
    """
    meta = hit.get("metadata", {}) or {}
    hit_file = _normalize(str(meta.get("file_path", "")))
    target = target.strip()
    if "::" in target:
        tfile, _, tsym = target.partition("::")
        tfile = _normalize(tfile)
        file_ok = bool(tfile) and (hit_file == tfile or hit_file.endswith("/" + tfile))
        return file_ok and (meta.get("symbol") == tsym.strip())
    tfile = _normalize(target)
    return bool(tfile) and (hit_file == tfile or hit_file.endswith("/" + tfile))


def score_case(hits: list[dict[str, Any]], case: EvalCase, k: int) -> dict[str, float]:
    """Score one case: recall@k, reciprocal rank, and hit (0/1) over top-k hits."""
    topk = hits[:k]
    targets = case.relevant
    if not targets:
        return {"recall": 0.0, "rr": 0.0, "hit": 0.0}

    found = sum(1 for t in targets if any(hit_matches(h, t) for h in topk))
    recall = found / len(targets)

    rr = 0.0
    for rank, h in enumerate(topk, 1):
        if any(hit_matches(h, t) for t in targets):
            rr = 1.0 / rank
            break

    return {"recall": recall, "rr": rr, "hit": 1.0 if rr > 0 else 0.0}


def aggregate(per_case: list[dict[str, float]]) -> dict[str, float]:
    """Mean recall@k, MRR, and hit-rate across cases."""
    n = max(1, len(per_case))
    return {
        "recall_at_k": sum(c["recall"] for c in per_case) / n,
        "mrr": sum(c["rr"] for c in per_case) / n,
        "hit_rate": sum(c["hit"] for c in per_case) / n,
        "num_cases": float(len(per_case)),
    }


def load_cases(path: Path) -> list[EvalCase]:
    """Load eval cases from a JSON or YAML file: a list of {question, relevant}."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # pyyaml is a core dependency

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    cases: list[EvalCase] = []
    for item in raw or []:
        cases.append(EvalCase(question=item["question"], relevant=list(item.get("relevant", []))))
    return cases


def evaluate(
    retriever: Any,
    cases: list[EvalCase],
    top_k: int = 8,
    expand: bool = True,
    lexical: bool = False,
) -> dict[str, Any]:
    """Run each case through the retriever and return aggregate + per-case metrics.

    ``expand`` toggles structural graph expansion and ``lexical`` toggles
    identifier/symbol-name fusion, so callers can isolate each signal's effect
    on the same labeled set (e.g. pure vector vs + lexical vs + lexical + graph).
    """
    per_case: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []
    for case in cases:
        hits = retriever.retrieve_hybrid(case.question, top_k=top_k, expand=expand, lexical=lexical)
        scores = score_case(hits, case, top_k)
        per_case.append(scores)
        details.append({"question": case.question, **scores})
    return {
        "metrics": aggregate(per_case),
        "cases": details,
        "expand": expand,
        "lexical": lexical,
        "top_k": top_k,
    }
