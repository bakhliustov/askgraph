"""Tests for the labeled retrieval-eval metrics (pure, no model/retriever)."""

from __future__ import annotations

import json
from pathlib import Path

from askgraph.eval import EvalCase, aggregate, hit_matches, load_cases, score_case


def _hit(file_path: str, symbol: str | None = None) -> dict:
    meta: dict = {"file_path": file_path}
    if symbol:
        meta["symbol"] = symbol
    return {"metadata": meta, "text": ""}


def test_hit_matches_file_and_symbol():
    h = _hit("pkg/a.py", "Foo")
    assert hit_matches(h, "pkg/a.py")  # exact file
    assert hit_matches(h, "a.py")  # suffix match (label relative to subpackage)
    assert hit_matches(h, "pkg/a.py::Foo")  # file + symbol
    assert not hit_matches(h, "pkg/a.py::Bar")  # wrong symbol
    assert not hit_matches(h, "pkg/b.py")  # wrong file


def test_score_case_recall_and_rr():
    hits = [_hit("b.py"), _hit("a.py", "Foo"), _hit("c.py")]
    case = EvalCase(question="q", relevant=["a.py::Foo", "c.py"])
    s = score_case(hits, case, k=3)
    assert s["recall"] == 1.0  # both targets found
    assert s["rr"] == 0.5  # first relevant ("a.py::Foo") at rank 2
    assert s["hit"] == 1.0


def test_score_case_respects_k():
    hits = [_hit("x.py"), _hit("y.py"), _hit("a.py", "Foo")]
    case = EvalCase(question="q", relevant=["a.py::Foo"])
    assert score_case(hits, case, k=2)["recall"] == 0.0  # target is at rank 3, beyond k=2
    assert score_case(hits, case, k=3)["recall"] == 1.0


def test_aggregate_means():
    agg = aggregate(
        [
            {"recall": 1.0, "rr": 1.0, "hit": 1.0},
            {"recall": 0.0, "rr": 0.0, "hit": 0.0},
        ]
    )
    assert agg["recall_at_k"] == 0.5
    assert agg["mrr"] == 0.5
    assert agg["hit_rate"] == 0.5
    assert agg["num_cases"] == 2.0


def test_load_cases_json(tmp_path: Path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps([{"question": "How does X work?", "relevant": ["a.py::X"]}]))
    cases = load_cases(p)
    assert len(cases) == 1
    assert cases[0].question == "How does X work?"
    assert cases[0].relevant == ["a.py::X"]
