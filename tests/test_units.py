"""Fast unit tests for parsing, provenance, and retrieval helpers.

These don't require network access, an embedding model, or Ollama — they cover
the structural/graph logic directly so the core stays regression-safe.
"""

from __future__ import annotations

from askgraph.indexing.indexer import LocalIndexer
from askgraph.indexing.parsers import parse_file, parse_python
from askgraph.query.retriever import _rows_from_query


def test_parse_python_extracts_symbols():
    src = "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        pass\n"
    parsed = parse_python(src, "ex.py")
    names = {s["name"] for s in parsed["symbols"]}
    assert {"foo", "Bar", "baz"} <= names
    assert parsed["tree_sitter_ok"] is True


def test_parse_file_skips_unsupported_languages():
    # Regression: non-Python files used to be parsed with the Python grammar,
    # polluting the graph with bogus symbols. They should now yield no symbols.
    md = parse_file("# Title\n\nSome **markdown**.\n", "README.md")
    assert md["symbols"] == []
    assert md["tree_sitter_ok"] is False
    assert md["language"] == "md"

    go = parse_file("package main\nfunc main() {}\n", "main.go")
    assert go["symbols"] == []


def test_parse_file_handles_python():
    parsed = parse_file("def hello():\n    return 'hi'\n", "mod.py")
    assert any(s["name"] == "hello" for s in parsed["symbols"])


def test_symbol_history_is_most_recent_first_and_deduped():
    blame_map = {
        1: {
            "commit": "aaaa",
            "author": "Old",
            "date": "2020-01-01T00:00:00+00:00",
            "message": "init",
        },
        2: {
            "commit": "bbbb",
            "author": "New",
            "date": "2024-06-01T00:00:00+00:00",
            "message": "rework",
        },
        3: {
            "commit": "aaaa",
            "author": "Old",
            "date": "2020-01-01T00:00:00+00:00",
            "message": "init",
        },
    }
    history = LocalIndexer._symbol_history(blame_map, 1, 3)
    # Deduped to two unique commits, newest first.
    assert [h["commit"] for h in history] == ["bbbb", "aaaa"]


def test_symbol_history_empty_when_no_blame():
    assert LocalIndexer._symbol_history({}, 1, 10) == []


def test_rows_from_query_is_none_safe():
    # An empty/None Chroma result must not raise.
    assert _rows_from_query({"ids": [[]]}) == []
    assert _rows_from_query({}) == []


def test_rows_from_query_maps_fields_and_extra():
    results = {
        "ids": [["c1", "c2"]],
        "documents": [["doc one", "doc two"]],
        "metadatas": [[{"file_path": "a.py"}, None]],
        "distances": [[0.1, 0.2]],
    }
    rows = _rows_from_query(results, extra={"from_graph": True})
    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "c1"
    assert rows[0]["text"] == "doc one"
    assert rows[0]["metadata"] == {"file_path": "a.py"}
    assert rows[1]["metadata"] == {}  # None coerced to {}
    assert all(r["from_graph"] for r in rows)
