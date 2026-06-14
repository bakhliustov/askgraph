"""Regression tests for incremental (non-force) re-indexing.

These guard the two bugs that previously made a plain ``askgraph index .`` corrupt
an existing index:

1. Graph drift — the persisted graph used to be overwritten with *only* the
   changed file's nodes/edges, shrinking it on every incremental run.
2. Stale vectors — chunk ids are deterministic and the indexer used ``add`` (not
   upsert) without deleting first, so changed content kept its old embedding and
   chunks for deleted symbols/files were never removed.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from askgraph.config import get_index_path
from askgraph.indexing.indexer import _load_graph, index_codebase


def _graph(target: Path) -> dict:
    return _load_graph(get_index_path(target))


def _node_ids(target: Path) -> set[str]:
    return {n["id"] for n in _graph(target)["nodes"]}


def _file_paths(target: Path) -> set[str]:
    return {n["path"] for n in _graph(target)["nodes"] if n.get("type") == "file"}


def _chunks_for(target: Path, rel: str) -> list[str]:
    """Return the stored documents for a file's chunks, straight from Chroma."""
    client = chromadb.PersistentClient(path=str(get_index_path(target) / "chroma"))
    col = client.get_collection("askgraph_code")
    res = col.get(where={"file_path": rel})
    return res.get("documents") or []


def test_incremental_reindex_keeps_graph_and_refreshes_vectors(tmp_path: Path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('def alpha():\n    return "ORIGINAL_ALPHA"\n\n\ndef beta():\n    return "BETA"\n')
    b.write_text('def gamma():\n    return "GAMMA"\n')

    # First (non-force) index: both files land in the graph and Chroma.
    index_codebase(tmp_path, force=False, show_progress=False)
    assert _file_paths(tmp_path) == {"a.py", "b.py"}
    assert {"symbol:a.py:alpha", "symbol:a.py:beta", "symbol:b.py:gamma"} <= _node_ids(tmp_path)
    assert any("ORIGINAL_ALPHA" in d for d in _chunks_for(tmp_path, "a.py"))

    # Change only a.py (rewrite alpha's body, drop beta), re-index WITHOUT --force.
    a.write_text('def alpha():\n    return "UPDATED_ALPHA"\n')
    index_codebase(tmp_path, force=False, show_progress=False)

    # Graph drift bug: b.py must survive, not get wiped by the partial rebuild.
    assert _file_paths(tmp_path) == {"a.py", "b.py"}
    ids = _node_ids(tmp_path)
    assert "symbol:b.py:gamma" in ids  # untouched file preserved
    assert "symbol:a.py:alpha" in ids
    assert "symbol:a.py:beta" not in ids  # deleted symbol pruned from the graph

    # Stale vector bug: a.py's chunk must hold the NEW content, with no leftover
    # chunks for the removed symbol — and the count must not have grown.
    a_chunks = _chunks_for(tmp_path, "a.py")
    assert len(a_chunks) == 1
    assert any("UPDATED_ALPHA" in d for d in a_chunks)
    assert all("ORIGINAL_ALPHA" not in d and "BETA" not in d for d in a_chunks)
    # b.py was untouched: its vectors stay intact.
    assert any("GAMMA" in d for d in _chunks_for(tmp_path, "b.py"))


def test_incremental_reindex_no_changes_is_idempotent(tmp_path: Path):
    (tmp_path / "a.py").write_text('def alpha():\n    return "ALPHA"\n')
    index_codebase(tmp_path, force=False, show_progress=False)
    before = len(_chunks_for(tmp_path, "a.py"))

    # Re-running with no changes must be a no-op (no growth, reported up-to-date).
    stats = index_codebase(tmp_path, force=False, show_progress=False)
    assert stats.get("status") == "up-to-date"
    assert len(_chunks_for(tmp_path, "a.py")) == before


def test_incremental_reindex_handles_deletions(tmp_path: Path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('def alpha():\n    return "ALPHA"\n')
    b.write_text('def gamma():\n    return "GAMMA"\n')
    index_codebase(tmp_path, force=False, show_progress=False)
    assert _file_paths(tmp_path) == {"a.py", "b.py"}

    # Delete b.py and re-index without --force.
    b.unlink()
    stats = index_codebase(tmp_path, force=False, show_progress=False)
    assert stats.get("files_deleted") == 1

    # The deleted file's nodes and chunks are gone; a.py is untouched.
    assert _file_paths(tmp_path) == {"a.py"}
    assert "symbol:b.py:gamma" not in _node_ids(tmp_path)
    assert _chunks_for(tmp_path, "b.py") == []
    assert any("ALPHA" in d for d in _chunks_for(tmp_path, "a.py"))
