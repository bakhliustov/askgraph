"""Basic tests for askgraph core paths."""

from pathlib import Path

from askgraph.indexing.indexer import index_codebase
from askgraph.query.retriever import LocalRetriever
from askgraph.report.generator import build_networkx_graph, generate_markdown_report


def test_index_and_retrieve(tmp_path: Path):
    # Create a tiny python file
    code = tmp_path / "example.py"
    code.write_text("""
def hello():
    return "world"

class Foo:
    def bar(self):
        pass
""")
    stats = index_codebase(tmp_path, force=True)
    assert stats["files_indexed"] >= 1
    assert stats["graph_nodes"] > 0

    retr = LocalRetriever(tmp_path)
    hits = retr.retrieve_hybrid("how does hello work", top_k=3)
    assert len(hits) > 0
    assert any("hello" in h["text"].lower() for h in hits)


def test_graph_and_report(tmp_path: Path):
    # Minimal graph data
    graph_data = {
        "nodes": [
            {"id": "file:ex.py", "type": "file", "path": "ex.py"},
            {"id": "symbol:ex.py:hello", "type": "function", "name": "hello", "path": "ex.py"},
        ],
        "edges": [{"source": "file:ex.py", "target": "symbol:ex.py:hello", "type": "contains"}],
    }
    G = build_networkx_graph(graph_data)
    report = generate_markdown_report(graph_data, G, tmp_path)
    assert "God Nodes" in report
    assert "Communities" in report or "Overview" in report  # at least basic sections
