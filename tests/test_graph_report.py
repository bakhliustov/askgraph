"""Tests for the graph analysis behind `status`, `report`, and the MCP tools.

These exercise the same functions the MCP server delegates to (god nodes,
communities, blast radius), so a missing import or broken algorithm is caught
here — the class of bug that previously made the MCP graph tools crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from askgraph.report.generator import (
    build_networkx_graph,
    compute_blast_radius,
    compute_god_nodes,
    compute_stats,
    detect_communities,
    find_surprising_connections,
    generate_html,
    generate_markdown_report,
    simple_communities,
)

SAMPLE_GRAPH = {
    "nodes": [
        {"id": "file:a.py", "type": "file", "path": "pkg/a.py"},
        {"id": "file:b.py", "type": "file", "path": "pkg/b.py"},
        {"id": "symbol:a.py:Foo", "type": "class", "name": "Foo", "path": "pkg/a.py"},
        {"id": "symbol:a.py:helper", "type": "function", "name": "helper", "path": "pkg/a.py"},
        {"id": "symbol:b.py:run", "type": "function", "name": "run", "path": "pkg/b.py"},
    ],
    "edges": [
        {"source": "file:a.py", "target": "symbol:a.py:Foo", "type": "contains"},
        {"source": "file:a.py", "target": "symbol:a.py:helper", "type": "contains"},
        {"source": "file:b.py", "target": "symbol:b.py:run", "type": "contains"},
        {"source": "file:b.py", "target": "symbol:a.py:Foo", "type": "imports"},
    ],
}


def test_build_networkx_graph():
    g = build_networkx_graph(SAMPLE_GRAPH)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4


def test_compute_god_nodes_ranks_by_degree():
    g = build_networkx_graph(SAMPLE_GRAPH)
    gods = compute_god_nodes(g, top_n=3)
    assert gods[0]["id"] in ("file:a.py", "symbol:a.py:Foo")  # most connected
    assert all("degree" in n and "name" in n for n in gods)


def test_compute_stats():
    g = build_networkx_graph(SAMPLE_GRAPH)
    stats = compute_stats(g, SAMPLE_GRAPH)
    assert stats["num_files"] == 2
    assert stats["num_symbols"] == 3
    assert stats["num_edges"] == 4


def test_detect_communities_runs():
    g = build_networkx_graph(SAMPLE_GRAPH)
    comms = detect_communities(g, min_size=1)
    assert isinstance(comms, list)
    assert sum(len(c) for c in comms) >= 1


def test_compute_blast_radius():
    g = build_networkx_graph(SAMPLE_GRAPH)
    radius = compute_blast_radius(g, "symbol:a.py:Foo", hops=1)
    # Foo is contained by a.py and imported by b.py → both within 1 hop.
    assert "file:a.py" in radius["nodes"]
    assert "file:b.py" in radius["nodes"]
    assert radius["hops"] == 1


def test_compute_blast_radius_unknown_node():
    g = build_networkx_graph(SAMPLE_GRAPH)
    radius = compute_blast_radius(g, "symbol:does.not:Exist", hops=2)
    assert radius["nodes"] == []


def test_simple_and_surprising():
    assert "pkg" in simple_communities(SAMPLE_GRAPH)
    # b.py imports a symbol outside its own top-level dir grouping — best-effort list.
    assert isinstance(
        find_surprising_connections(build_networkx_graph(SAMPLE_GRAPH), SAMPLE_GRAPH), list
    )


def test_generate_markdown_and_html(tmp_path: Path):
    g = build_networkx_graph(SAMPLE_GRAPH)
    md = generate_markdown_report(SAMPLE_GRAPH, g, tmp_path)
    assert "God Nodes" in md and "Overview" in md
    html = generate_html(SAMPLE_GRAPH, g, tmp_path)
    assert "<svg" in html and "Codebase Graph" in html


def test_mcp_server_imports_cleanly():
    # Guards against module-level NameError/ImportError (e.g. the missing `nx`
    # import that previously broke the MCP graph tools).
    pytest.importorskip("mcp")
    import askgraph.mcp_server as mcp

    assert hasattr(mcp, "server")
    assert callable(mcp.run_mcp_server)
