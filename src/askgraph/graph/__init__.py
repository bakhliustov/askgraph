"""Graph utilities for askgraph.

Consolidated module for loading, adjacency, expansion, god nodes,
communities, and blast radius analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


def load_graph(index_dir: Path) -> dict[str, Any] | None:
    """Load graph.json if it exists."""
    graph_path = index_dir / "graph.json"
    if not graph_path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(graph_path.read_text())
        return data
    except Exception:
        return None


def build_adjacency(graph_data: dict[str, Any]) -> dict[str, list[str]]:
    """Build simple adjacency list from edges."""
    adj: dict[str, list[str]] = {}
    for edge in graph_data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src and tgt:
            adj.setdefault(src, []).append(tgt)
            adj.setdefault(tgt, []).append(src)
    return adj


def build_networkx_graph(graph_data: dict[str, Any]) -> nx.Graph:
    """Build a networkx Graph from graph.json data."""
    G = nx.Graph()
    for node in graph_data.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in graph_data.get("edges", []):
        G.add_edge(edge.get("source"), edge.get("target"), **edge)
    return G


def get_related_entities(
    graph_data: dict[str, Any] | None,
    file_path: str,
    symbol: str | None = None,
    max_neighbors: int = 6,
) -> list[dict[str, Any]]:
    """Return related entities from the structural graph."""
    if not graph_data:
        return []
    nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
    adj = build_adjacency(graph_data)

    related: list[dict[str, Any]] = []
    seen = set()

    file_id = f"file:{file_path}"
    if file_id in adj:
        for neigh_id in adj[file_id][:max_neighbors]:
            if neigh_id in seen:
                continue
            seen.add(neigh_id)
            node = nodes.get(neigh_id, {"id": neigh_id})
            related.append(
                {
                    "id": neigh_id,
                    "type": node.get("type", "unknown"),
                    "name": node.get("name") or node.get("path") or neigh_id,
                    "path": node.get("path"),
                    "relation": "contains" if node.get("type") in ("function", "class") else "related",
                }
            )

    if symbol:
        sym_id = f"symbol:{file_path}:{symbol}"
        if sym_id in adj:
            for neigh_id in adj[sym_id][:max_neighbors]:
                if neigh_id in seen:
                    continue
                seen.add(neigh_id)
                node = nodes.get(neigh_id, {"id": neigh_id})
                related.append(
                    {
                        "id": neigh_id,
                        "type": node.get("type", "unknown"),
                        "name": node.get("name") or neigh_id,
                        "path": node.get("path"),
                        "relation": "related",
                    }
                )
    return related[:max_neighbors]
