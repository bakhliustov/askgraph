"""Graph loading and expansion for hybrid retrieval.

Loads the structural graph.json produced during indexing and uses it to
expand semantic hits with related entities (neighbors, siblings in same file,
cross-file imports, etc.). This provides the "structure not just similarity"
benefit inspired by Graphify.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from askgraph.utils.logging import get_logger

logger = get_logger(__name__)


def load_graph(index_dir: Path) -> dict[str, Any] | None:
    """Load graph.json if it exists."""
    graph_path = index_dir / "graph.json"
    if not graph_path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(graph_path.read_text())
        return data
    except Exception as e:
        logger.warning("Failed to load graph.json: %s", e)
        return None


def build_adjacency(graph_data: dict[str, Any]) -> dict[str, list[str]]:
    """Build simple adjacency list from edges."""
    adj: dict[str, list[str]] = {}
    for edge in graph_data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src and tgt:
            adj.setdefault(src, []).append(tgt)
            adj.setdefault(tgt, []).append(src)  # undirected for neighborhood
    return adj


def get_related_entities(
    graph_data: dict[str, Any] | None,
    file_path: str,
    symbol: str | None = None,
    max_neighbors: int = 6,
) -> list[dict[str, Any]]:
    """Return related entities (neighbors) for a file/symbol from the graph."""
    if not graph_data:
        return []

    nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
    adj = build_adjacency(graph_data)

    related: list[dict[str, Any]] = []
    seen = set()

    # Start from the file node
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
                    "relation": "contains"
                    if node.get("type") in ("function", "class")
                    else "related",
                }
            )

    # If we have a specific symbol, expand from it too
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


def format_graph_expansion(
    graph_data: dict[str, Any] | None,
    hits: list[dict[str, Any]],
    max_per_hit: int = 3,
) -> str:
    """Produce a compact string of graph context for the top hits.

    This gets injected into the LLM prompt for better structural awareness.
    """
    if not graph_data or not hits:
        return ""

    lines = ["Additional structural context from the codebase graph:"]
    for hit in hits[:3]:  # only top few
        meta = hit.get("metadata", {})
        fp = meta.get("file_path")
        sym = meta.get("symbol")
        if not fp:
            continue

        related = get_related_entities(graph_data, fp, sym, max_neighbors=max_per_hit)
        if not related:
            continue

        loc = fp
        if sym:
            loc += f" :: {sym}"

        rel_strs = []
        for r in related:
            name = r.get("name", r["id"])
            rel_type = r.get("relation", "related")
            rel_strs.append(f"{name} ({rel_type})")

        if rel_strs:
            lines.append(f"- {loc} is connected to: {', '.join(rel_strs)}")

    return "\n".join(lines) if len(lines) > 1 else ""
