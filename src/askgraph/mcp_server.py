"""MCP server for askgraph - makes the tool natively usable by AI coding agents.

Run with: askgraph mcp [path]

Exposes tools like:
- index_codebase
- ask (with full hybrid + provenance)
- get_god_nodes / graph_summary
- export_agent_context (rich, token-aware pack for agents)

Fully local-first. Agents get structured, low-token context with real code relationships and git history.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from askgraph.config import get_index_path
from askgraph.indexing.indexer import index_codebase
from askgraph.query import LocalRetriever, synthesize_answer
from askgraph.query.graph import format_graph_expansion, load_graph
from askgraph.report.generator import (
    build_networkx_graph,
    detect_communities,
    generate_artifacts,
)
from askgraph.report.generator import (
    compute_blast_radius as _compute_blast_radius,
)

# Create the MCP server
server = Server("askgraph")

# Default codebase root the server was launched against. Tool calls may still
# override this per-call via their "path" argument.
_DEFAULT_TARGET = Path(".")


def _resolve_target(arguments: dict[str, Any]) -> Path:
    """Resolve the codebase path for a tool call, falling back to the server default."""
    raw = arguments.get("path")
    base = Path(raw) if raw else _DEFAULT_TARGET
    return base.resolve()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for agents."""
    return [
        Tool(
            name="index_codebase",
            description="Index a local codebase (builds structural graph + semantic index). Run this first if no index exists. Returns summary stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the codebase root. Defaults to current directory.",
                        "default": ".",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force full re-index even if one exists.",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="ask",
            description=(
                "Ask a natural language question about the indexed codebase. "
                "Uses hybrid retrieval (embeddings + structural graph) + LLM synthesis with git provenance. "
                "Returns a clear answer with citations and provenance info. Perfect for agents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the codebase (e.g. 'How does dependency injection work and when was it introduced?').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the codebase (must have been indexed). Defaults to current dir.",
                        "default": ".",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve before synthesis.",
                        "default": 8,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="get_god_nodes",
            description="Return the top 'god nodes' (most connected/central symbols and files) from the structural graph. Great for understanding architecture hotspots and suggesting questions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": [],
            },
        ),
        Tool(
            name="get_communities",
            description="Detect communities in the code graph (modularity-based). Useful for understanding modules and subsystems.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                },
                "required": [],
            },
        ),
        Tool(
            name="compute_blast_radius",
            description="Compute the blast radius / impact of a symbol or file (nodes within N hops). Helps with refactoring risk and change impact analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_id": {
                        "type": "string",
                        "description": "Node id like 'symbol:path/to/file.py:FunctionName' or file id.",
                    },
                    "hops": {"type": "integer", "default": 2},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["start_id"],
            },
        ),
        Tool(
            name="export_agent_context",
            description=(
                "Export a rich, agent-optimized context bundle for a question. "
                "Includes hybrid-retrieved chunks, relevant subgraph, git provenance, and a suggested prompt template. "
                "Low token count, highly structured - ideal for dropping into another agent session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "top_k": {"type": "integer", "default": 12},
                    "include_graph": {"type": "boolean", "default": True},
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch tool calls from agents."""
    if name == "index_codebase":
        target = _resolve_target(arguments)
        force = arguments.get("force", False)
        stats = index_codebase(target, force=force)
        # Auto-generate artifacts for agents
        index_dir = get_index_path(target)
        if (index_dir / "graph.json").exists():
            with contextlib.suppress(Exception):
                generate_artifacts(index_dir)
        return [
            TextContent(
                type="text",
                text=f"Indexed {stats.get('files_indexed', 0)} files, {stats.get('chunks_added', 0)} chunks. "
                f"Graph: {stats.get('graph_nodes', 0)} nodes, {stats.get('graph_edges', 0)} edges. "
                f"Artifacts ready in {index_dir}.",
            )
        ]

    elif name == "ask":
        question = arguments["question"]
        target = _resolve_target(arguments)
        top_k = arguments.get("top_k", 8)

        retriever = LocalRetriever(target)
        hits = retriever.retrieve_hybrid(question, top_k=top_k, expand=True)

        graph_data = load_graph(get_index_path(target))
        graph_context = ""
        if graph_data:
            graph_context = format_graph_expansion(graph_data, hits)

        answer = synthesize_answer(question, hits, index_dir=get_index_path(target))

        # Structured output for agents
        sources = []
        for h in hits[:5]:
            meta = h.get("metadata", {})
            src = {
                "file": meta.get("file_path"),
                "symbol": meta.get("symbol"),
                "lines": f"L{meta.get('start_line')}-{meta.get('end_line')}",
                "snippet": h["text"][:300],
            }
            # Attach git provenance if present in graph
            if graph_data and meta.get("symbol"):
                sym_id = f"symbol:{meta['file_path']}:{meta['symbol']}"
                for node in graph_data.get("nodes", []):
                    if node.get("id") == sym_id and "git" in node:
                        src["provenance"] = node["git"]
                        break
            sources.append(src)

        result = {
            "answer": answer,
            "sources": sources,
            "graph_context": graph_context,
            "question": question,
        }
        return [TextContent(type="text", text=str(result))]

    elif name == "get_god_nodes":
        target = _resolve_target(arguments)
        limit = arguments.get("limit", 8)
        graph_data = load_graph(get_index_path(target))
        if not graph_data:
            return [TextContent(type="text", text="No graph found. Run index_codebase first.")]

        # Simple degree-based god nodes (same as TUI/CLI)
        from collections import Counter

        edges = graph_data.get("edges", [])
        degrees: Counter[str] = Counter()
        for e in edges:
            degrees[e.get("source", "")] += 1
            degrees[e.get("target", "")] += 1

        nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
        top = []
        for nid, deg in sorted(degrees.items(), key=lambda x: -x[1])[:limit]:
            n = nodes.get(nid, {})
            entry = {
                "id": nid,
                "name": n.get("name") or n.get("path") or nid,
                "type": n.get("type"),
                "degree": deg,
            }
            if "git" in n:
                entry["provenance"] = n["git"]
            top.append(entry)

        return [TextContent(type="text", text=str({"god_nodes": top}))]

    elif name == "get_communities":
        target = _resolve_target(arguments)
        graph_data = load_graph(get_index_path(target))
        if not graph_data:
            return [TextContent(type="text", text="No graph found. Run index_codebase first.")]
        graph = build_networkx_graph(graph_data)
        comms = detect_communities(graph)
        nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
        communities: list[dict[str, Any]] = []
        for i, comm in enumerate(comms[:6], 1):
            names = [
                nodes.get(nid, {}).get("name") or nodes.get(nid, {}).get("path") or nid
                for nid in comm[:6]
            ]
            communities.append({"community": i, "size": len(comm), "examples": names})
        return [TextContent(type="text", text=str({"communities": communities}))]

    elif name == "compute_blast_radius":
        start_id = arguments["start_id"]
        hops = arguments.get("hops", 2)
        target = _resolve_target(arguments)
        graph_data = load_graph(get_index_path(target))
        if not graph_data:
            return [TextContent(type="text", text="No graph found. Run index_codebase first.")]
        graph = build_networkx_graph(graph_data)
        radius = _compute_blast_radius(graph, start_id, hops)
        return [TextContent(type="text", text=str(radius))]

    elif name == "export_agent_context":
        question = arguments["question"]
        target = _resolve_target(arguments)
        top_k = arguments.get("top_k", 12)
        include_graph = arguments.get("include_graph", True)

        retriever = LocalRetriever(target)
        hits = retriever.retrieve_hybrid(question, top_k=top_k, expand=True)

        graph_data = load_graph(get_index_path(target)) if include_graph else None
        graph_context = format_graph_expansion(graph_data, hits) if graph_data else ""

        answer = synthesize_answer(question, hits, index_dir=get_index_path(target))

        # Rich agent pack
        pack = {
            "question": question,
            "synthesized_answer": answer,
            "hybrid_sources": [
                {
                    "file": h["metadata"].get("file_path"),
                    "symbol": h["metadata"].get("symbol"),
                    "lines": f"L{h['metadata'].get('start_line')}-{h['metadata'].get('end_line')}",
                    "text": h["text"],
                    "provenance": next(
                        (
                            n.get("git_history") or n.get("git")
                            for n in (graph_data or {}).get("nodes", [])
                            if n.get("id")
                            == f"symbol:{h['metadata'].get('file_path')}:{h['metadata'].get('symbol')}"
                            and ("git_history" in n or "git" in n)
                        ),
                        None,
                    ),
                }
                for h in hits
            ],
            "graph_context": graph_context,
            "suggested_prompt_template": (
                "Use the following structured codebase context (with provenance) to answer questions accurately and cite sources:\n\n"
                f"{answer}\n\nSources and graph above."
            ),
            "token_estimate": len(str(hits)) // 4 + len(answer) // 4,  # rough
        }
        return [TextContent(type="text", text=str(pack))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def run_mcp_server(target: Path | None = None) -> None:
    """Run the MCP server over stdio (for agent hosts).

    The launch target becomes the default codebase for tool calls that don't
    pass an explicit ``path`` argument.
    """
    global _DEFAULT_TARGET
    _DEFAULT_TARGET = (target or Path(".")).resolve()

    import asyncio

    asyncio.run(_run_server())


async def _run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    run_mcp_server()
