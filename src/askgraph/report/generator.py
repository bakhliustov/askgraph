"""Generate GRAPH_REPORT.md and self-contained graph.html from graph.json.

Inspired by Graphify's excellent artifacts: high-signal markdown summary
+ beautiful offline interactive visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from rich.console import Console

from askgraph.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def load_graph(index_dir: Path) -> dict[str, Any]:
    graph_path = index_dir / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph.json found at {graph_path}. Run `askgraph index` first.")
    data: dict[str, Any] = json.loads(graph_path.read_text())
    return data


def build_networkx_graph(graph_data: dict[str, Any]) -> nx.Graph:
    G = nx.Graph()
    for node in graph_data.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in graph_data.get("edges", []):
        G.add_edge(edge.get("source"), edge.get("target"), **edge)
    return G


def compute_god_nodes(G: nx.Graph, top_n: int = 8) -> list[dict[str, Any]]:
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    gods = []
    for node_id, degree in sorted_nodes:
        attrs = G.nodes.get(node_id, {})
        gods.append(
            {
                "id": node_id,
                "degree": degree,
                "name": attrs.get("name") or attrs.get("path") or node_id,
                "type": attrs.get("type", "unknown"),
                "path": attrs.get("path"),
            }
        )
    return gods


def compute_stats(G: nx.Graph, graph_data: dict) -> dict[str, Any]:
    files = [n for n in graph_data.get("nodes", []) if n.get("type") == "file"]
    symbols = [n for n in graph_data.get("nodes", []) if n.get("type") in ("function", "class")]
    return {
        "num_files": len(files),
        "num_symbols": len(symbols),
        "num_nodes": len(G.nodes()),
        "num_edges": len(G.edges()),
        "avg_degree": round(sum(dict(G.degree()).values()) / max(len(G), 1), 2),
        "density": round(nx.density(G), 4),
    }


def simple_communities(graph_data: dict) -> dict[str, list[str]]:
    """Group by top-level directory for a simple community view."""
    communities: dict[str, list[str]] = {}
    for node in graph_data.get("nodes", []):
        if node.get("type") != "file":
            continue
        path = node.get("path", "")
        top = path.split("/")[0] if "/" in path else path
        communities.setdefault(top, []).append(path)
    return {k: v for k, v in communities.items() if len(v) > 0}


def detect_communities(G: nx.Graph, min_size: int = 2) -> list[list[str]]:
    """Detect communities using greedy modularity (networkx built-in, no extra deps)."""
    try:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(G))
        return [list(c) for c in communities if len(c) >= min_size]
    except Exception:
        # Fallback to connected components if modularity fails
        return [list(c) for c in nx.connected_components(G) if len(c) >= min_size]


def compute_blast_radius(G: nx.Graph, start_id: str, hops: int = 2) -> dict[str, Any]:
    """Compute nodes within 'hops' distance from start_id (impact / blast radius)."""
    if start_id not in G:
        return {"nodes": [], "edges": 0, "hops": hops}
    nodes_at_dist = {0: {start_id}}
    visited = {start_id}
    current = {start_id}
    for d in range(1, hops + 1):
        next_layer = set()
        for node in current:
            for neigh in G.neighbors(node):
                if neigh not in visited:
                    next_layer.add(neigh)
                    visited.add(neigh)
        nodes_at_dist[d] = next_layer
        current = next_layer
    all_nodes = set()
    for layer in nodes_at_dist.values():
        all_nodes.update(layer)
    induced = G.subgraph(all_nodes)
    return {
        "start": start_id,
        "hops": hops,
        "nodes": sorted(list(all_nodes)),
        "num_nodes": len(all_nodes),
        "num_edges": len(induced.edges()),
        "by_distance": {d: sorted(list(s)) for d, s in nodes_at_dist.items()},
    }


def find_surprising_connections(G: nx.Graph, graph_data: dict, max_examples: int = 6) -> list[str]:
    """Find imports that cross major directory boundaries."""
    surprising = []
    node_lookup = {n["id"]: n for n in graph_data.get("nodes", [])}
    for u, v, data in G.edges(data=True):
        if data.get("type") != "imports":
            continue
        u_attrs = node_lookup.get(u, {})
        u_path = u_attrs.get("path", "")
        if not u_path or not isinstance(v, str):
            continue
        u_top = u_path.split("/")[0]
        if u_top and u_top not in v and len(u_path.split("/")) > 1:
            surprising.append(f"{u_path} imports something outside its top dir ({u_top}): {v}")
            if len(surprising) >= max_examples:
                break
    return surprising[:max_examples]


def generate_markdown_report(graph_data: dict, G: nx.Graph, index_dir: Path) -> str:
    stats = compute_stats(G, graph_data)
    god_nodes = compute_god_nodes(G)
    communities = simple_communities(graph_data)
    deeper_communities = detect_communities(G)
    surprising = find_surprising_connections(G, graph_data)

    # Example blast radius on top god node
    blast = {}
    if god_nodes:
        top_id = god_nodes[0]["id"]
        blast = compute_blast_radius(G, top_id, hops=2)

    lines = []
    lines.append("# Codebase Graph Report\n")
    lines.append(f"_Generated by askgraph from `{index_dir}`_\n")

    lines.append("## Overview\n")
    lines.append(f"- **Files**: {stats['num_files']}")
    lines.append(f"- **Symbols (functions + classes)**: {stats['num_symbols']}")
    lines.append(f"- **Total graph nodes**: {stats['num_nodes']}")
    lines.append(f"- **Edges**: {stats['num_edges']}")
    lines.append(f"- **Average degree**: {stats['avg_degree']}")
    lines.append(f"- **Graph density**: {stats['density']}\n")

    lines.append("## God Nodes (highest connectivity)\n")
    lines.append("These are the most central pieces of the codebase:\n")
    nodes_by_id = {n["id"]: n for n in graph_data.get("nodes", [])}
    for g in god_nodes:
        path_info = f" ({g['path']})" if g.get("path") else ""
        line = f"- **{g['name']}** — {g['type']}{path_info} (degree: {g['degree']})"
        node = nodes_by_id.get(g["id"], {})
        git_history = node.get("git_history", [])
        if git_history:
            prov = git_history[0]
            line += f"\n  last touched: {prov.get('commit')} by {prov.get('author')} on {prov.get('date')[:10]} — {prov.get('message')}"
            if len(git_history) > 1:
                line += f" (+{len(git_history) - 1} prior changes)"
        elif node.get("git"):
            git = node["git"]
            line += f"\n  last touched: {git.get('commit')} by {git.get('author')} on {git.get('date')[:10]} — {git.get('message')}"
        lines.append(line)
    lines.append("")

    if communities:
        lines.append("## Simple Communities (by top-level directory)\n")
        for comm, files in sorted(communities.items(), key=lambda x: -len(x[1]))[:8]:
            lines.append(f"- **{comm}/** — {len(files)} files")
        lines.append("")

    if deeper_communities:
        lines.append("## Detected Communities (graph modularity)\n")
        for i, community in enumerate(deeper_communities[:5], 1):
            names = []
            for nid in community[:5]:
                n = nodes_by_id.get(nid, {})
                names.append(n.get("name") or n.get("path") or nid)
            suffix = "..." if len(community) > 5 else ""
            lines.append(f"- Community {i}: {', '.join(names)}{suffix} ({len(community)} nodes)")
        lines.append("")

    if blast and blast.get("num_nodes", 0) > 1:
        lines.append("## Example Blast Radius (impact of top god node, 2 hops)\n")
        lines.append(
            f"Starting from {blast['start']} affects ~{blast['num_nodes']} nodes and {blast['num_edges']} relationships within {blast['hops']} hops.\n"
        )

    if surprising:
        lines.append("## Interesting Cross-Cutting Connections\n")
        for s in surprising:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## Suggested Questions to Ask\n")
    suggestions = [
        "How does the main entry point interact with the core modules?",
        "What are the most depended-upon utilities?",
        "Where is configuration loaded and used?",
        "Are there any god nodes that might be good candidates for refactoring?",
    ]
    for q in suggestions:
        lines.append(f"- `{q}`")
    lines.append("")

    lines.append("## How to use this report\n")
    lines.append(
        'Run `askgraph ask "your question"` (or `askgraph tui` / MCP) for natural language exploration with git provenance.'
    )
    lines.append(
        "The accompanying `graph.html` provides an interactive visual map. Agents: use `export_agent_context` or the MCP `get_communities` / `compute_blast_radius` tools.\n"
    )

    return "\n".join(lines)


def generate_html(graph_data: dict, G: nx.Graph, index_dir: Path) -> str:
    """Self-contained interactive HTML visualization.

    Uses Tailwind (CDN for convenience + polish) + pure SVG + vanilla JS.
    Positions pre-computed with networkx spring layout for a good starting point.
    Fully usable offline after the first load.
    """
    try:
        pos = nx.spring_layout(G, seed=42, k=1.2 / (len(G) ** 0.5 + 1), iterations=50)
    except Exception:
        pos = {node: (i % 20 * 0.05, i // 20 * 0.08) for i, node in enumerate(G.nodes())}

    min_x = min(p[0] for p in pos.values())
    max_x = max(p[0] for p in pos.values())
    min_y = min(p[1] for p in pos.values())
    max_y = max(p[1] for p in pos.values())
    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)
    scale = 900 / max(width, height)

    nodes_js = []
    for node_id, (x, y) in pos.items():
        attrs = G.nodes.get(node_id, {})
        ntype = attrs.get("type", "unknown")
        label = attrs.get("name") or attrs.get("path") or node_id.split(":")[-1]
        px = (x - min_x) * scale + 50
        py = (y - min_y) * scale + 50
        size = 18 if ntype == "file" else (12 if ntype in ("function", "class") else 8)
        color = "#3b82f6" if ntype == "file" else ("#10b981" if ntype == "function" else "#8b5cf6")
        nodes_js.append(
            {
                "id": node_id,
                "x": round(px, 1),
                "y": round(py, 1),
                "label": label[:40],
                "type": ntype,
                "size": size,
                "color": color,
                "path": attrs.get("path", ""),
            }
        )

    edges_js = []
    for u, v in G.edges():
        if u in pos and v in pos:
            edges_js.append({"u": u, "v": v})

    nodes_json = json.dumps(nodes_js)
    edges_json = json.dumps(edges_js)
    stats = compute_stats(G, graph_data)

    # Safe template + replace
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>askgraph — Codebase Visualization</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: ui-sans-serif, system-ui, sans-serif; }
        .node { cursor: pointer; transition: all 0.1s; }
        .node:hover { stroke: #fff; stroke-width: 3; }
        .edge { stroke: #64748b; stroke-opacity: 0.35; }
        .highlight { stroke: #f59e0b !important; stroke-opacity: 0.9; stroke-width: 2.5; }
        .sidebar { max-height: 70vh; overflow-y: auto; }
    </style>
</head>
<body class="bg-zinc-950 text-zinc-200">
    <div class="max-w-[1400px] mx-auto p-6">
        <div class="flex items-center justify-between mb-6">
            <div>
                <h1 class="text-3xl font-semibold tracking-tight">Codebase Graph</h1>
                <p class="text-zinc-400 text-sm mt-1">Generated by askgraph • INDEX_DIR</p>
            </div>
            <div class="flex gap-3 text-sm">
                <div class="px-3 py-1 rounded bg-zinc-900 border border-zinc-800">
                    STATS_FILES files • STATS_SYMBOLS symbols • STATS_EDGES edges
                </div>
                <button onclick="resetView()" 
                        class="px-3 py-1 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-sm">
                    Reset View
                </button>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div class="lg:col-span-8 bg-zinc-900 border border-zinc-800 rounded-2xl p-4">
                <div class="flex items-center gap-4 mb-3 text-sm">
                    <div class="flex items-center gap-2">
                        <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                        <span class="text-zinc-400">File</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="w-3 h-3 rounded-full bg-emerald-500"></div>
                        <span class="text-zinc-400">Function</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="w-3 h-3 rounded-full bg-violet-500"></div>
                        <span class="text-zinc-400">Class</span>
                    </div>
                    <div class="ml-auto flex items-center gap-2 text-xs text-zinc-500">
                        Click nodes • Drag to move • Use search
                    </div>
                </div>
                <svg id="graph" width="100%" height="620" class="bg-zinc-950 rounded-xl border border-zinc-800"></svg>
            </div>

            <div class="lg:col-span-4">
                <div class="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 sidebar">
                    <input id="search" type="text" placeholder="Search nodes..." 
                           class="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm mb-4 focus:outline-none focus:border-zinc-500">
                    
                    <div class="mb-4">
                        <div class="text-xs uppercase tracking-widest text-zinc-500 mb-1.5">Filters</div>
                        <div class="flex flex-wrap gap-2 text-sm" id="filters">
                            <label class="flex items-center gap-1.5 cursor-pointer">
                                <input type="checkbox" checked data-type="file" class="accent-blue-500"> 
                                <span class="text-blue-400">Files</span>
                            </label>
                            <label class="flex items-center gap-1.5 cursor-pointer">
                                <input type="checkbox" checked data-type="function" class="accent-emerald-500"> 
                                <span class="text-emerald-400">Functions</span>
                            </label>
                            <label class="flex items-center gap-1.5 cursor-pointer">
                                <input type="checkbox" checked data-type="class" class="accent-violet-500"> 
                                <span class="text-violet-400">Classes</span>
                            </label>
                        </div>
                    </div>

                    <div id="node-details" class="text-sm">
                        <div class="text-zinc-400 text-xs">Click a node to see details and neighbors.</div>
                    </div>
                </div>

                <div class="mt-4 text-[10px] text-zinc-500 px-1">
                    This visualization is 100% self-contained. 
                    Data comes from the structural graph built during <code>askgraph index</code>.
                </div>
            </div>
        </div>
    </div>

    <script>
        const NODES = NODES_JSON;
        const EDGES = EDGES_JSON;
        const svg = document.getElementById('graph');
        const width = 1000;
        const height = 620;
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        function draw() {
            svg.innerHTML = '';
            const gEdges = document.createElementNS("http://www.w3.org/2000/svg", "g");
            EDGES.forEach(e => {
                const u = NODES.find(n => n.id === e.u);
                const v = NODES.find(n => n.id === e.v);
                if (!u || !v) return;
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute('x1', u.x);
                line.setAttribute('y1', u.y);
                line.setAttribute('x2', v.x);
                line.setAttribute('y2', v.y);
                line.setAttribute('class', 'edge');
                line.setAttribute('stroke-width', '1.5');
                gEdges.appendChild(line);
            });
            svg.appendChild(gEdges);

            const gNodes = document.createElementNS("http://www.w3.org/2000/svg", "g");
            NODES.forEach(node => {
                const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
                g.setAttribute('class', 'node');
                g.setAttribute('data-id', node.id);
                g.setAttribute('data-type', node.type);

                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute('cx', node.x);
                circle.setAttribute('cy', node.y);
                circle.setAttribute('r', node.size);
                circle.setAttribute('fill', node.color);
                circle.setAttribute('stroke', '#18181b');
                circle.setAttribute('stroke-width', '2');

                const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                text.setAttribute('x', node.x + node.size + 4);
                text.setAttribute('y', node.y + 4);
                text.setAttribute('fill', '#a1a1aa');
                text.setAttribute('font-size', '9.5');
                text.textContent = node.label;

                g.appendChild(circle);
                g.appendChild(text);

                g.onclick = () => showNodeDetails(node.id);
                g.onmousedown = (ev) => dragNode(ev, node, circle, text, g);

                gNodes.appendChild(g);
            });
            svg.appendChild(gNodes);
        }

        function dragNode(ev, node, circle, text, g) {
            const startX = ev.clientX;
            const startY = ev.clientY;
            const origX = node.x;
            const origY = node.y;

            function move(e) {
                const dx = (e.clientX - startX) * 1.2;
                const dy = (e.clientY - startY) * 1.2;
                node.x = origX + dx;
                node.y = origY + dy;
                circle.setAttribute('cx', node.x);
                circle.setAttribute('cy', node.y);
                text.setAttribute('x', node.x + node.size + 4);
                text.setAttribute('y', node.y + 4);
                redrawEdges();
            }
            function up() {
                document.removeEventListener('mousemove', move);
                document.removeEventListener('mouseup', up);
            }
            document.addEventListener('mousemove', move);
            document.addEventListener('mouseup', up, { once: true });
        }

        function redrawEdges() { draw(); }

        function showNodeDetails(nodeId) {
            const node = NODES.find(n => n.id === nodeId);
            if (!node) return;
            const container = document.getElementById('node-details');
            let html = `<div class="mb-3"><span class="font-semibold text-lg">${node.label}</span>`;
            html += ` <span class="text-xs px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">${node.type}</span></div>`;
            if (node.path) html += `<div class="text-xs text-zinc-400 mb-2 font-mono">${node.path}</div>`;
            const neighbors = [];
            EDGES.forEach(e => { if (e.u === nodeId) neighbors.push(e.v); if (e.v === nodeId) neighbors.push(e.u); });
            const neighborNodes = neighbors.map(id => NODES.find(n => n.id === id)).filter(Boolean);
            if (neighborNodes.length) {
                html += `<div class="text-xs uppercase tracking-widest text-zinc-500 mt-3 mb-1">Connected to</div>`;
                neighborNodes.slice(0, 12).forEach(n => {
                    html += `<div class="text-xs py-0.5 cursor-pointer hover:text-white" onclick="showNodeDetails('${n.id}')">${n.label} <span class="text-[10px] text-zinc-500">(${n.type})</span></div>`;
                });
            }
            container.innerHTML = html;
            highlightNode(nodeId);
        }

        function highlightNode(nodeId) {
            document.querySelectorAll('.node').forEach(el => el.classList.remove('highlight'));
            const target = document.querySelector(`.node[data-id="${nodeId}"]`);
            if (target) target.classList.add('highlight');
        }

        function filterNodes() {
            const search = document.getElementById('search').value.toLowerCase();
            const activeTypes = Array.from(document.querySelectorAll('#filters input:checked')).map(i => i.dataset.type);
            document.querySelectorAll('.node').forEach(g => {
                const type = g.dataset.type;
                const id = g.dataset.id;
                const node = NODES.find(n => n.id === id);
                const label = node ? node.label.toLowerCase() : '';
                const matchesType = activeTypes.includes(type);
                const matchesSearch = !search || label.includes(search);
                g.style.display = (matchesType && matchesSearch) ? '' : 'none';
            });
        }

        function resetView() {
            document.getElementById('search').value = '';
            document.querySelectorAll('#filters input').forEach(i => i.checked = true);
            document.querySelectorAll('.node').forEach(g => g.style.display = '');
            document.getElementById('node-details').innerHTML = '<div class="text-zinc-400 text-xs">Click a node to see details and neighbors.</div>';
            draw();
        }

        function init() {
            draw();
            document.getElementById('search').addEventListener('input', filterNodes);
            document.querySelectorAll('#filters input').forEach(input => { input.addEventListener('change', filterNodes); });
            document.addEventListener('keydown', (e) => {
                if (e.key === '/' && document.activeElement.tagName === 'BODY') { e.preventDefault(); document.getElementById('search').focus(); }
            });
        }
        init();
    </script>
</body>
</html>"""

    html = template.replace("NODES_JSON", nodes_json)
    html = html.replace("EDGES_JSON", edges_json)
    html = html.replace("INDEX_DIR", str(index_dir))
    html = html.replace("STATS_FILES", str(stats["num_files"]))
    html = html.replace("STATS_SYMBOLS", str(stats["num_symbols"]))
    html = html.replace("STATS_EDGES", str(stats["num_edges"]))

    return html


def generate_artifacts(index_dir: Path) -> dict[str, Path]:
    """Main entry point. Generates both artifacts next to graph.json."""
    graph_data = load_graph(index_dir)
    G = build_networkx_graph(graph_data)

    md = generate_markdown_report(graph_data, G, index_dir)
    md_path = index_dir / "GRAPH_REPORT.md"
    md_path.write_text(md, encoding="utf-8")

    html = generate_html(graph_data, G, index_dir)
    html_path = index_dir / "graph.html"
    html_path.write_text(html, encoding="utf-8")

    logger.info("Generated artifacts: %s and %s", md_path.name, html_path.name)
    return {"report": md_path, "viz": html_path}
