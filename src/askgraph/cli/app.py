"""Main Typer CLI for askgraph — local-first codebase QA."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel

from askgraph import __version__
from askgraph.config import get_index_path, settings
from askgraph.indexing.indexer import index_codebase
from askgraph.query import LocalRetriever, ollama_available, synthesize_answer
from askgraph.report import generate_artifacts

# Lazy imports for optional components
AskGraphTUI = None
try:
    from askgraph.tui import AskGraphTUI as _AskGraphTUI

    AskGraphTUI = _AskGraphTUI
except ImportError:
    pass

MCP_AVAILABLE = False
try:
    from askgraph.mcp_server import run_mcp_server

    MCP_AVAILABLE = True
except ImportError:
    pass  # mcp extra not installed

app = typer.Typer(
    name="askgraph",
    help="Local-first, privacy-first codebase QA. Build structural + semantic knowledge graphs and ask questions about your code.",
    add_completion=True,
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", "-v", help="Show version and exit.")
    ] = False,
) -> None:
    if version:
        console.print(f"askgraph {__version__}")
        raise typer.Exit()


@app.command()
def index(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to the directory or repository to index.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Rebuild index from scratch even if incremental is possible."
        ),
    ] = False,
    show_progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
    no_report: Annotated[
        bool,
        typer.Option(
            "--no-report", help="Skip automatic generation of GRAPH_REPORT.md and graph.html"
        ),
    ] = False,
) -> None:
    """Index a codebase (builds structural graph + semantic index).

    Creates a local .askgraph/ directory with everything needed for fast private queries.
    Nothing is sent anywhere. Uses fastembed (local) + Chroma (local persistent).
    By default also generates the report artifacts (GRAPH_REPORT.md + graph.html).
    """
    target = path.resolve()
    index_path = get_index_path(target)

    console.print(
        Panel.fit(
            f"[bold]Indexing[/bold] {target}\n"
            f"Index location: [cyan]{index_path}[/cyan]\n"
            f"Local-only mode: [green]{settings.local_only}[/green]",
            title="askgraph index",
            border_style="blue",
        )
    )

    stats = index_codebase(target, force=force, show_progress=show_progress)

    if stats.get("status") == "up-to-date":
        console.print("[green]Index already up to date.[/green]")
    else:
        console.print(
            f"[green]✓[/green] Indexed [bold]{stats.get('files_indexed', 0)}[/bold] files, "
            f"added [bold]{stats.get('chunks_added', 0)}[/bold] chunks."
        )
        if stats.get("graph_nodes"):
            console.print(
                f"   Structural graph: [bold]{stats.get('graph_nodes')}[/bold] nodes, "
                f"[bold]{stats.get('graph_edges')}[/bold] edges  →  [cyan]graph.json[/cyan]"
            )

    console.print(f"Index lives at: [cyan]{index_path}[/cyan]")

    # Auto-generate report artifacts unless disabled
    if not no_report and (index_path / "graph.json").exists():
        try:
            with console.status("[bold green]Generating GRAPH_REPORT.md and graph.html..."):
                artifacts = generate_artifacts(index_path)
            console.print(
                f"[green]✓[/green] Report artifacts generated: "
                f"[cyan]{artifacts['report'].name}[/cyan] + [cyan]{artifacts['viz'].name}[/cyan]"
            )
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to auto-generate report: {e}")
            console.print("You can run `askgraph report .` manually.")

    console.print('You can now run [bold]askgraph ask "your question here"[/bold]')


def _print_hits(hits: list[dict]) -> None:
    """Pretty-print retrieved chunks with hybrid graph relations if present."""
    console.print(f"\n[bold]Found {len(hits)} relevant chunks:[/bold]\n")
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        fp = meta.get("file_path", "?")
        sym = meta.get("symbol")
        sym_type = meta.get("symbol_type")

        header = f"[cyan]{fp}"
        if sym:
            label = f"{sym_type} " if sym_type else ""
            header += f"  →  [bold]{label}{sym}[/bold]"
        if meta.get("start_line"):
            header += f" (L{meta['start_line']}-{meta.get('end_line', '?')})"
        header += "[/cyan]"

        strategy = meta.get("strategy", "")
        if strategy:
            header += f" [dim]({strategy})[/dim]"

        console.print(f"{i}. {header}")
        snippet = hit["text"][:420].replace("\n", " ")
        console.print(f"   [dim]{snippet}{'...' if len(hit['text']) > 420 else ''}[/dim]")

        # Show graph relations for hybrid flavor
        rels = meta.get("graph_relations", [])
        if rels:
            rel_names = [r.get("name", r.get("id", "")) for r in rels[:3]]
            console.print(f"      [dim]graph neighbors: {', '.join(rel_names)}[/dim]")

        console.print()  # extra line


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question about the codebase.")],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Root of the indexed codebase (defaults to current dir).",
            exists=True,
            file_okay=False,
        ),
    ] = Path("."),
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override default Ollama model for this query"),
    ] = None,
    raw: Annotated[
        bool, typer.Option("--raw", help="Show only retrieved chunks (skip LLM synthesis)")
    ] = False,
    show_context: Annotated[bool, typer.Option("--context/--no-context")] = True,
) -> None:
    """Ask a natural language question about the indexed codebase.

    Hybrid retrieval: embeddings (semantic) + structural graph expansion
    (neighborhoods, contains/imports relations from tree-sitter analysis).
    By default synthesizes a cited answer via local Ollama.
    """
    target = path.resolve()
    index_path = get_index_path(target)

    if not index_path.exists():
        console.print(
            f"[red]No index found at {index_path}[/red]\n"
            f"Run [bold]askgraph index {target}[/bold] first."
        )
        raise typer.Exit(code=1)

    k = top_k or settings.top_k

    header_info = f"[bold]Question:[/bold] {question}\nCodebase: [cyan]{target}[/cyan]\nHybrid retrieval (embeddings + graph)"
    if model:
        header_info += f"\nModel override: [magenta]{model}[/magenta]"

    console.print(Panel.fit(header_info, title="askgraph ask", border_style="green"))

    try:
        retriever = LocalRetriever(target)
        hits = retriever.retrieve_hybrid(question, top_k=k, expand=True)
    except Exception as exc:
        console.print(f"[red]Retrieval failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Raw mode or Ollama unavailable → just show chunks
    if raw or not ollama_available():
        _print_hits(hits)
        if not raw and not ollama_available():
            console.print(
                "\n[yellow]Ollama not reachable[/yellow] — showing raw retrieval.\n"
                "To get synthesized answers run:\n"
                "  [bold]ollama serve[/bold]\n"
                f"  [bold]ollama pull {settings.default_llm_model}[/bold]"
            )
        if show_context:
            console.print("[dim]Use without --raw once Ollama is available for full answers.[/dim]")
        return

    # Full synthesis path
    try:
        answer = synthesize_answer(question, hits, index_dir=index_path, model=model)

        # Beautiful answer output
        console.print(Panel(answer, title="Answer", border_style="green", padding=(1, 2)))

        if show_context:
            console.print("\n[bold dim]Sources (used for the answer above):[/bold dim]")
            _print_hits(hits)

    except Exception as exc:
        console.print(f"[red]Answer synthesis failed[/red]: {exc}")
        console.print("Falling back to raw retrieval results:\n")
        _print_hits(hits)


@app.command()
def status(
    path: Annotated[Path, typer.Argument(help="Path to show status for")] = Path("."),
) -> None:
    """Show status of the local index for a codebase.

    Includes graph stats when available (nodes, edges, god nodes).
    """
    target = path.resolve()
    index_path = get_index_path(target)

    console.print(f"Target: [cyan]{target}[/cyan]")
    console.print(f"Index dir: [cyan]{index_path}[/cyan]")

    if not index_path.exists():
        console.print("[yellow]No index yet.[/yellow] Run `askgraph index .`")
        return

    console.print("[green]Index exists[/green]")

    # Metadata
    meta_path = index_path / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            files = meta.get("files", {})
            console.print(f"  Files indexed: [bold]{len(files)}[/bold]")
            total_chunks = sum(f.get("chunks", 0) for f in files.values())
            console.print(f"  Total chunks: [bold]{total_chunks}[/bold]")
        except Exception:
            pass

    # Graph stats
    graph_path = index_path / "graph.json"
    if graph_path.exists():
        try:
            gdata = json.loads(graph_path.read_text())
            nodes = gdata.get("nodes", [])
            edges = gdata.get("edges", [])
            stats = gdata.get("stats", {})
            console.print(f"  Graph nodes: [bold]{len(nodes)}[/bold] (files + symbols)")
            console.print(f"  Graph edges: [bold]{len(edges)}[/bold]")
            if stats:
                console.print(
                    f"  Files: {stats.get('num_files', '?')}  Symbols: {stats.get('num_symbols', '?')}"
                )

            # Simple god nodes (top by degree)
            from collections import Counter

            degrees: Counter[str] = Counter()
            for e in edges:
                degrees[e.get("source", "")] += 1
                degrees[e.get("target", "")] += 1
            top = degrees.most_common(5)
            if top:
                console.print("  Top connected (god nodes):")
                for nid, deg in top:
                    # find nice name
                    name = nid
                    for n in nodes:
                        if n.get("id") == nid:
                            name = n.get("name") or n.get("path") or nid
                            break
                    console.print(f"    - {name} (degree {deg})")
        except Exception as e:
            console.print(f"  [yellow]Could not read graph stats: {e}[/yellow]")

    # Report artifacts
    if (index_path / "GRAPH_REPORT.md").exists() and (index_path / "graph.html").exists():
        console.print("  Report artifacts: [green]present[/green] (GRAPH_REPORT.md + graph.html)")
    else:
        console.print("  Report artifacts: [dim]not generated yet[/dim] (run `askgraph report .`)")


@app.command()
def export(
    question: Annotated[str, typer.Argument(help="Question to build context for.")],
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Root of the indexed codebase."),
    ] = Path("."),
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output markdown file."),
    ] = Path("askgraph-context.md"),
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 12,
) -> None:
    """Export a rich context bundle (hybrid retrieval + graph neighbors + provenance) as markdown.

    Optimized for humans and AI agents. For the richest agent pack (JSON + subgraph + suggested prompt),
    use the MCP tool `export_agent_context` or `askgraph mcp`.
    """
    target = path.resolve()
    index_path = get_index_path(target)

    if not index_path.exists():
        console.print(f"[red]No index at {index_path}[/red]. Run index first.")
        raise typer.Exit(1)

    console.print(f"Building hybrid context for: [bold]{question}[/bold]")

    retriever = LocalRetriever(target)
    hits = retriever.retrieve_hybrid(question, top_k=top_k, expand=True)

    graph_data = None
    with contextlib.suppress(Exception):
        graph_data = json.loads((index_path / "graph.json").read_text())

    lines = [
        "# askgraph Context Export\n\n",
        f"**Question:** {question}\n\n",
        f"**Source:** {target}\n\n",
        "## Retrieved Context (Hybrid)\n\n",
    ]

    nodes = {n["id"]: n for n in graph_data.get("nodes", [])} if graph_data else {}

    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        fp = meta.get("file_path", "?")
        sym = meta.get("symbol")
        loc = fp
        if sym:
            loc += f" :: {sym}"
        if meta.get("start_line"):
            loc += f" (L{meta['start_line']}-{meta.get('end_line', '?')})"

        lines.append(f"### {i}. {loc}\n\n")
        lines.append("```python\n")
        lines.append(hit["text"])
        lines.append("\n```\n\n")

        rels = meta.get("graph_relations", [])
        if rels:
            lines.append(
                "**Graph neighbors:** " + ", ".join(r.get("name", "") for r in rels[:5]) + "\n\n"
            )

        # Git provenance if available for the symbol
        if sym and graph_data:
            sym_id = f"symbol:{fp}:{sym}"
            node = nodes.get(sym_id, {})
            git = node.get("git", {})
            if git:
                lines.append(
                    f"**Git provenance:** last touched in {git.get('commit')} "
                    f"by {git.get('author')} on {git.get('date')[:10]} — {git.get('message')}\n\n"
                )

    out_path = out.resolve()
    out_path.write_text("".join(lines), encoding="utf-8")
    console.print(f"[green]Exported to[/green] {out_path}")
    console.print("You can now paste this into Claude, Cursor, Aider, etc.")


@app.command()
def report(
    path: Annotated[Path, typer.Argument(help="Path whose index to report on")] = Path("."),
) -> None:
    """Generate (or regenerate) GRAPH_REPORT.md + self-contained interactive graph.html.

    These are the beautiful, shareable artifacts inspired by Graphify:
    - GRAPH_REPORT.md: god nodes, communities, surprising connections, suggested questions
    - graph.html: fully offline interactive visualization (no external dependencies after load)
    """
    target = path.resolve()
    index_path = get_index_path(target)

    if not (index_path / "graph.json").exists():
        console.print(f"[red]No graph found at {index_path}[/red]")
        console.print("Run [bold]askgraph index {target}[/bold] first.")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"Target: [cyan]{target}[/cyan]\nReading graph from: [cyan]{index_path}[/cyan]",
            title="askgraph report",
            border_style="violet",
        )
    )

    with console.status("[bold]Analyzing graph and building artifacts..."):
        artifacts = generate_artifacts(index_path)

    console.print("[green]✓[/green] Generated:")
    console.print(f"  • [cyan]{artifacts['report'].name}[/cyan]  — high-signal structural summary")
    console.print(
        f"  • [cyan]{artifacts['viz'].name}[/cyan]     — beautiful self-contained interactive viz"
    )

    console.print(
        f"\nOpen [bold]{artifacts['viz'].name}[/bold] in your browser for the interactive graph."
    )
    console.print("Both files live inside the index directory and can be committed if desired.")


@app.command()
def eval(
    path: Annotated[
        Path, typer.Argument(help="Path to evaluate on (public repo recommended)")
    ] = Path("."),
    queries: Annotated[int, typer.Option("--queries", "-q", help="Number of sample queries")] = 5,
) -> None:
    """Lightweight eval harness: run sample queries, report retrieval coverage, provenance hit rate, rough token savings.

    Useful for publishing benchmarks on public repos (ties into your finrag-eval experience).
    """
    target = path.resolve()
    index_path = get_index_path(target)
    if not index_path.exists():
        console.print("No index — running index first...")
        index_codebase(target)

    retriever = LocalRetriever(target)
    sample_queries = [
        "How does the main entrypoint or app initialization work?",
        "What are the core data models or classes and their relationships?",
        "Explain the routing or request handling logic.",
        "How is configuration or settings loaded and used?",
        "What are the key utilities or helper functions and who uses them?",
    ][:queries]

    results: list[dict[str, Any]] = []
    total_hits_with_prov = 0
    total_hits = 0
    for q in sample_queries:
        hits = retriever.retrieve_hybrid(q, top_k=6, expand=True)
        prov_count = sum(1 for h in hits if h.get("metadata", {}).get("symbol"))
        total_hits_with_prov += prov_count
        total_hits += len(hits)
        results.append({"query": q, "hits": len(hits), "with_symbol": prov_count})

    console.print(
        Panel.fit(
            f"Eval on {target}\n"
            f"Queries: {len(sample_queries)}\n"
            f"Avg hits per query: {total_hits / max(1, len(sample_queries)):.1f}\n"
            f"Symbol/provenance coverage: {total_hits_with_prov / max(1, total_hits) * 100:.0f}%",
            title="Light Eval Results",
            border_style="green",
        )
    )
    for r in results:
        console.print(f"- {r['query'][:60]}... → {r['hits']} hits, {r['with_symbol']} with symbols")


@app.command()
def tui(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to the codebase to chat with (defaults to current dir).",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("."),
) -> None:
    """Launch the polished Textual TUI chat.

    Features:
    • Hybrid retrieval + full synthesis with git provenance
    • Clickable god-node sidebar (with commit info)
    • Rich sources panel showing provenance
    • In-TUI commands: /index /status /export /clear /help
    • Perfect for long, contextual explorations

    Install extra: uv pip install -e '.[tui]'
    """
    if AskGraphTUI is None:
        console.print(
            "[red]TUI not available.[/red] Install with:\n"
            "  uv pip install -e '.[tui]'\n"
            "or\n"
            "  pip install 'askgraph[tui]'"
        )
        raise typer.Exit(1)
    target = path.resolve()
    AskGraphTUI(target).run()


@app.command()
def mcp(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to expose via MCP (agents will pass paths in tool calls too).",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("."),
) -> None:
    """Run askgraph as an MCP server.

    AI coding agents (Claude Code, OpenCode, Cursor, Aider, etc.) can then
    natively call index, ask (with provenance), get_god_nodes, and
    export_agent_context.

    Install with: uv pip install -e '.[mcp]'
    """
    if not MCP_AVAILABLE:
        console.print(
            "[red]MCP support not available.[/red]\n"
            "Install with:\n"
            "  uv pip install -e '.[mcp]'\n"
            "or\n"
            "  pip install 'askgraph[mcp]'"
        )
        raise typer.Exit(1)

    target = path.resolve()
    console.print(f"[bold green]Starting askgraph MCP server[/bold green] for {target}")
    console.print(
        "Agents can now discover and use these tools: index_codebase, ask, get_god_nodes, export_agent_context"
    )
    run_mcp_server(target)


if __name__ == "__main__":
    app()
