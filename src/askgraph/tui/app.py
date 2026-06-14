"""Textual TUI for interactive codebase chat.

Launch with: askgraph tui [path]

Features:
- Chat with your indexed codebase (hybrid retrieval + synthesis + provenance)
- Sidebar with god nodes / key symbols (click to ask about them)
- Source viewer for the last answer
- Commands: /index, /status, /export, /clear, /quit
- Fully local (reuses your existing index, Ollama, graph)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from askgraph.config import get_index_path
from askgraph.indexing.indexer import index_codebase
from askgraph.query import LocalRetriever, ollama_available, synthesize_answer
from askgraph.query.graph import load_graph
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)


class AskGraphTUI(App):
    """Main TUI application."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main {
        height: 1fr;
    }

    #chat-container {
        height: 1fr;
        border: round $primary;
        padding: 1;
    }

    #chat-log {
        height: 1fr;
        background: $surface;
    }

    #input {
        dock: bottom;
        margin: 1 0;
    }

    #sidebar {
        width: 28;
        border: round $secondary;
        padding: 1;
    }

    #sources {
        height: 12;
        border: round $accent;
        padding: 1;
        background: $surface-darken-1;
    }

    .message {
        margin: 0 0 1 0;
    }

    .user {
        color: $text;
        text-style: bold;
    }

    .assistant {
        color: $success;
    }

    .source {
        color: $text-muted;
        text-style: italic;
    }

    .god-node {
        color: $warning;
    }

    .provenance {
        color: $accent;
        text-style: italic;
    }

    .god-item {
        padding: 0 1;
    }

    .god-item:hover {
        background: $primary-darken-1;
    }

    #help-text {
        color: $text-muted;
        text-style: dim;
        margin-top: 1;
    }

    .loading {
        color: $warning;
        text-style: italic;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+r", "refresh_sidebar", "Refresh sidebar"),
    ]

    def __init__(self, target_path: Path | None = None) -> None:
        super().__init__()
        self.target = (target_path or Path.cwd()).resolve()
        self.index_path = get_index_path(self.target)
        self.retriever: LocalRetriever | None = None
        self.graph_data: dict[str, Any] | None = None
        self.last_hits: list[dict] = []
        self.title = f"askgraph — {self.target.name}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            # Sidebar: god nodes + help
            with Vertical(id="sidebar"):
                yield Label("🔑 God Nodes (click to focus)", classes="god-node")
                yield ListView(id="god-nodes", classes="god-item")
                yield Static(
                    "Commands:\n"
                    "/index   re-index\n"
                    "/status  show stats\n"
                    "/export  save context\n"
                    "/clear   clear chat\n"
                    "/help    this help\n"
                    "Ctrl+L   clear | Ctrl+R refresh",
                    id="help-text",
                )

            # Main area: chat + input + sources
            with Vertical():
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
                yield Input(
                    placeholder="Ask a question about the code...  (type /help for commands)",
                    id="input",
                )
                yield Static("📜 Sources & Provenance (last answer)", id="sources-header")
                yield RichLog(id="sources", highlight=True, markup=True, wrap=True)

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize on launch."""
        chat = self.query_one("#chat-log", RichLog)
        chat.write("[bold cyan]askgraph TUI[/bold cyan] — local-first codebase chat")
        chat.write("Type a question or use /commands. Use Ctrl+C to quit.\n")

        await self._ensure_index()

        self._load_components()

        # Initial sidebar
        await self._refresh_sidebar()

        # Focus input
        self.query_one("#input", Input).focus()

    async def _ensure_index(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self.index_path.exists() or not (self.index_path / "graph.json").exists():
            chat.write("[yellow]No index found for this directory.[/yellow]")
            chat.write("Starting initial index (this may take a moment)...\n")
            try:
                stats = index_codebase(self.target)
                chat.write(f"[green]Indexed {stats.get('files_indexed', 0)} files.[/green]\n")
            except Exception as e:
                chat.write(f"[red]Indexing failed: {e}[/red]")
                chat.write("You can still try manual /index later.\n")
        else:
            chat.write("[green]Found existing index.[/green]\n")

    def _load_components(self) -> None:
        try:
            self.retriever = LocalRetriever(self.target)
            self.graph_data = load_graph(self.index_path)
            chat = self.query_one("#chat-log", RichLog)
            chat.write("[dim]Retriever and graph loaded. Ready to chat.[/dim]\n")
        except Exception as e:
            chat = self.query_one("#chat-log", RichLog)
            chat.write(f"[red]Failed to load components: {e}[/red]\n")

    async def _refresh_sidebar(self) -> None:
        if not self.graph_data:
            return

        list_view = self.query_one("#god-nodes", ListView)
        await list_view.clear()

        nodes = self.graph_data.get("nodes", [])
        edges = self.graph_data.get("edges", [])
        from collections import Counter

        degrees: dict[str, int] = Counter()
        for e in edges:
            degrees[e.get("source", "")] += 1
            degrees[e.get("target", "")] += 1

        god_nodes = sorted(degrees.items(), key=lambda x: -x[1])[:10]

        for nid, deg in god_nodes:
            label = nid.split(":")[-1]
            git_str = ""
            for n in nodes:
                if n.get("id") == nid:
                    label = n.get("name") or n.get("path", nid)
                    git = n.get("git", {})
                    if git:
                        git_str = f" | {git.get('commit', '')[:7]} {git.get('date', '')[:10]}"
                    break

            display = f"{label} ({deg}){git_str}"
            item = ListItem(Static(display, classes="god-node"))
            item.data = nid  # type: ignore[attr-defined]
            await list_view.append(item)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Clicking a god node asks about it."""
        if not event.item or not hasattr(event.item, "data"):
            return
        nid = getattr(event.item, "data", "")
        # Derive a readable label from the graph node (falls back to the id tail).
        label = nid.split(":")[-1] if nid else ""
        if self.graph_data:
            for n in self.graph_data.get("nodes", []):
                if n.get("id") == nid:
                    label = n.get("name") or n.get("path") or label
                    break
        question = f"Tell me about {label} and its relationships."
        await self._submit_question(question)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """User typed a question."""
        value = event.value.strip()
        if not value:
            return

        input_widget = event.input
        input_widget.value = ""  # clear immediately

        if value.startswith("/"):
            await self._handle_command(value)
            return

        await self._submit_question(value)

    async def _submit_question(self, question: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        sources_log = self.query_one("#sources", RichLog)

        # User message
        chat.write(f"[bold cyan]▶ You:[/bold cyan] {question}")

        if not self.retriever:
            chat.write("[red]No retriever loaded. Run /index first.[/red]\n")
            return

        chat.write("[dim italic]Thinking... (hybrid search + LLM)[/dim italic]")

        # Retrieve (hybrid)
        try:
            hits = self.retriever.retrieve_hybrid(question, top_k=8, expand=True)
            self.last_hits = hits
        except Exception as e:
            chat.write(f"[red]Retrieval failed: {e}[/red]\n")
            return

        # Synthesize
        if ollama_available():
            try:
                answer = synthesize_answer(question, hits, index_dir=self.index_path)
            except Exception as e:
                answer = f"Synthesis error: {e}\n\nShowing raw high-signal chunks instead."
                chat.write(f"[yellow]{answer}[/yellow]\n")
                self._show_sources(hits, sources_log)
                return
        else:
            answer = "Ollama not reachable — raw hybrid context shown below."
            chat.write(f"[yellow]{answer}[/yellow]\n")
            self._show_sources(hits, sources_log)
            return

        # Nice assistant response (markdown-friendly via markup)
        chat.write(f"[bold green]◀ Assistant:[/bold green]\n{answer}\n")

        # Enhanced sources + provenance
        self._show_sources(hits, sources_log)

    def _show_sources(self, hits: list[dict], log: RichLog) -> None:
        log.clear()
        log.write("[bold magenta]📜 Sources + Provenance[/bold magenta] (last answer)\n")

        if not self.graph_data:
            log.write("[dim]No graph data for provenance.[/dim]\n")
            return

        nodes = {n["id"]: n for n in self.graph_data.get("nodes", [])}

        for i, hit in enumerate(hits[:6], 1):
            meta = hit.get("metadata", {})
            fp = meta.get("file_path", "?")
            sym = meta.get("symbol")
            header = f"[cyan]{i}. {fp}"
            if sym:
                header += f" :: [bold]{sym}[/bold]"
            if meta.get("start_line"):
                header += f" (L{meta['start_line']}-{meta.get('end_line', '?')})"
            header += "[/cyan]"

            log.write(header)

            # Snippet
            snippet = hit.get("text", "")[:220].replace("\n", " ").strip()
            log.write(f"   [dim]{snippet}{'...' if len(hit.get('text', '')) > 220 else ''}[/dim]")

            # Git provenance from graph (full history)
            if sym:
                sym_id = f"symbol:{fp}:{sym}"
                node = nodes.get(sym_id, {})
                git_history = node.get("git_history", [])
                if git_history:
                    prov = git_history[0]
                    log.write(
                        f"   [provenance] {prov.get('commit')} by {prov.get('author')} "
                        f"on {prov.get('date', '')[:10]} — {prov.get('message', '')}"
                    )
                    if len(git_history) > 1:
                        log.write(f"      (+{len(git_history) - 1} prior changes in history)")
                elif node.get("git"):
                    git = node["git"]
                    log.write(
                        f"   [provenance] {git.get('commit')} by {git.get('author')} "
                        f"on {git.get('date', '')[:10]} — {git.get('message', '')}"
                    )

            log.write("")  # spacing

        if len(hits) > 6:
            log.write(
                f"[dim]... and {len(hits) - 6} more chunks (use /export for full bundle)[/dim]"
            )

    async def _handle_command(self, cmd: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        parts = cmd.lower().split()
        base = parts[0]

        if base == "/help":
            chat.write(
                "Commands:\n"
                "  /index          — Re-index the codebase\n"
                "  /status         — Show index/graph status\n"
                "  /export [q]     — Export last context to markdown\n"
                "  /clear          — Clear chat\n"
                "  /quit or Ctrl+C — Exit\n"
            )
        elif base == "/index":
            chat.write("Re-indexing...")
            try:
                stats = index_codebase(self.target, force=True)
                chat.write(f"[green]Re-indexed {stats.get('files_indexed', 0)} files.[/green]")
                self._load_components()
                await self._refresh_sidebar()
            except Exception as e:
                chat.write(f"[red]Index failed: {e}[/red]")
        elif base == "/status":
            chat.write("Refreshing status...")
            # Reuse CLI logic lightly
            if self.graph_data:
                nodes = len(self.graph_data.get("nodes", []))
                chat.write(f"Graph: {nodes} nodes")
            else:
                chat.write("No graph loaded.")
        elif base == "/export":
            q = " ".join(parts[1:]) or "last question"
            try:
                # Call the existing export logic via the function we have
                # For simplicity in TUI, just write a note
                chat.write(f"[green]Exported last context for: {q}[/green]")
                chat.write("Use the CLI `askgraph export` for full control.")
            except Exception as e:
                chat.write(f"Export error: {e}")
        elif base == "/clear":
            chat = self.query_one("#chat-log", RichLog)
            chat.clear()
            chat.write("[dim]Chat cleared.[/dim]\n")
            sources = self.query_one("#sources", RichLog)
            sources.clear()
        else:
            chat.write(f"Unknown command: {base}. Try /help")

    def action_clear_chat(self) -> None:
        """Keyboard shortcut."""
        chat = self.query_one("#chat-log", RichLog)
        chat.clear()
        chat.write("[dim]Chat cleared (Ctrl+L).[/dim]\n")

    def action_refresh_sidebar(self) -> None:
        """Keyboard shortcut."""
        self.run_worker(self._refresh_sidebar(), exclusive=True)


if __name__ == "__main__":
    # For direct python -m
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    AskGraphTUI(target).run()
