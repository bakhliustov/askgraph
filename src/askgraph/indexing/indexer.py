"""Core local indexer.

Builds a semantic index (Chroma + fastembed) + a lightweight structural graph.
Everything stays on disk in the target's .askgraph/ directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from fastembed import TextEmbedding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from askgraph.config import get_index_path, settings
from askgraph.indexing.chunking import CodeChunk, chunk_file
from askgraph.indexing.parsers import parse_file
from askgraph.utils.discovery import discover_files
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)

# Optional git enrichment
try:
    from git import InvalidGitRepositoryError, Repo

    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False
    Repo = None  # type: ignore
    InvalidGitRepositoryError = Exception  # type: ignore


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_metadata(index_dir: Path) -> dict[str, Any]:
    meta_path = index_dir / "metadata.json"
    if meta_path.exists():
        meta: dict[str, Any] = json.loads(meta_path.read_text())
        return meta
    return {"files": {}, "version": "0.1"}


def _save_metadata(index_dir: Path, meta: dict[str, Any]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


class LocalIndexer:
    """Local-only indexer using fastembed + Chroma (persistent, private)."""

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir.resolve()
        self.index_dir = get_index_path(self.target_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_dir = self.index_dir / "chroma"
        self.chroma_dir.mkdir(exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="askgraph_code",
            metadata={"hnsw:space": "cosine"},
        )

        # Local embeddings — fastembed downloads the model to ~/.cache/fastembed and runs on CPU.
        # bge-small is a strong, compact default for code + docs.
        self.embedder = TextEmbedding(model_name=settings.embedding_model)

        self.metadata = _load_metadata(self.index_dir)

        # Load git repo once for provenance (optional)
        self.git_repo = None
        self.git_commit = None
        if _GIT_AVAILABLE:
            try:
                self.git_repo = Repo(self.target_dir, search_parent_directories=True)
                self.git_commit = self.git_repo.head.commit
            except (InvalidGitRepositoryError, Exception):
                pass

        # Per-symbol blame is opt-in (it spawns a git subprocess per file and can be
        # slow on large/deep-history repos). File-level last-commit info is always cheap.
        self.use_git_blame = settings.use_git_blame and self.git_repo is not None

        logger.info("LocalIndexer ready for %s (index at %s)", self.target_dir, self.index_dir)

    def _blame_file(self, rel_path: str) -> dict[int, dict[str, str]]:
        """Blame a whole file once and map each final line number to its commit info.

        Returns {line_no: {commit, author, date, message}}. Best-effort — returns an
        empty map on any failure (binary files, paths outside the repo, etc.).
        """
        if not self.git_repo:
            return {}
        try:
            # Use the absolute path: rel_path is relative to the index target, which
            # may be a subdirectory of the git repo (blame resolves paths against the
            # repo root, so a target-relative path would silently match nothing).
            output = self.git_repo.git.blame(
                "--line-porcelain", "--", str(self.target_dir / rel_path)
            )
        except Exception:
            return {}

        line_info: dict[int, dict[str, str]] = {}
        cur: dict[str, str] = {}
        final_line: int | None = None
        for raw in output.split("\n"):
            if not raw:
                continue
            if raw[0] == "\t":  # the actual source line — flush the accumulated header
                if final_line is not None and cur.get("commit"):
                    line_info[final_line] = {
                        "commit": cur["commit"][:8],
                        "author": cur.get("author", "?"),
                        "date": cur.get("date", ""),
                        "message": cur.get("summary", "")[:100],
                    }
                continue
            parts = raw.split(" ")
            head = parts[0]
            if len(head) == 40 and all(c in "0123456789abcdef" for c in head):
                cur = {"commit": head}
                if len(parts) >= 3:
                    with contextlib.suppress(ValueError):
                        final_line = int(parts[2])
            elif head == "author":
                cur["author"] = raw[len("author ") :]
            elif head == "committer-time":
                with contextlib.suppress(ValueError):
                    cur["date"] = datetime.fromtimestamp(int(parts[1]), tz=UTC).isoformat()
            elif head == "summary":
                cur["summary"] = raw[len("summary ") :]
        return line_info

    @staticmethod
    def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop duplicate edges (same source/target/type), preserving order."""
        seen: set[tuple[str, str, str]] = set()
        out: list[dict[str, Any]] = []
        for e in edges:
            key = (e.get("source", ""), e.get("target", ""), e.get("type", ""))
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    @staticmethod
    def _symbol_history(
        blame_map: dict[int, dict[str, str]], start_line: int, end_line: int
    ) -> list[dict[str, str]]:
        """Unique commits touching a symbol's line range, most recent first (top 5)."""
        seen: dict[str, dict[str, str]] = {}
        for ln in range(start_line, end_line + 1):
            info = blame_map.get(ln)
            if info and info["commit"] not in seen:
                seen[info["commit"]] = info
        commits = sorted(seen.values(), key=lambda c: c.get("date", ""), reverse=True)
        return commits[:5]

    def _process_file(
        self,
        f: Path,
        all_chunks: list[CodeChunk],
        graph_nodes: list[dict[str, Any]],
        graph_edges: list[dict[str, Any]],
    ) -> None:
        """Chunk one file, record its file/symbol nodes, edges, and metadata."""
        rel_path = str(f.relative_to(self.target_dir))
        text = f.read_text(encoding="utf-8", errors="ignore")
        lang = "python" if f.suffix == ".py" else f.suffix.lstrip(".")

        chunks = chunk_file(text, rel_path, language=lang)
        all_chunks.extend(chunks)

        file_node_id = f"file:{rel_path}"
        file_meta: dict[str, Any] = {
            "id": file_node_id,
            "type": "file",
            "path": rel_path,
            "language": lang,
        }
        if self.git_commit is not None:
            file_meta["last_commit"] = str(self.git_commit)[:8]
            file_meta["last_commit_date"] = self.git_commit.committed_datetime.isoformat()
        graph_nodes.append(file_meta)

        # Structural symbols (best-effort).
        try:
            parsed = parse_file(text, rel_path)
            symbols = parsed.get("symbols", [])

            # Blame the file once (opt-in) and reuse it for every symbol in the file.
            blame_map = self._blame_file(rel_path) if (symbols and self.use_git_blame) else {}

            for sym in symbols:
                sym_id = f"symbol:{rel_path}:{sym['name']}"
                sym_node: dict[str, Any] = {
                    "id": sym_id,
                    "type": sym["type"],
                    "name": sym["name"],
                    "path": rel_path,
                    "start_line": sym["start_line"],
                    "end_line": sym["end_line"],
                }
                if blame_map:
                    history = self._symbol_history(blame_map, sym["start_line"], sym["end_line"])
                    if history:
                        sym_node["git_history"] = history
                        sym_node["git"] = history[0]  # most recent = "last touched"

                graph_nodes.append(sym_node)
                graph_edges.append({"source": file_node_id, "target": sym_id, "type": "contains"})
            for imp in parsed.get("imports", []):
                graph_edges.append({"source": file_node_id, "target": imp, "type": "imports"})
        except Exception:
            pass  # structural info is best-effort

        self.metadata.setdefault("files", {})[rel_path] = {
            "hash": _file_hash(f),
            "size": f.stat().st_size,
            "chunks": len(chunks),
        }

    def index(self, force: bool = False, show_progress: bool = True) -> dict[str, Any]:
        """Discover, chunk, embed, and store. Returns stats."""
        files = discover_files(self.target_dir)
        logger.info("Found %d candidate files", len(files))

        new_or_changed = []
        for f in files:
            fh = _file_hash(f)
            prev = self.metadata.get("files", {}).get(str(f.relative_to(self.target_dir)))
            if force or not prev or prev.get("hash") != fh:
                new_or_changed.append(f)

        if not new_or_changed and not force:
            logger.info("No changes detected. Index is up to date.")
            return {"files_indexed": 0, "chunks_added": 0, "status": "up-to-date"}

        # For a clean force we can delete collection contents (simple approach for v0)
        if force:
            with contextlib.suppress(Exception):
                self.client.delete_collection("askgraph_code")
            self.collection = self.client.get_or_create_collection(
                name="askgraph_code",
                metadata={"hnsw:space": "cosine"},
            )
            self.metadata["files"] = {}

        all_chunks: list[CodeChunk] = []
        graph_nodes: list[dict[str, Any]] = []
        graph_edges: list[dict[str, Any]] = []

        progress = _make_progress() if show_progress else None
        if progress is not None:
            progress.start()

        try:
            parse_task = (
                progress.add_task("Parsing & graphing files", total=len(new_or_changed))
                if progress is not None
                else None
            )
            for f in new_or_changed:
                try:
                    self._process_file(f, all_chunks, graph_nodes, graph_edges)
                except Exception as e:
                    logger.warning("Failed to process %s: %s", f, e)
                finally:
                    if progress is not None and parse_task is not None:
                        progress.advance(parse_task)

            if not all_chunks:
                return {"files_indexed": 0, "chunks_added": 0}

            # Embed locally (batched so the heavy CPU phase shows progress).
            texts = [c.text for c in all_chunks]
            embeddings: list = []
            batch_size = 64
            n = len(texts)
            embed_task = (
                progress.add_task("Embedding chunks (local CPU)", total=n)
                if progress is not None
                else None
            )
            for i in range(0, n, batch_size):
                embeddings.extend(list(self.embedder.embed(texts[i : i + batch_size])))
                done = min(i + batch_size, n)
                if progress is not None and embed_task is not None:
                    progress.update(embed_task, completed=done)
                elif done == n or (i // batch_size) % 10 == 0:
                    logger.info("Embedded %d/%d chunks", done, n)

            self.collection.add(
                ids=[c.chunk_id for c in all_chunks],
                documents=texts,
                metadatas=[c.to_metadata() for c in all_chunks],  # type: ignore[arg-type]
                embeddings=embeddings,  # type: ignore[arg-type]
            )
        finally:
            if progress is not None:
                progress.stop()

        _save_metadata(self.index_dir, self.metadata)

        # Deduplicate structural edges (same source/target/type) so the stored graph
        # stays compact and every downstream count (CLI, report, viz) agrees.
        graph_edges = self._dedupe_edges(graph_edges)

        # Persist the lightweight structural graph (nodes + edges). This is the
        # foundation for reports, neighborhood expansion, god-node detection, and
        # agent context.
        if graph_nodes or graph_edges:
            graph_data = {
                "version": "0.1",
                "nodes": graph_nodes,
                "edges": graph_edges,
                "stats": {
                    "num_files": len([n for n in graph_nodes if n.get("type") == "file"]),
                    "num_symbols": len(
                        [n for n in graph_nodes if n.get("type") in ("function", "class")]
                    ),
                },
            }
            (self.index_dir / "graph.json").write_text(json.dumps(graph_data, indent=2))
            logger.info(
                "Wrote structural graph.json (%d nodes, %d edges)",
                len(graph_nodes),
                len(graph_edges),
            )

        logger.info(
            "Indexed %d new/changed files → %d chunks", len(new_or_changed), len(all_chunks)
        )

        return {
            "files_indexed": len(new_or_changed),
            "chunks_added": len(all_chunks),
            "total_files": len(self.metadata.get("files", {})),
            "index_dir": str(self.index_dir),
            "graph_nodes": len(graph_nodes),
            "graph_edges": len(graph_edges),
        }


def index_codebase(target: Path, force: bool = False, show_progress: bool = True) -> dict[str, Any]:
    """Convenience entry point."""
    indexer = LocalIndexer(target)
    return indexer.index(force=force, show_progress=show_progress)
