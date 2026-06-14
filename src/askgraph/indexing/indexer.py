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


def _load_graph(index_dir: Path) -> dict[str, Any]:
    graph_path = index_dir / "graph.json"
    if graph_path.exists():
        graph: dict[str, Any] = json.loads(graph_path.read_text())
        graph.setdefault("nodes", [])
        graph.setdefault("edges", [])
        return graph
    return {"version": "0.1", "nodes": [], "edges": []}


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
    def _resolve_call_edges(
        graph_nodes: list[dict[str, Any]], pending_calls: list[tuple[str, list[str]]]
    ) -> list[dict[str, Any]]:
        """Turn collected call names into `calls` edges between repo symbols.

        Resolution is by symbol name and intentionally conservative: a call edge
        is added only when the callee name maps to exactly one symbol in the repo
        (avoids fan-out noise from common names like ``__init__``/``forward``).
        """
        name_to_ids: dict[str, list[str]] = {}
        for n in graph_nodes:
            if n.get("type") in ("function", "class"):
                name_to_ids.setdefault(n["name"], []).append(n["id"])

        edges: list[dict[str, Any]] = []
        for caller_id, names in pending_calls:
            for nm in names:
                ids = name_to_ids.get(nm, [])
                if len(ids) == 1 and ids[0] != caller_id:
                    edges.append({"source": caller_id, "target": ids[0], "type": "calls"})
        return edges

    @classmethod
    def _merge_graph(
        cls,
        old_graph: dict[str, Any],
        affected_files: set[str],
        new_nodes: list[dict[str, Any]],
        new_edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Merge freshly-parsed graph fragments into the existing graph.

        Incremental re-index only re-parses ``affected_files`` (changed + deleted).
        We drop every old node/edge belonging to those files, then splice in the
        fresh ones. "Belonging" is keyed on a node's ``path`` (both file and symbol
        nodes carry it) and, for edges, on the file of their *source* node — which
        is how each edge was originally emitted (contains/imports from a file node,
        calls from a caller symbol).

        Finally we prune structural edges whose endpoints no longer exist (e.g. a
        call into a now-deleted file) so the merged graph stays internally
        consistent. ``imports`` edges point at free-text module names rather than
        node ids, so they are exempt from the target check.
        """
        old_nodes = old_graph.get("nodes", [])
        old_edges = old_graph.get("edges", [])
        old_id_to_path = {n["id"]: n.get("path") for n in old_nodes}

        kept_nodes = [n for n in old_nodes if n.get("path") not in affected_files]
        kept_edges = [
            e for e in old_edges if old_id_to_path.get(e.get("source")) not in affected_files
        ]

        merged_nodes = kept_nodes + new_nodes
        merged_edges = kept_edges + new_edges

        node_ids = {n["id"] for n in merged_nodes}
        pruned_edges: list[dict[str, Any]] = []
        for e in merged_edges:
            if e.get("source") not in node_ids:
                continue
            if e.get("type") != "imports" and e.get("target") not in node_ids:
                continue
            pruned_edges.append(e)

        return merged_nodes, cls._dedupe_edges(pruned_edges)

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
        pending_calls: list[tuple[str, list[str]]],
    ) -> None:
        """Chunk one file, record its file/symbol nodes, edges, and metadata.

        Records (caller_symbol_id, [called_names]) into ``pending_calls`` for
        repo-wide call-edge resolution after all files are seen.
        """
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
                if sym.get("calls"):
                    pending_calls.append((sym_id, sym["calls"]))
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
        """Discover, chunk, embed, and store. Returns stats.

        Non-force runs are *incremental*: only changed/new files are re-parsed and
        re-embedded, deleted files are pruned, and the persisted graph + Chroma
        collection are updated in place rather than rebuilt from scratch.
        """
        files = discover_files(self.target_dir)
        logger.info("Found %d candidate files", len(files))

        rel_on_disk = {str(f.relative_to(self.target_dir)) for f in files}
        prev_files = set(self.metadata.get("files", {}).keys())

        new_or_changed = []
        for f in files:
            fh = _file_hash(f)
            prev = self.metadata.get("files", {}).get(str(f.relative_to(self.target_dir)))
            if force or not prev or prev.get("hash") != fh:
                new_or_changed.append(f)

        # Files we indexed before that have since been removed from disk.
        deleted_files = sorted(prev_files - rel_on_disk) if not force else []

        if not new_or_changed and not deleted_files and not force:
            logger.info("No changes detected. Index is up to date.")
            return {"files_indexed": 0, "chunks_added": 0, "status": "up-to-date"}

        changed_rel = {str(f.relative_to(self.target_dir)) for f in new_or_changed}
        affected_files = changed_rel | set(deleted_files)

        # For a clean force we can delete collection contents (simple approach for v0).
        # Otherwise start from the existing graph and surgically replace affected files.
        if force:
            with contextlib.suppress(Exception):
                self.client.delete_collection("askgraph_code")
            self.collection = self.client.get_or_create_collection(
                name="askgraph_code",
                metadata={"hnsw:space": "cosine"},
            )
            self.metadata["files"] = {}
            old_graph: dict[str, Any] = {"version": "0.1", "nodes": [], "edges": []}
        else:
            old_graph = _load_graph(self.index_dir)
            # Drop stale vectors: chunk ids are deterministic, so changed content
            # would otherwise keep its old embedding and removed symbols would
            # linger. Deleting every chunk of an affected file before re-adding the
            # current ones keeps Chroma exact and counts stable across re-runs.
            for rel in affected_files:
                with contextlib.suppress(Exception):
                    self.collection.delete(where={"file_path": rel})
            for rel in deleted_files:
                self.metadata.get("files", {}).pop(rel, None)

        all_chunks: list[CodeChunk] = []
        graph_nodes: list[dict[str, Any]] = []
        graph_edges: list[dict[str, Any]] = []
        pending_calls: list[tuple[str, list[str]]] = []

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
                    self._process_file(f, all_chunks, graph_nodes, graph_edges, pending_calls)
                except Exception as e:
                    logger.warning("Failed to process %s: %s", f, e)
                finally:
                    if progress is not None and parse_task is not None:
                        progress.advance(parse_task)

            graph_edges.extend(self._resolve_call_edges(graph_nodes, pending_calls))

            # Embed locally (batched so the heavy CPU phase shows progress).
            if all_chunks:
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

        # Merge the freshly-parsed fragments into the existing graph (removing the
        # affected files' old nodes/edges first), then dedupe. This is the
        # foundation for reports, neighborhood expansion, god-node detection, and
        # agent context — and keeps it correct across incremental re-indexes.
        merged_nodes, merged_edges = self._merge_graph(
            old_graph, affected_files, graph_nodes, graph_edges
        )
        graph_data = {
            "version": "0.1",
            "nodes": merged_nodes,
            "edges": merged_edges,
            "stats": {
                "num_files": len([n for n in merged_nodes if n.get("type") == "file"]),
                "num_symbols": len(
                    [n for n in merged_nodes if n.get("type") in ("function", "class")]
                ),
            },
        }
        (self.index_dir / "graph.json").write_text(json.dumps(graph_data, indent=2))
        logger.info(
            "Wrote structural graph.json (%d nodes, %d edges)",
            len(merged_nodes),
            len(merged_edges),
        )

        logger.info(
            "Indexed %d new/changed files (%d deleted) → %d chunks",
            len(new_or_changed),
            len(deleted_files),
            len(all_chunks),
        )

        return {
            "files_indexed": len(new_or_changed),
            "files_deleted": len(deleted_files),
            "chunks_added": len(all_chunks),
            "total_files": len(self.metadata.get("files", {})),
            "index_dir": str(self.index_dir),
            "graph_nodes": len(merged_nodes),
            "graph_edges": len(merged_edges),
        }


def index_codebase(target: Path, force: bool = False, show_progress: bool = True) -> dict[str, Any]:
    """Convenience entry point."""
    indexer = LocalIndexer(target)
    return indexer.index(force=force, show_progress=show_progress)
