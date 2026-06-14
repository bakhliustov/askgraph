"""Local retrieval over the Chroma index.

Hybrid retrieval: semantic similarity (embeddings) + structural graph expansion.
The graph provides explicit relationships (contains, imports) that pure vector
search misses. Expansion happens in the synthesizer for now (context enrichment
for the LLM) and can be used for raw retrieval display too.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from fastembed import TextEmbedding

from askgraph.config import get_index_path, settings
from askgraph.query.graph import get_related_entities, load_graph
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)

# Very small stopword list — enough to stop generic words from dominating the
# lexical signal without needing an NLP dependency.
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "is",
        "are",
        "how",
        "does",
        "do",
        "what",
        "where",
        "when",
        "which",
        "and",
        "or",
        "for",
        "on",
        "with",
        "this",
        "that",
        "it",
        "its",
        "as",
        "by",
        "from",
        "work",
        "works",
        "used",
        "use",
        "using",
        "into",
    ]
)
_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text/identifiers into lowercase terms (splits camelCase + snake_case).

    `sendMessageStream` -> [send, message, stream]; `run_full_council` ->
    [run, full, council]. Code retrieval leans on identifier tokens, so this is
    the backbone of the lexical signal.
    """
    out: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", text):
        for w in _TOKEN_RE.findall(part):
            lw = w.lower()
            if len(lw) >= 2 and lw not in _STOPWORDS:
                out.append(lw)
    return out


def lexical_score(query_terms: set[str], hit: dict[str, Any]) -> float:
    """Field-weighted lexical overlap between the query and a hit.

    Symbol-name matches count most (they're the strongest code-retrieval signal),
    then file path, then chunk body.
    """
    if not query_terms:
        return 0.0
    meta = hit.get("metadata", {}) or {}
    sym_tokens = set(tokenize(str(meta.get("symbol") or "")))
    path_tokens = set(tokenize(str(meta.get("file_path") or "")))
    text_tokens = set(tokenize(hit.get("text", "")))
    return (
        4.0 * len(query_terms & sym_tokens)
        + 1.5 * len(query_terms & path_tokens)
        + 1.0 * len(query_terms & text_tokens)
    )


def _minmax(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]; all-equal inputs map to 0.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def fuse_lexical(
    query: str, candidates: list[dict[str, Any]], top_k: int, alpha: float = 0.5
) -> list[dict[str, Any]]:
    """Re-rank vector candidates by blending vector similarity with lexical score.

    ``alpha`` is the lexical weight (0 = pure vector, 1 = pure lexical). Both
    signals are min-max normalized across the candidate pool before blending.
    """
    if not candidates:
        return []
    q_terms = set(tokenize(query))
    if not q_terms:
        return candidates[:top_k]

    def _sim(c: dict[str, Any]) -> float:
        dist = c.get("distance")
        return max(0.0, 1.0 - (float(dist) if dist is not None else 1.0))

    sims = [_sim(c) for c in candidates]
    lex = [lexical_score(q_terms, c) for c in candidates]
    sims_n, lex_n = _minmax(sims), _minmax(lex)

    scored = [(c, (1 - alpha) * sims_n[i] + alpha * lex_n[i]) for i, c in enumerate(candidates)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


def _rows_from_query(results: Any, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Normalize a Chroma query result into a flat list of hit dicts (None-safe)."""
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    rows: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        rows.append(
            {
                "chunk_id": cid,
                "text": docs[i] if i < len(docs) else "",
                "metadata": (metas[i] if i < len(metas) else None) or {},
                "distance": dists[i] if i < len(dists) else None,
                **(extra or {}),
            }
        )
    return rows


class LocalRetriever:
    """Semantic retriever backed by the local Chroma index created by LocalIndexer."""

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir.resolve()
        self.index_dir = get_index_path(self.target_dir)
        chroma_dir = self.index_dir / "chroma"

        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"No index found at {self.index_dir}. Run `askgraph index {target_dir}` first."
            )

        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(name="askgraph_code")

        # Same embedding model used at index time for consistency
        self.embedder = TextEmbedding(model_name=settings.embedding_model)

    def _embed(self, query: str) -> Any:
        """Embed a single query string (kept in one place so it's only computed once)."""
        return next(iter(self.embedder.embed([query])))

    def retrieve(
        self, query: str, top_k: int = 8, query_embedding: Any | None = None
    ) -> list[dict[str, Any]]:
        """Return top matching chunks with text + metadata."""
        q_emb = query_embedding if query_embedding is not None else self._embed(query)

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = _rows_from_query(results)
        logger.info("Retrieved %d chunks for query", len(hits))
        return hits

    def expand_hits(self, hits: list[dict[str, Any]], max_related: int = 3) -> list[dict[str, Any]]:
        """Attach graph neighborhood info to hits for hybrid context.

        Returns the hits with an extra 'graph_relations' key in metadata.
        """
        graph_data = load_graph(self.index_dir)
        if not graph_data:
            return hits

        for hit in hits:
            meta = hit.get("metadata", {})
            fp = meta.get("file_path")
            sym = meta.get("symbol")
            if not fp:
                continue

            related = get_related_entities(graph_data, fp, sym, max_neighbors=max_related)
            if related:
                meta = dict(meta)  # copy
                meta["graph_relations"] = related
                hit["metadata"] = meta

        return hits

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 8,
        expand: bool = True,
        graph_depth: int = 1,
        lexical: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic retrieval + optional lexical fusion + structural graph expansion.

        - Embedding-based candidates (a wider pool when ``lexical`` is on).
        - ``lexical`` re-ranks that pool by blending vector similarity with
          identifier/symbol-name overlap (strong signal for code).
        - ``expand`` then annotates hits with graph neighbors and pulls extra
          chunks from graph-related files for richer context.
        """
        q_emb = self._embed(query)
        if lexical and settings.lexical_alpha > 0:
            pool = max(top_k * 4, 20)
            candidates = self.retrieve(query, top_k=pool, query_embedding=q_emb)
            hits = fuse_lexical(query, candidates, top_k, alpha=settings.lexical_alpha)
        else:
            hits = self.retrieve(query, top_k=top_k, query_embedding=q_emb)
        if not expand:
            return hits

        hits = self.expand_hits(hits)

        # Deeper hybrid: pull extra chunks from graph-related files
        if graph_depth > 0:
            try:
                graph_data = load_graph(self.index_dir)
                if graph_data:
                    related_files = set()
                    for hit in hits[: max(3, top_k // 2)]:  # focus on top hits
                        meta = hit.get("metadata", {})
                        fp = meta.get("file_path")
                        rels = meta.get("graph_relations", [])
                        if fp:
                            related_files.add(fp)
                        for r in rels:
                            rpath = r.get("path")
                            if rpath:
                                related_files.add(rpath)

                    if related_files:
                        # Fetch additional chunks from related files (limit to avoid bloat)
                        where: Any = {"file_path": {"$in": list(related_files)}}
                        extra_results = self.collection.query(
                            query_embeddings=[q_emb],
                            n_results=min(8, len(related_files) * 2),
                            where=where,
                            include=["documents", "metadatas", "distances"],
                        )
                        seen_ids = {h["chunk_id"] for h in hits}
                        for row in _rows_from_query(extra_results, extra={"from_graph": True}):
                            if row["chunk_id"] not in seen_ids:
                                hits.append(row)
                                seen_ids.add(row["chunk_id"])
            except Exception as e:
                logger.debug("Graph chunk expansion failed: %s", e)

        return hits
