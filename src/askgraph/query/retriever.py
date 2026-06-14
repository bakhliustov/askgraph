"""Local retrieval over the Chroma index.

Hybrid retrieval: semantic similarity (embeddings) + structural graph expansion.
The graph provides explicit relationships (contains, imports) that pure vector
search misses. Expansion happens in the synthesizer for now (context enrichment
for the LLM) and can be used for raw retrieval display too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from fastembed import TextEmbedding

from askgraph.config import get_index_path, settings
from askgraph.query.graph import get_related_entities, load_graph
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)


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
        self, query: str, top_k: int = 8, expand: bool = True, graph_depth: int = 1
    ) -> list[dict[str, Any]]:
        """Semantic retrieval + structural graph expansion.

        - First does embedding-based top_k.
        - Then expands with graph neighbors (related files/symbols).
        - Fetches additional chunks from related files using metadata filter
          to give the synthesizer (and user) richer structural context.
        """
        q_emb = self._embed(query)
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
