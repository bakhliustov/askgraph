"""Local LLM synthesis for answers (Ollama by default).

This turns raw retrieval into useful, cited explanations.
Designed for local-first use: calls only the Ollama server the user controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama

from askgraph.config import settings
from askgraph.query.graph import format_graph_expansion, load_graph
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)


def format_hits_for_context(hits: list[dict[str, Any]], graph_data: dict | None = None) -> str:
    """Format retrieved chunks into a clear, citable context block.
    Includes git provenance (commit/author/date/message) from graph nodes when available.
    """
    nodes = {n["id"]: n for n in graph_data.get("nodes", [])} if graph_data else {}

    parts: list[str] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        file_path = meta.get("file_path", "unknown")
        symbol = meta.get("symbol")
        sym_type = meta.get("symbol_type")
        start = meta.get("start_line")
        end = meta.get("end_line")

        loc = file_path
        if symbol:
            label = f"{sym_type} " if sym_type else ""
            loc += f" :: {label}{symbol}"
        if start:
            loc += f" (L{start}-{end or '?'})"

        strategy = meta.get("strategy", "")
        tag = f" [{strategy}]" if strategy else ""

        text = hit.get("text", "").strip()
        block = f"[{loc}]{tag}\n{text}"

        # Attach git provenance for the symbol if present in the graph
        if symbol and graph_data:
            sym_id = f"symbol:{file_path}:{symbol}"
            node = nodes.get(sym_id, {})
            git_history = node.get("git_history", [])
            if git_history:
                prov = git_history[0]
                block += (
                    f"\n  [git provenance] last touched in {prov.get('commit')} "
                    f"by {prov.get('author')} on {prov.get('date')[:10]} — {prov.get('message')}"
                )
                if len(git_history) > 1:
                    block += f" (+{len(git_history) - 1} prior changes)"
            elif node.get("git"):
                git = node["git"]
                block += (
                    f"\n  [git provenance] last touched in {git.get('commit')} "
                    f"by {git.get('author')} on {git.get('date')[:10]} — {git.get('message')}"
                )

        parts.append(block)

    return "\n\n---\n\n".join(parts)


def load_light_graph_context(index_dir: Path, hits: list[dict[str, Any]]) -> str:
    """Use the structural graph to provide rich neighborhood context for the hits.

    This is the hybrid expansion: semantic hits + explicit code relationships.
    """
    graph_data = load_graph(index_dir)
    if not graph_data:
        return ""

    try:
        expansion = format_graph_expansion(graph_data, hits, max_per_hit=4)
        if expansion:
            return expansion
    except Exception as e:
        logger.debug("Graph expansion failed: %s", e)
    return ""


def synthesize_answer(
    question: str,
    hits: list[dict[str, Any]],
    index_dir: Path | None = None,
    model: str | None = None,
) -> str:
    """Generate a synthesized answer using local Ollama.

    Falls back to raising if Ollama is unreachable so the caller can decide
    what to show the user.
    """
    if not hits:
        return "No relevant context was found in the index for this question."

    graph_data = load_graph(index_dir) if index_dir else None
    context = format_hits_for_context(hits, graph_data)

    graph_context = ""
    if index_dir:
        graph_context = load_light_graph_context(index_dir, hits)

    prompt = f"""You are an expert software engineer who knows this codebase extremely well.

Answer the user's question **using only the provided context**. 

- Cite sources inline using the exact format shown in the context blocks, e.g. [path/to/file.py :: function_name (L42-67)].
- When git provenance is present in a context block (commit, author, date, short message; possibly multiple prior changes), reference it to explain history or "why this code looks the way it does".
- If the answer cannot be determined from the context, clearly say: "I don't have enough information in the indexed context to answer this confidently."
- Prefer explaining architecture, relationships, and intent over just quoting code.
- Be concise but complete. Use bullet points or short paragraphs when helpful.

Question:
{question}

Context:
{context}

{graph_context}

Answer:"""

    effective_model = model or settings.default_llm_model
    try:
        resp = ollama.generate(
            model=effective_model,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
            },
            stream=False,
        )
        answer: str = resp.get("response", "").strip()
        if not answer:
            raise RuntimeError("Empty response from model")
        return answer
    except ollama.ResponseError as e:
        logger.error("Ollama response error: %s", e)
        raise
    except Exception as e:
        # Covers connection refused, model not found, etc.
        logger.warning("Ollama call failed: %s", e)
        raise


def ollama_available() -> bool:
    """Quick health check for the configured Ollama instance."""
    try:
        ollama.list()
        return True
    except Exception:
        return False
