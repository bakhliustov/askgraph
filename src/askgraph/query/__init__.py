"""Query / retrieval + answer generation (local-first).

Hybrid = semantic (Chroma + fastembed) + structural (graph.json neighborhoods).
"""

from askgraph.query.graph import (
    format_graph_expansion,
    get_related_entities,
    load_graph,
)
from askgraph.query.retriever import LocalRetriever
from askgraph.query.synthesizer import (
    ollama_available,
    synthesize_answer,
)

__all__ = [
    "LocalRetriever",
    "format_graph_expansion",
    "get_related_entities",
    "load_graph",
    "ollama_available",
    "synthesize_answer",
]
