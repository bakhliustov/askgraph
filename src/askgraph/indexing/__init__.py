"""Indexing pipeline: discovery, chunking, embedding, storage (local Chroma + graph)."""

from askgraph.indexing.chunking import CodeChunk, chunk_file, chunk_python_file, chunk_text
from askgraph.indexing.indexer import index_codebase
from askgraph.indexing.parsers import parse_python

__all__ = [
    "CodeChunk",
    "chunk_file",
    "chunk_python_file",
    "chunk_text",
    "index_codebase",
    "parse_python",
]
