#!/usr/bin/env python
"""
Self-demo: Index the askgraph repo and ask a question.

Run with:
    uv run python examples/self_demo.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from askgraph.indexing.indexer import LocalIndexer
from askgraph.query import LocalRetriever, synthesize_answer

def main():
    root = Path(__file__).parent.parent.resolve()
    print(f"Demo: Indexing {root}...")

    indexer = LocalIndexer(root)
    indexer.index_codebase(force=False, show_progress=True, no_report=True)

    print("\nDemo: Asking a question about the codebase...")
    retriever = LocalRetriever(root)
    question = "How does the structural graph help retrieval?"

    hits = retriever.retrieve_hybrid(question, top_k=5, expand=True, lexical=True)
    answer = synthesize_answer(question, hits, retriever.index_dir)

    print("\n" + "="*60)
    print("ANSWER:")
    print(answer)
    print("="*60)

if __name__ == "__main__":
    main()
