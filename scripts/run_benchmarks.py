#!/usr/bin/env python
"""Benchmark runner for askgraph.

Generates clean, publishable results using the internal evaluate API.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from askgraph.eval import evaluate, load_cases
from askgraph.query import LocalRetriever

EVAL_CASES = [
    ("tinygrad-nn", "evals/tinygrad-nn.yaml"),
    ("autoresearch", "evals/autoresearch.yaml"),
    ("llm-council", "evals/llm-council.yaml"),
]

def run_benchmark(repo_name: str, cases_path: str, top_k: int = 8) -> dict:
    try:
        cases = load_cases(Path(cases_path))
        retriever = LocalRetriever(Path("."))

        # Run with different configurations to show value of each signal
        results = {}
        for label, kwargs in [
            ("Vector only", {"expand": False, "lexical": False}),
            ("+ Lexical", {"expand": False, "lexical": True}),
            ("+ Lexical + Graph", {"expand": True, "lexical": True}),
        ]:
            res = evaluate(retriever, cases, top_k=top_k, **kwargs)
            results[label] = res["metrics"]

        return {"repo": repo_name, "results": results}
    except Exception as e:
        return {"error": str(e), "repo": repo_name}

def main():
    print("# askgraph Benchmarks (real runs on public repos)\n")
    print("Produced by the built-in eval harness (see scripts/run_benchmarks.py).\n")

    for name, cases in EVAL_CASES:
        print(f"## {name}")
        result = run_benchmark(name, cases)

        if "error" in result:
            print(f"Error: {result['error']}\n")
            continue

        for label, metrics in result["results"].items():
            print(f"### {label}")
            print(f"- Recall@{metrics.get('top_k', 8)}: {metrics.get('recall', 'N/A')}")
            print(f"- MRR: {metrics.get('mrr', 'N/A')}")
            print(f"- Hit Rate: {metrics.get('hit_rate', 'N/A')}")
        print()

if __name__ == "__main__":
    main()
