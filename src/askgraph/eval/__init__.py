"""Ground-truth retrieval evaluation for askgraph.

Measures *retrieval quality* against labeled cases (question -> relevant
files/symbols) so the "structure beats similarity" claim can be quantified —
e.g. recall@k and MRR with graph expansion on vs off.
"""

from askgraph.eval.harness import (
    EvalCase,
    aggregate,
    evaluate,
    hit_matches,
    load_cases,
    score_case,
)

__all__ = [
    "EvalCase",
    "aggregate",
    "evaluate",
    "hit_matches",
    "load_cases",
    "score_case",
]
