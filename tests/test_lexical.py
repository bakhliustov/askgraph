"""Tests for lexical tokenization / scoring / fusion (pure, no model)."""

from __future__ import annotations

from askgraph.query.retriever import fuse_lexical, lexical_score, tokenize


def _hit(symbol: str, text: str, distance: float, file_path: str = "x.py") -> dict:
    return {
        "chunk_id": symbol,
        "text": text,
        "metadata": {"file_path": file_path, "symbol": symbol},
        "distance": distance,
    }


def test_tokenize_splits_identifiers_and_drops_stopwords():
    assert tokenize("sendMessageStream") == ["send", "message", "stream"]
    assert tokenize("run_full_council") == ["run", "full", "council"]
    # stopwords like "how/does/the" are removed; identifiers survive
    toks = tokenize("How does the parse_ranking work")
    assert "parse" in toks and "ranking" in toks
    assert "the" not in toks and "how" not in toks


def test_lexical_score_weights_symbol_name_highest():
    q = set(tokenize("stream a chat message"))
    strong = _hit("sendMessageStream", "body", 0.5)  # symbol matches message+stream
    weak = _hit("unrelated", "a chat message appears in this text", 0.5)  # text-only matches
    assert lexical_score(q, strong) > lexical_score(q, weak)


def test_fuse_lexical_promotes_symbol_match_over_closer_vector_hit():
    # `helper` is the closest vector hit, but `sendMessageStream` is a reasonably
    # close vector hit AND a strong symbol-name match — fusion should lift it to #1.
    cands = [
        _hit("helper", "misc utility code", distance=0.10),
        _hit("sendMessageStream", "post to the server", distance=0.15),
        _hit("unrelated", "totally different", distance=0.40),
    ]
    ranked = fuse_lexical("how to send a message stream", cands, top_k=3, alpha=0.5)
    assert ranked[0]["metadata"]["symbol"] == "sendMessageStream"


def test_fuse_lexical_empty_and_no_query_terms():
    assert fuse_lexical("anything", [], top_k=5) == []
    cands = [_hit("a", "x", 0.1), _hit("b", "y", 0.2)]
    # query is all stopwords -> no lexical terms -> falls back to vector order
    assert fuse_lexical("the and of to", cands, top_k=1)[0]["metadata"]["symbol"] == "a"
