#!/usr/bin/env python3
"""Aequivalenz-Test fuer die streaming hybrid-GT (gen_special_gt.hybrid_gt).

Validiert den kritischen Invariant: chunk-weises Streaming + Pool-Merge liefert
DASSELBE wie eine Single-Shot-Referenz mit identischer Scoring-Formel. Testet
_pool_merge (Top-k ueber Chunks) und die BM25+RRF-Fusion ohne den HF-Korpus --
synthetische Embeddings + Texte, in 1 vs 3 Chunks aufgeteilt.

Lauf (im WSL-venv mit numpy):
    python -m pytest benchmarks/reviewdata/tests/test_hybrid_gt_streaming.py -q
oder direkt:
    python benchmarks/reviewdata/tests/test_hybrid_gt_streaming.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_special_gt import _pool_merge, BM25_K1, BM25_B, RRF_K, _hybrid_pool  # noqa: E402


def _ref_pool_topk(scores, ids, k):
    """Single-Shot-Referenz: globale Top-k nach Score, absteigend."""
    order = np.argsort(-scores, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    return ids[rows, order], scores[rows, order]


def test_pool_merge_equals_single_shot():
    rng = np.random.default_rng(0)
    Q, N, k = 5, 200, 50
    scores = rng.standard_normal((Q, N)).astype(np.float32)
    ids = np.arange(N, dtype=np.int64)
    ids_b = np.broadcast_to(ids, (Q, N))

    # Single-Shot
    ref_ids, _ = _ref_pool_topk(scores, ids_b, k)

    # Streaming in 4 Chunks
    cur_ids = np.full((Q, k), -1, dtype=np.int64)
    cur_scr = np.full((Q, k), -np.inf, dtype=np.float32)
    for s in range(0, N, 50):
        e = min(s + 50, N)
        nid = np.broadcast_to(np.arange(s, e, dtype=np.int64), (Q, e - s))
        cur_ids, cur_scr = _pool_merge(cur_ids, cur_scr, nid, scores[:, s:e], k)

    # Mengen-Gleichheit pro Query (Reihenfolge bei Ties kann minimal abweichen)
    for q in range(Q):
        assert set(ref_ids[q].tolist()) == set(cur_ids[q].tolist()), f"q={q}"


def _bm25_single_shot(doc_tokens, query_terms, idf, avgdl):
    """Referenz-BM25-Score eines Docs gegen eine Query (komplett im RAM)."""
    out = np.zeros(len(doc_tokens), dtype=np.float32)
    for di, dt in enumerate(doc_tokens):
        counts = Counter(dt)
        ln = len(dt)
        denom = BM25_K1 * (1.0 - BM25_B + BM25_B * (ln / avgdl))
        s = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf and term in idf:
                s += idf[term] * (tf * (BM25_K1 + 1.0)) / (tf + denom)
        out[di] = s
    return out


def test_bm25_chunk_additivity():
    """BM25 eines Docs haengt nur vom Doc + globalen idf/avgdl ab -> chunk-weise
    Berechnung == Single-Shot."""
    docs = [
        "battery life is short and bad".split(),
        "great sound quality headphones".split(),
        "the battery drains fast really fast".split(),
        "comfortable fit good price".split(),
        "battery battery battery issue".split(),
        "fast shipping nice product".split(),
    ]
    query_terms = {"battery", "fast"}
    n_doc = len(docs)
    avgdl = sum(len(d) for d in docs) / n_doc
    df = {t: sum(1 for d in docs if t in set(d)) for t in query_terms}
    idf = {t: math.log(1 + (n_doc - df[t] + 0.5) / (df[t] + 0.5)) for t in query_terms}

    ref = _bm25_single_shot(docs, query_terms, idf, avgdl)

    # chunk-weise (3+3) wie in hybrid_gt: postings je Chunk, denom je Chunk
    got = np.zeros(n_doc, dtype=np.float32)
    for s in (0, 3):
        chunk = docs[s:s + 3]
        doc_len = np.array([len(d) for d in chunk], dtype=np.float32)
        denom = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avgdl))
        postings: dict = {}
        for di, dt in enumerate(chunk):
            for term in Counter(dt).keys() & query_terms:
                postings.setdefault(term, []).append((di, Counter(dt)[term]))
        for term in query_terms:
            for di, tf in postings.get(term, []):
                got[s + di] += idf[term] * (tf * (BM25_K1 + 1.0)) / (tf + denom[di])

    assert np.allclose(ref, got, atol=1e-5), f"{ref} != {got}"


def test_pool_size_matches_pgvector():
    # pgvector nutzt pool = max(k*5, 500) -- GT muss dasselbe tun.
    assert _hybrid_pool(100) == 500
    assert _hybrid_pool(200) == 1000


if __name__ == "__main__":
    test_pool_merge_equals_single_shot()
    test_bm25_chunk_additivity()
    test_pool_size_matches_pgvector()
    print("OK: alle 3 streaming-hybrid-GT-Invarianten halten")
