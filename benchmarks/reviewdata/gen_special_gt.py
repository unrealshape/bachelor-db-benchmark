#!/usr/bin/env python3
"""Spezialisierte Ground Truth fuer filtered und hybrid Workloads.

`gen_queries.py` baut die Default-GT als Brute-Force-Cosine ueber den ganzen
Korpus. Fuer filtered und hybrid Workloads stimmt diese GT nicht: ein
Metadatenfilter schliesst Treffer aus, die in der Default-GT vorkommen;
Hybrid sortiert mit BM25-Anteil anders. Dieses Script erzeugt fuer beide
Faelle separate GT-Files mit eindeutigen Namen.

Aufruf
------
    python gen_special_gt.py --corpus-dir <CORPUS> --filter rating_gte=4
    python gen_special_gt.py --corpus-dir <CORPUS> --hybrid-alpha 0.5

Output (im Queries-Verzeichnis):
    ground_truth_filter_<spec>_ids.npy
    ground_truth_filter_<spec>_scores.npy
    queries_groundtruth_filter_<spec>.parquet
    (analog fuer hybrid_alpha_<NN>)

Der `<spec>` Suffix landet auch im Config-Feld `filter`/`hybrid` und wird
von runner.py genutzt, um die passende GT zu laden.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_corpus_chunks(corpus_dir: Path):
    chunks = sorted(corpus_dir.glob("chunk_*.parquet"))
    if not chunks:
        raise SystemExit(f"Keine Korpus-Chunks in {corpus_dir}")
    return chunks


def _filter_spec_to_suffix(spec: str) -> str:
    # 'rating_gte=4' -> 'rating_gte_4', identisch zum Schema in runner.py
    return spec.replace("=", "_").replace(":", "_").replace(".", "_")


def _hybrid_suffix(alpha: float) -> str:
    return f"alpha_{int(round(alpha * 100)):02d}"


def filter_gt(corpus_dir: Path, filter_spec: str, top_k: int) -> None:
    """Berechnet Brute-Force-GT nur ueber Reviews die den Filter erfuellen."""
    m = re.match(r"^(\w+)=([\d\.]+)$", filter_spec)
    if not m:
        raise SystemExit(f"Filter erwartet 'field=value', bekommen: {filter_spec}")
    field, val = m.group(1), m.group(2)
    # rating_gte=4 -> field=rating, op=gte, threshold=4
    m2 = re.match(r"^(\w+)_gte$", field)
    if not m2:
        raise SystemExit(f"nur *_gte Filter unterstuetzt, nicht: {field}")
    base_field = m2.group(1)
    threshold = float(val)

    q_dir = corpus_dir / "queries"
    queries = np.load(q_dir / "queries.npy").astype(np.float32)
    Q = queries.shape[0]
    print(f"  Filter: {base_field} >= {threshold} | Queries: {Q} | Top-k: {top_k}")

    # Running Top-k State (Scores, IDs) absteigend nach Score
    top_scores = np.full((Q, top_k), -np.inf, dtype=np.float32)
    top_ids = np.full((Q, top_k), -1, dtype=np.int64)

    chunks = _read_corpus_chunks(corpus_dir)
    n_keep_total = 0
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        tbl = pq.read_table(chunk, columns=["id", base_field, "embedding"])
        mask = np.asarray(tbl[base_field]) >= threshold
        if not mask.any():
            print(f"    chunk {ci+1}/{len(chunks)}  filter leer  {time.time()-t0:.1f}s")
            continue
        ids = np.asarray(tbl["id"])[mask].astype(np.int64)
        emb = np.stack([np.asarray(v.as_py(), dtype=np.float32)
                        for v in tbl["embedding"]])[mask]
        n_keep_total += emb.shape[0]
        # Scores: Q x K_in_chunk
        scores = queries @ emb.T
        # Merge mit running top-k
        merged_scores = np.concatenate([top_scores, scores], axis=1)
        merged_ids = np.concatenate([top_ids, np.broadcast_to(ids, (Q, ids.shape[0]))], axis=1)
        order = np.argsort(-merged_scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(merged_scores, order, axis=1)
        top_ids = np.take_along_axis(merged_ids, order, axis=1)
        print(f"    chunk {ci+1}/{len(chunks)}  keep={emb.shape[0]:>8,}  total_keep={n_keep_total:>10,}  {time.time()-t0:.1f}s")

    suffix = _filter_spec_to_suffix(filter_spec)
    np.save(q_dir / f"ground_truth_filter_{suffix}_ids.npy", top_ids)
    np.save(q_dir / f"ground_truth_filter_{suffix}_scores.npy", top_scores)
    table = pa.table({
        "query_id": pa.array(np.arange(Q, dtype=np.int64)),
        "gt_ids": pa.array([row.tolist() for row in top_ids], type=pa.list_(pa.int64())),
        "gt_scores": pa.array([row.tolist() for row in top_scores], type=pa.list_(pa.float32())),
    })
    pq.write_table(table, q_dir / f"queries_groundtruth_filter_{suffix}.parquet",
                   compression="zstd")
    print(f"\n  Gefilterte GT geschrieben fuer {suffix} (n_keep_total={n_keep_total:,})")


# RRF-Konstante. Muss zu Weaviates Ranked-Fusion (RANK_CONSTANT=60) passen,
# damit die GT dieselbe Fusion abbildet wie die nativen DB-Queries.
RRF_K = 60


# BM25-Parameter (Okapi). Manuelle Implementierung statt rank-bm25, weil
# BM25Okapi den GANZEN tokenisierten Korpus im RAM haelt (OOM ab L). Wir
# streamen stattdessen.
BM25_K1 = 1.5
BM25_B = 0.75

# Pool-Groesse fuer die RRF-Fusion. Passt zu pgvector (`pool = max(k*5, 500)`)
# und Weaviates Ranked-Fusion: die nativen DB-Queries fusionieren ebenfalls nur
# Top-Pools, nicht den Vollkorpus. Daher ist die Pool-RRF-GT sogar treuer zur
# DB-Realitaet als die alte Voll-Rang-Variante.
def _hybrid_pool(top_k: int) -> int:
    return max(top_k * 5, 500)


def _pool_merge(cur_ids, cur_scores, new_ids, new_scores, k):
    """Mergt (cur) mit (new) und behaelt die Top-k nach Score (absteigend
    sortiert). Identisch zur Logik in gen_queries.topk_merge."""
    cat_scores = np.concatenate([cur_scores, new_scores], axis=1)
    cat_ids = np.concatenate([cur_ids, new_ids], axis=1)
    kth = min(k, cat_scores.shape[1] - 1)
    idx = np.argpartition(-cat_scores, kth, axis=1)[:, :k]
    rows = np.arange(cat_scores.shape[0])[:, None]
    best_scores = cat_scores[rows, idx]
    best_ids = cat_ids[rows, idx]
    order = np.argsort(-best_scores, axis=1)
    return best_ids[rows, order], best_scores[rows, order]


def hybrid_gt(corpus_dir: Path, alpha: float, top_k: int) -> None:
    """Hybrid-GT per Reciprocal Rank Fusion (RRF), STREAMING (skaliert auf L+):

        score = alpha / (RRF_K + vrank) + (1-alpha) / (RRF_K + trank)

    vrank/trank = Rang im Vektor- bzw. BM25-Top-Pool (nicht globaler Voll-Rang).
    Das deckt sich mit den nativen DB-Hybrid-Queries, die ebenfalls Top-Pools
    fusionieren. Zwei Streaming-Paesse, Peak-RAM = ein Chunk + (Q x Pool):

      Pass 1: globale BM25-Statistik (df pro Query-Term, avgdl) chunkweise.
      Pass 2: pro Chunk Cosine + BM25 scoren, in laufende Top-Pools mergen.
      Fusion: RRF ueber die Pool-Raenge.
    """
    q_dir = corpus_dir / "queries"
    queries = np.load(q_dir / "queries.npy").astype(np.float32)
    query_texts_tbl = pq.read_table(q_dir / "queries.parquet", columns=["review_text"])
    query_texts = [t.as_py() for t in query_texts_tbl["review_text"]]
    Q = queries.shape[0]
    pool = _hybrid_pool(top_k)
    print(f"  Hybrid: alpha={alpha} | Queries: {Q} | Top-k: {top_k} | Pool: {pool}")

    query_term_sets = [set((t or "").lower().split()) for t in query_texts]
    all_query_terms: set = set().union(*query_term_sets) if query_term_sets else set()

    chunks = _read_corpus_chunks(corpus_dir)

    # ---- Pass 1: globale BM25-Statistik (streaming) ----
    print("  Pass 1: BM25-Statistik (df, avgdl) streamend...", flush=True)
    df: dict[str, int] = {}
    n_doc = 0
    total_len = 0
    for ci, chunk in enumerate(chunks):
        tbl = pq.read_table(chunk, columns=["review_text"])
        for t in tbl["review_text"]:
            toks = (t.as_py() or "").lower().split()
            n_doc += 1
            total_len += len(toks)
            for term in set(toks) & all_query_terms:
                df[term] = df.get(term, 0) + 1
    avgdl = (total_len / n_doc) if n_doc else 1.0
    idf = {t: math.log(1 + (n_doc - dfi + 0.5) / (dfi + 0.5)) for t, dfi in df.items()}
    print(f"    n_doc={n_doc:,}  avgdl={avgdl:.1f}  query-vocab∩korpus={len(idf):,}",
          flush=True)

    # ---- Pass 2: streaming scoring + laufende Top-Pools ----
    print("  Pass 2: Cosine + BM25 streamend...", flush=True)
    vec_ids = np.full((Q, pool), -1, dtype=np.int64)
    vec_scr = np.full((Q, pool), -np.inf, dtype=np.float32)
    bm_ids = np.full((Q, pool), -1, dtype=np.int64)
    bm_scr = np.full((Q, pool), -np.inf, dtype=np.float32)
    QBATCH = 64

    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        tbl = pq.read_table(chunk, columns=["id", "review_text", "embedding"])
        ids = np.asarray(tbl["id"]).astype(np.int64)
        emb = np.stack([np.asarray(v.as_py(), dtype=np.float32)
                        for v in tbl["embedding"]])
        n_chunk = len(ids)

        # Vektor: batched Cosine -> Vektor-Pool
        for qs in range(0, Q, QBATCH):
            qe = min(qs + QBATCH, Q)
            sims = (queries[qs:qe] @ emb.T).astype(np.float32)
            new_ids = np.broadcast_to(ids, sims.shape)
            vec_ids[qs:qe], vec_scr[qs:qe] = _pool_merge(
                vec_ids[qs:qe], vec_scr[qs:qe], new_ids, sims, pool)

        # BM25: chunk-lokaler Inverted-Index nur ueber Query-Terme
        doc_tokens = [(t.as_py() or "").lower().split() for t in tbl["review_text"]]
        doc_len = np.array([len(dt) for dt in doc_tokens], dtype=np.float32)
        denom = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avgdl))
        postings: dict[str, list] = {}
        for di, dt in enumerate(doc_tokens):
            if not dt:
                continue
            counts = Counter(dt)
            for term in counts.keys() & all_query_terms:
                postings.setdefault(term, []).append((di, counts[term]))

        bm_chunk = np.zeros((Q, n_chunk), dtype=np.float32)
        for q in range(Q):
            row = bm_chunk[q]
            for term in query_term_sets[q]:
                idf_t = idf.get(term)
                post = postings.get(term)
                if idf_t is None or not post:
                    continue
                for di, tf in post:
                    row[di] += idf_t * (tf * (BM25_K1 + 1.0)) / (tf + denom[di])
        for qs in range(0, Q, QBATCH):
            qe = min(qs + QBATCH, Q)
            new_ids = np.broadcast_to(ids, (qe - qs, n_chunk))
            bm_ids[qs:qe], bm_scr[qs:qe] = _pool_merge(
                bm_ids[qs:qe], bm_scr[qs:qe], new_ids, bm_chunk[qs:qe], pool)

        print(f"    chunk {ci+1}/{len(chunks)}  n={n_chunk:,}  {time.time()-t0:.1f}s",
              flush=True)

    # ---- Fusion: RRF ueber die Pool-Raenge ----
    print("  RRF-Fusion...", flush=True)
    top_scores = np.zeros((Q, top_k), dtype=np.float32)
    top_ids = np.full((Q, top_k), -1, dtype=np.int64)
    for q in range(Q):
        vrank = {int(i): r + 1 for r, i in enumerate(vec_ids[q]) if i >= 0}
        trank = {int(i): r + 1 for r, i in enumerate(bm_ids[q]) if i >= 0}
        scored = []
        for cid in set(vrank) | set(trank):
            s = (alpha / (RRF_K + vrank.get(cid, 10**9))
                 + (1.0 - alpha) / (RRF_K + trank.get(cid, 10**9)))
            scored.append((s, cid))
        scored.sort(reverse=True)
        for j in range(min(top_k, len(scored))):
            top_scores[q, j] = scored[j][0]
            top_ids[q, j] = scored[j][1]

    suffix = _hybrid_suffix(alpha)
    np.save(q_dir / f"ground_truth_hybrid_{suffix}_ids.npy", top_ids)
    np.save(q_dir / f"ground_truth_hybrid_{suffix}_scores.npy", top_scores)
    table = pa.table({
        "query_id": pa.array(np.arange(Q, dtype=np.int64)),
        "gt_ids": pa.array([row.tolist() for row in top_ids], type=pa.list_(pa.int64())),
        "gt_scores": pa.array([row.tolist() for row in top_scores], type=pa.list_(pa.float32())),
    })
    pq.write_table(table, q_dir / f"queries_groundtruth_hybrid_{suffix}.parquet",
                   compression="zstd")
    print(f"\n  Hybrid-GT geschrieben fuer {suffix}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--filter", type=str, help="z.B. 'rating_gte=4'")
    g.add_argument("--hybrid-alpha", type=float, help="0..1, BM25 + Vektor Mix")
    args = p.parse_args()

    if args.filter:
        filter_gt(args.corpus_dir, args.filter, args.top_k)
    else:
        hybrid_gt(args.corpus_dir, args.hybrid_alpha, args.top_k)


if __name__ == "__main__":
    main()
