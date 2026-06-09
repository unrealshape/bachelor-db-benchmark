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
import sys
import time
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


def hybrid_gt(corpus_dir: Path, alpha: float, top_k: int) -> None:
    """Berechnet Hybrid-GT per Reciprocal Rank Fusion (RRF), passend zu den
    nativen DB-Hybrid-Queries (Weaviate Ranked-Fusion, pgvector RRF-SQL):

        score = alpha / (RRF_K + vrank) + (1-alpha) / (RRF_K + trank)

    vrank = Rang nach Cosine (absteigend) ueber den ganzen Korpus, trank = Rang
    nach BM25 (absteigend) unter den Treffern mit BM25 > 0. Dokumente ohne
    BM25-Treffer tragen nur den Vektor-Term bei."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit("rank-bm25 fehlt: pip install rank-bm25")

    q_dir = corpus_dir / "queries"
    queries = np.load(q_dir / "queries.npy").astype(np.float32)
    query_texts_tbl = pq.read_table(q_dir / "queries.parquet", columns=["review_text"])
    query_texts = [t.as_py() for t in query_texts_tbl["review_text"]]
    Q = queries.shape[0]
    print(f"  Hybrid: alpha={alpha} | Queries: {Q} | Top-k: {top_k}")

    chunks = _read_corpus_chunks(corpus_dir)
    all_ids = []
    all_texts = []
    all_embs = []
    for chunk in chunks:
        tbl = pq.read_table(chunk, columns=["id", "review_text", "embedding"])
        all_ids.append(np.asarray(tbl["id"]).astype(np.int64))
        all_texts.extend(t.as_py() for t in tbl["review_text"])
        all_embs.append(np.stack([np.asarray(v.as_py(), dtype=np.float32)
                                  for v in tbl["embedding"]]))
    ids = np.concatenate(all_ids)
    embs = np.concatenate(all_embs, axis=0)
    print(f"  Korpus: {ids.shape[0]:,} Vektoren")

    print("  BM25-Index bauen...", flush=True)
    tokenized = [t.lower().split() for t in all_texts]
    bm25 = BM25Okapi(tokenized)

    print("  Scores berechnen...", flush=True)
    top_scores = np.zeros((Q, top_k), dtype=np.float32)
    top_ids = np.zeros((Q, top_k), dtype=np.int64)
    n_doc = ids.shape[0]
    for q in range(Q):
        cos = embs @ queries[q]
        bm = bm25.get_scores(query_texts[q].lower().split()).astype(np.float32)

        # Vektor-Raenge (1-basiert) ueber alle Dokumente.
        vorder = np.argsort(-cos)
        vrank = np.empty(n_doc, dtype=np.float64)
        vrank[vorder] = np.arange(1, n_doc + 1)

        # Text-Raenge nur unter BM25 > 0; Rest traegt keinen Text-Term bei.
        rrf_text = np.zeros(n_doc, dtype=np.float64)
        pos = np.where(bm > 0)[0]
        if pos.size:
            torder = pos[np.argsort(-bm[pos])]
            trank = np.arange(1, torder.size + 1)
            rrf_text[torder] = 1.0 / (RRF_K + trank)

        score = alpha * (1.0 / (RRF_K + vrank)) + (1.0 - alpha) * rrf_text
        order = np.argsort(-score)[:top_k]
        top_scores[q] = score[order].astype(np.float32)
        top_ids[q] = ids[order]
        if (q + 1) % 50 == 0:
            print(f"    {q+1}/{Q}", flush=True)

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
