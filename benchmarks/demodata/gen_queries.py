#!/usr/bin/env python3
"""
Erzeugt Query-Embeddings und Ground-Truth Top-k fuer einen Datensatz.

Liest die corpus-chunks (output von generate.py), zieht n Query-Vektoren aus
der gleichen Verteilung und berechnet per Brute-Force die Top-k aehnlichsten
Korpus-Eintraege (Cosine). Output:
    queries.npy            -- shape (n_queries, dim), float32, normalisiert
    ground_truth_ids.npy   -- shape (n_queries, top_k), int64
    ground_truth_scores.npy -- shape (n_queries, top_k), float32

Achtung: Brute-Force ueber den ganzen Korpus. Fuer S/M auf CPU machbar,
fuer L/XL besser mit GPU (faiss-gpu, cuml) oder grossem Server-RAM.

Beispiel:
    python gen_queries.py --corpus-dir ./out/S --output-dir ./out/S/queries
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--corpus-dir", required=True, type=Path,
                   help="Verzeichnis mit chunk_*.parquet (von generate.py)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-queries", type=int, default=10_000)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=4242,
                   help="Anderer Seed als der Korpus -- Queries sind unabhaengig")
    p.add_argument("--query-batch", type=int, default=128,
                   help="Queries pro Distance-Matrix Batch (RAM-Trade-off)")
    return p.parse_args()


def load_chunks(corpus_dir):
    paths = sorted(corpus_dir.glob("chunk_*.parquet"))
    if not paths:
        raise SystemExit(f"Keine Chunks unter {corpus_dir}")
    return paths


def read_chunk(path, dim):
    tbl = pq.read_table(path, columns=["id", "embedding"])
    ids = tbl["id"].to_numpy()
    emb_col = tbl["embedding"].combine_chunks()
    flat = emb_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    emb = flat.reshape(-1, dim)
    return ids, emb


def topk_merge(cur_ids, cur_scores, new_ids, new_scores, k):
    """Mischt aktuellen Top-k Stand mit neuen Kandidaten, hoehere Scores
    gewinnen (Cosine bei normalisierten Vektoren -> in [-1, 1])."""
    cat_scores = np.concatenate([cur_scores, new_scores], axis=1)
    cat_ids = np.concatenate([cur_ids, new_ids], axis=1)
    idx = np.argpartition(-cat_scores, k, axis=1)[:, :k]
    rows = np.arange(cat_scores.shape[0])[:, None]
    best_scores = cat_scores[rows, idx]
    best_ids = cat_ids[rows, idx]
    order = np.argsort(-best_scores, axis=1)
    return best_ids[rows, order], best_scores[rows, order]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    Q = args.n_queries
    K = args.top_k

    # Queries erzeugen
    queries = rng.standard_normal((Q, args.dim), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    np.save(args.output_dir / "queries.npy", queries)
    print(f"Queries: {queries.shape}, gespeichert in queries.npy", flush=True)

    # Running Top-k Stand ueber alle Chunks
    topk_ids = np.full((Q, K), -1, dtype=np.int64)
    topk_scores = np.full((Q, K), -np.inf, dtype=np.float32)

    chunks = load_chunks(args.corpus_dir)
    t0 = time.time()
    seen = 0
    for ci, cpath in enumerate(chunks):
        ids, emb = read_chunk(cpath, args.dim)
        seen += len(ids)
        # Pro Query-Batch eine Distance-Matrix
        for qs in range(0, Q, args.query_batch):
            qe = min(qs + args.query_batch, Q)
            qb = queries[qs:qe]
            sims = qb @ emb.T  # (batch, n_chunk)
            new_ids = np.broadcast_to(ids, sims.shape)
            topk_ids[qs:qe], topk_scores[qs:qe] = topk_merge(
                topk_ids[qs:qe], topk_scores[qs:qe],
                new_ids, sims, K,
            )
        dt = time.time() - t0
        print(f"  chunk {ci+1:>3}/{len(chunks)}  seen={seen:>12,}  "
              f"elapsed={dt:6.1f}s", flush=True)

    np.save(args.output_dir / "ground_truth_ids.npy", topk_ids)
    np.save(args.output_dir / "ground_truth_scores.npy", topk_scores)
    total = time.time() - t0
    print(f"\nFertig in {total:.1f}s")
    print(f"Output:")
    print(f"  queries.npy              ({Q}, {args.dim})")
    print(f"  ground_truth_ids.npy     ({Q}, {K})")
    print(f"  ground_truth_scores.npy  ({Q}, {K})")


if __name__ == "__main__":
    main()
