#!/usr/bin/env python3
"""Echter Korpus aus Amazon Reviews + MiniLM-Embeddings für die Thesis-Stufen.

Pipeline:
  1. Stream Reviews aus einer Amazon-Reviews HuggingFace-Datasource
  2. Encode mit sentence-transformers/all-MiniLM-L6-v2 (384 dim)
  3. Schreibe Parquet-Chunks mit dem Exposé-Schema:
        id, product_id, product_title, user_id, rating, review_text,
        timestamp, embedding
  4. Halte n_queries Reviews als Query-Set zurück
  5. Berechne Brute-Force Top-100 Ground Truth via numpy

Beispiel:
  python build_corpus.py --output-dir ~/.cache/bachelor-db-benchmark/S --size S
  python build_corpus.py --output-dir ./tmp --n-records 5000 --n-queries 200

Stufen aus Exposé Kapitel 5.1.3:
  S   = 100.000 Dokumente
  M   = 500.000 Dokumente
  L   = 1.000.000 Dokumente
  XL  = 5.000.000 Dokumente
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SIZE_PRESETS = {
    "T":    20_000,
    "T2":  100_000,
    "S":   100_000,
    "M":   500_000,
    "L": 1_000_000,
    "XL":5_000_000,
}

DEFAULT_DATASET = "fancyzhx/amazon_polarity"
DEFAULT_CONFIG = None
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# fancyzhx/amazon_polarity hat: label (0/1), title, content.
# Das deckt review_text (= content) und product_title (= title) ab.
# user_id, product_id, timestamp und ein 5-Sterne-rating gibt es nicht --
# wir synthetisieren sie konsistent (Hash-basiert), damit die Daten der
# Pipeline genügen (Embedding-basierte Tests + Filter rating>=4).
SOURCE_FIELDS = {
    "review_text": "content",
    "product_title": "title",
    "label": "label",  # 0=negativ, 1=positiv -> rating 1.0 oder 5.0
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, type=Path)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--size", choices=list(SIZE_PRESETS))
    grp.add_argument("--n-records", type=int)
    p.add_argument("--n-queries", type=int, default=2000,
                   help="Wieviele Reviews als Query-Set zurückhalten")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--chunk-records", type=int, default=100_000)
    p.add_argument("--compression", default="zstd",
                   choices=["zstd", "snappy", "none"])
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps -- Auto-Detect wenn leer")
    p.add_argument("--seed", type=int, default=4242)
    return p.parse_args()


# ---------------------------------------------------------------------------

def autodetect_device(forced: str | None) -> str:
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stream_reviews(dataset: str, config: str | None, n: int):
    """Yields normalisierte Review-Dicts. Streamt, lädt nicht alles in RAM.
    Synthetisiert konsistent fehlende Metadaten (user_id, product_id,
    timestamp). rating wird aus dem polarity-label abgeleitet."""
    import hashlib
    from datetime import datetime, timedelta
    from datasets import load_dataset

    print(f"Streaming {dataset}/{config} (n={n:,})...", flush=True)
    if config:
        ds = load_dataset(dataset, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset, split="train", streaming=True)

    base_time = datetime(2023, 1, 1)
    seen = 0
    for row in ds:
        text = row.get(SOURCE_FIELDS["review_text"]) or ""
        title = row.get(SOURCE_FIELDS["product_title"]) or ""
        if len(text) < 20:
            continue
        label = row.get(SOURCE_FIELDS["label"])
        # Rating-Mapping: polarity 0->1.0, 1->5.0. Daraus lassen sich
        # rating>=4-Filter sinnvoll testen (positive Reviews).
        rating = 5.0 if label == 1 else 1.0
        # product_id = stabiler Hash über den Titel -> Reviews zum gleichen
        # Produkt gruppieren sich.
        product_id = hashlib.md5(title.encode("utf-8", errors="replace")).hexdigest()[:12]
        # user_id und timestamp synthetisieren (für konsistente Reproduktion
        # via Index).
        user_id = f"user_{seen:08d}"
        ts = (base_time + timedelta(seconds=seen * 47)).isoformat()
        yield {
            "review_text": text,
            "product_title": title,
            "product_id": product_id,
            "user_id": user_id,
            "rating": rating,
            "timestamp": ts,
        }
        seen += 1
        if seen >= n:
            return


def encode_batches(items: list[dict], model_name: str, device: str, batch_size: int):
    """Embeddet alle review_text-Felder in einem Rutsch und gibt das
    (N, 384) Embedding-Array zurück, L2-normalisiert für Cosine-Similarity."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    texts = [it["review_text"] for it in items]
    print(f"Embedding {len(texts):,} Texte auf {device}...", flush=True)
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"  fertig in {time.time()-t0:.1f}s -- {emb.shape}", flush=True)
    return emb


def write_corpus(out_dir: Path, items: list[dict], emb: np.ndarray,
                 chunk_records: int, compression: str):
    """Schreibt Parquet-Chunks mit dem Exposé-Schema."""
    n = len(items)
    dim = emb.shape[1]
    compr = None if compression == "none" else compression

    ids = np.arange(n, dtype=np.int64)
    for offset in range(0, n, chunk_records):
        end = min(offset + chunk_records, n)
        idx = (offset, end)
        chunk_ids = ids[offset:end]
        chunk_emb = emb[offset:end]
        chunk_items = items[offset:end]

        flat = pa.array(chunk_emb.reshape(-1), type=pa.float32())
        emb_arr = pa.FixedSizeListArray.from_arrays(flat, dim)

        table = pa.Table.from_pydict({
            "id": pa.array(chunk_ids, type=pa.int64()),
            "product_id": pa.array(
                [it["product_id"] for it in chunk_items], type=pa.string(),
            ),
            "product_title": pa.array(
                [it["product_title"] for it in chunk_items], type=pa.string(),
            ),
            "user_id": pa.array(
                [it["user_id"] for it in chunk_items], type=pa.string(),
            ),
            "rating": pa.array(
                [it["rating"] for it in chunk_items], type=pa.float32(),
            ),
            "review_text": pa.array(
                [it["review_text"] for it in chunk_items], type=pa.string(),
            ),
            "timestamp": pa.array(
                [str(it["timestamp"]) if it["timestamp"] is not None else None
                 for it in chunk_items],
                type=pa.string(),
            ),
            "embedding": emb_arr,
        })
        idx_str = f"{offset:08d}"
        chunk_path = out_dir / f"chunk_{idx_str}.parquet"
        pq.write_table(table, chunk_path, compression=compr)
        print(f"  Chunk {chunk_path.name}: {end-offset:,} Zeilen", flush=True)


def build_ground_truth(queries: np.ndarray, corpus: np.ndarray, top_k: int,
                       batch_size: int = 256):
    """Brute-Force Top-k via Matrix-Mul (normalisierte Vektoren -> Dot
    Product ≡ Cosine)."""
    Q = queries.shape[0]
    N = corpus.shape[0]
    print(f"Ground Truth: Brute-Force {Q} Queries vs {N:,} Korpus-Vektoren...",
          flush=True)
    t0 = time.time()
    top_ids = np.full((Q, top_k), -1, dtype=np.int64)
    top_scores = np.full((Q, top_k), -np.inf, dtype=np.float32)
    # Chunked über Queries
    for qs in range(0, Q, batch_size):
        qe = min(qs + batch_size, Q)
        sims = queries[qs:qe] @ corpus.T  # (b, N)
        # Top-k argpartition
        idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
        rows = np.arange(sims.shape[0])[:, None]
        sc = sims[rows, idx]
        order = np.argsort(-sc, axis=1)
        top_ids[qs:qe] = idx[rows, order]
        top_scores[qs:qe] = sc[rows, order]
    print(f"  fertig in {time.time()-t0:.1f}s", flush=True)
    return top_ids, top_scores


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    n_total = args.n_records if args.n_records else SIZE_PRESETS[args.size]
    n_total_with_queries = n_total + args.n_queries

    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = args.output_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    device = autodetect_device(args.device)
    rng = np.random.default_rng(args.seed)

    # 1. Streamen
    items = list(stream_reviews(args.dataset, args.config, n_total_with_queries))
    print(f"  geladen: {len(items):,} Reviews (gefiltert)", flush=True)
    if len(items) < n_total_with_queries:
        # Datasource gibt nicht genug -- mit dem zurück was wir haben.
        print(f"WARNUNG: nur {len(items)} statt {n_total_with_queries} verfügbar.",
              flush=True)
        n_total = max(0, len(items) - args.n_queries)

    # Shuffle damit Korpus/Query nicht durch Reihenfolge biasen
    perm = rng.permutation(len(items))
    items = [items[i] for i in perm]
    corpus_items = items[:n_total]
    query_items = items[n_total:n_total + args.n_queries]

    # 2. Embedden in einem Schwung (Korpus + Queries gleichzeitig, gleicher Model-Load)
    all_emb = encode_batches(
        corpus_items + query_items,
        args.model, device, args.batch_size,
    )
    corpus_emb = all_emb[:n_total]
    query_emb = all_emb[n_total:n_total + args.n_queries]

    # 3. Korpus schreiben
    print(f"Schreibe Korpus-Chunks nach {args.output_dir}...", flush=True)
    write_corpus(args.output_dir, corpus_items, corpus_emb,
                 args.chunk_records, args.compression)

    # 4. Queries persistieren (Vektor + Metadaten)
    np.save(queries_dir / "queries.npy", query_emb)
    # Query-Metadaten als Parquet (Rating + Text + product_id für Filter und Hybrid)
    q_table = pa.Table.from_pydict({
        "id": pa.array(np.arange(len(query_items), dtype=np.int64)),
        "rating": pa.array([it["rating"] for it in query_items], type=pa.float32()),
        "review_text": pa.array([it["review_text"] for it in query_items]),
        "product_id": pa.array([it["product_id"] for it in query_items]),
        "product_title": pa.array([it["product_title"] for it in query_items]),
    })
    pq.write_table(q_table, queries_dir / "queries.parquet")
    print(f"  queries.npy {query_emb.shape}", flush=True)

    # 5. Ground Truth
    gt_ids, gt_scores = build_ground_truth(query_emb, corpus_emb, args.top_k)
    np.save(queries_dir / "ground_truth_ids.npy", gt_ids)
    np.save(queries_dir / "ground_truth_scores.npy", gt_scores)

    # 6. Größen-Übersicht
    total_bytes = sum(p.stat().st_size for p in args.output_dir.glob("chunk_*.parquet"))
    total_bytes += sum(p.stat().st_size for p in queries_dir.glob("*"))
    print(f"\nFertig. Korpus: {n_total:,} | Queries: {len(query_items)}")
    print(f"On disk: {total_bytes / 2**30:.2f} GiB ({total_bytes:,} bytes)")


if __name__ == "__main__":
    main()
