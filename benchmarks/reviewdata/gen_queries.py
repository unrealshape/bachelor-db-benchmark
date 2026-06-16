#!/usr/bin/env python3
"""Erzeugt Query-Embeddings + Brute-Force Ground Truth fuer einen Reviews-Korpus.

Der Korpus wird von `load.py` geschrieben (Parquet-Chunks + corpus_meta.json).
Die Queries kommen aus einer **separaten Held-Out-Partition** der gleichen
HuggingFace-Quelle (McAuley-Lab/Amazon-Reviews-2023), damit die Queries aus
derselben Verteilung wie der Korpus stammen, aber **nicht im Korpus enthalten
sind**. Konkret:

  1. Aus jeder Korpus-Kategorie wird der erste Block (n_skip Zeilen) vom Korpus
     belegt; Queries werden ab einem festen `query_offset` weit hinter dem
     letzten Korpus-Eintrag gezogen. So koennen Korpus und Queries reproduzierbar
     ohne Ueberlappung gezogen werden.
  2. Pro Stufe min. 1.000 Query-Embeddings (Thesis 5.1.3).
  3. Embeddings via `BAAI/bge-large-en-v1.5` (1024 dim, L2-normalisiert).
     Query-Texte werden vor dem Embedden mit der BGE-Instruction prefixed:
     "Represent this sentence for searching relevant passages: ".
     Passages (Korpus) bekommen das nicht — so will es das BGE-Paper.

Ground Truth
------------
Brute-Force Top-100 ueber den ganzen Korpus, Cosine = Dot Product bei
normalisierten Vektoren. Berechnung chunked: pro Korpus-Chunk wird die
Aehnlichkeit zu allen Queries berechnet und mit dem running Top-100 gemerged.
Damit bleibt Peak-Speicher = einen Korpus-Chunk + (Q x K)-State.

Output
------
Im `--output-dir` (Default: `<corpus-dir>/queries/`):

  queries.parquet              -- id, product_id, rating, review_text, ...
  queries.npy                  -- (Q, 1024) float32, L2-normalisiert
  ground_truth_ids.npy         -- (Q, 100) int64
  ground_truth_scores.npy      -- (Q, 100) float32
  queries_groundtruth.parquet  -- (Q, 100) als Parquet (Tabelle mit Spalten
                                  query_id, gt_ids:list<int64>,
                                  gt_scores:list<float32>)

Resume
------
queries.npy / queries.parquet werden nur neu berechnet, wenn sie fehlen.
Ground Truth wird neu berechnet, wenn die Korpus-Chunks juenger sind als
`ground_truth_ids.npy` (oder wenn `--force-gt` gesetzt ist).

Aufruf
------
    python gen_queries.py --corpus-dir ~/.cache/bachelor-db-benchmark/reviewdata/S
    python gen_queries.py --corpus-dir ~/.cache/bachelor-db-benchmark/reviewdata/S \
        --n-queries 2000 --top-k 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Konstanten + Helpers aus load.py wiederverwenden, damit Schema, Modell
# und Kategorien aus einer Hand kommen.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import (  # noqa: E402
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BGE_QUERY_INSTRUCTION,
    HF_REPO_ID,
    REVIEW_PATH_TEMPLATE,
    MIN_REVIEW_CHARS,
    DEFAULT_CATEGORIES,
    BGEEmbedder,
    _hf_download,
    _open_jsonl,
    pick_device,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--corpus-dir", required=True, type=Path,
                   help="Verzeichnis mit chunk_*.parquet + corpus_meta.json")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <corpus-dir>/queries/")
    p.add_argument("--n-queries", type=int, default=1_000,
                   help="Thesis-Minimum 1.000")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--query-offset", type=int, default=10_000_000,
                   help="Ab welcher Zeile pro Kategorie Queries gezogen werden. "
                        "Default ist so gross, dass Korpus + Queries garantiert "
                        "disjoint bleiben (Kategorien haben Mio. Reviews).")
    p.add_argument("--max-chars", type=int, default=2000,
                   help="Trunkierung vor Embedding (BGE-Large hat 512 Token Limit).")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Embedding-Batchgroesse. Default: env BENCH_EMBED_BATCH oder "
                        "64 (CPU) / 256 (GPU/MPS).")
    p.add_argument("--device", type=str, default=None,
                   choices=("cuda", "mps", "cpu"),
                   help="Erzwingt ein Device. Default: auto.")
    p.add_argument("--gt-batch", type=int, default=64,
                   help="Queries pro Batch in der Brute-Force-Suche")
    p.add_argument("--force-gt", action="store_true",
                   help="Ground Truth neu berechnen, auch wenn vorhanden")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan zeigen, nichts schreiben, kein Modell-Download")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Held-Out-Sampling

def sample_query_texts(
    corpus_meta: dict,
    n_queries: int,
    query_offset: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> list[dict]:
    """Zieht n_queries Reviews aus der Held-Out-Partition der Korpus-Kategorien."""
    categories: list[str] = corpus_meta.get("categories", list(DEFAULT_CATEGORIES))
    rng = np.random.default_rng(seed)
    # Verteile die Queries einigermassen gleichmaessig auf die Kategorien.
    per_cat_base = n_queries // len(categories)
    remainder = n_queries - per_cat_base * len(categories)
    quota = [per_cat_base + (1 if i < remainder else 0) for i in range(len(categories))]

    out: list[dict] = []
    for cat, q in zip(categories, quota):
        if q == 0:
            continue
        print(f"  Held-Out aus {cat}: {q} Queries ab Zeile ~{query_offset:,}",
              flush=True)
        p = _hf_download(REVIEW_PATH_TEMPLATE.format(cat=cat))
        picked = 0
        line_no = 0
        with _open_jsonl(p) as f:
            for line in f:
                line_no += 1
                if line_no <= query_offset:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = d.get("text") or d.get("title") or ""
                if len(text) < min_chars:
                    continue
                if len(text) > max_chars:
                    text = text[:max_chars]
                out.append({
                    "product_id": d.get("parent_asin") or d.get("asin") or "",
                    "user_id": d.get("user_id") or "",
                    "rating": int(round(float(d.get("rating") or 0))),
                    "review_text": text,
                    "category": cat,
                })
                picked += 1
                if picked >= q:
                    break
        if picked < q:
            print(f"    WARNUNG: nur {picked}/{q} Held-Out-Reviews in {cat} "
                  f"(Datei zu klein?). Wird mit anderen Kategorien aufgefuellt.",
                  flush=True)
    # Wenn eine Kategorie zu wenig liefert, fuelle aus weiteren Held-Out-Bloecken
    # spaeterer Kategorien auf. Hier reicht ein einfacher Pass — wenn zu wenig,
    # warnen statt zu scheitern.
    if len(out) < n_queries:
        print(f"  WARNUNG: nur {len(out)}/{n_queries} Held-Out-Reviews insgesamt.",
              flush=True)
    # Shuffle, damit Reihenfolge nicht durch Kategorie sortiert ist
    perm = rng.permutation(len(out))
    return [out[i] for i in perm][:n_queries]


# ---------------------------------------------------------------------------
# Brute-Force Ground Truth

def load_corpus_chunks(corpus_dir: Path) -> list[Path]:
    chunks = sorted(corpus_dir.glob("chunk_*.parquet"))
    if not chunks:
        sys.exit(f"Keine chunk_*.parquet unter {corpus_dir}")
    return chunks


def read_chunk_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tbl = pq.read_table(path, columns=["id", "embedding"])
    ids = tbl["id"].to_numpy()
    flat = tbl["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False)
    emb = flat.astype(np.float32, copy=False).reshape(len(tbl), -1)
    return ids, emb


def topk_merge(cur_ids, cur_scores, new_ids, new_scores, k):
    cat_scores = np.concatenate([cur_scores, new_scores], axis=1)
    cat_ids = np.concatenate([cur_ids, new_ids], axis=1)
    idx = np.argpartition(-cat_scores, k, axis=1)[:, :k]
    rows = np.arange(cat_scores.shape[0])[:, None]
    best_scores = cat_scores[rows, idx]
    best_ids = cat_ids[rows, idx]
    order = np.argsort(-best_scores, axis=1)
    return best_ids[rows, order], best_scores[rows, order]


def brute_force_gt(
    queries: np.ndarray, corpus_dir: Path, top_k: int, batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    Q = queries.shape[0]
    chunks = load_corpus_chunks(corpus_dir)
    print(f"Brute-Force Top-{top_k} ueber {len(chunks)} Korpus-Chunks "
          f"({Q} Queries)...", flush=True)
    topk_ids = np.full((Q, top_k), -1, dtype=np.int64)
    topk_scores = np.full((Q, top_k), -np.inf, dtype=np.float32)

    t0 = time.time()
    seen = 0
    for ci, cpath in enumerate(chunks):
        ids, emb = read_chunk_embeddings(cpath)
        seen += len(ids)
        for qs in range(0, Q, batch):
            qe = min(qs + batch, Q)
            sims = queries[qs:qe] @ emb.T  # (b, n_chunk)
            new_ids = np.broadcast_to(ids, sims.shape)
            topk_ids[qs:qe], topk_scores[qs:qe] = topk_merge(
                topk_ids[qs:qe], topk_scores[qs:qe], new_ids, sims, top_k,
            )
        dt = time.time() - t0
        print(f"  chunk {ci+1:>3}/{len(chunks)}  seen={seen:>12,}  "
              f"elapsed={dt:6.1f}s", flush=True)

    return topk_ids, topk_scores


# ---------------------------------------------------------------------------
# Output

def write_queries_parquet(out_dir: Path, query_items: list[dict]) -> Path:
    q_table = pa.Table.from_pydict({
        "id": pa.array(np.arange(len(query_items), dtype=np.int64)),
        "product_id": pa.array([it["product_id"] for it in query_items]),
        "user_id": pa.array([it["user_id"] for it in query_items]),
        "rating": pa.array([it["rating"] for it in query_items], type=pa.int8()),
        "review_text": pa.array([it["review_text"] for it in query_items]),
        "category": pa.array([it["category"] for it in query_items]),
    })
    path = out_dir / "queries.parquet"
    pq.write_table(q_table, path)
    return path


def write_gt_parquet(out_dir: Path, gt_ids: np.ndarray, gt_scores: np.ndarray) -> Path:
    q = gt_ids.shape[0]
    table = pa.table({
        "query_id": pa.array(np.arange(q, dtype=np.int64)),
        "gt_ids": pa.array([row.tolist() for row in gt_ids],
                           type=pa.list_(pa.int64())),
        "gt_scores": pa.array([row.tolist() for row in gt_scores],
                              type=pa.list_(pa.float32())),
    })
    path = out_dir / "queries_groundtruth.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


# ---------------------------------------------------------------------------
# Dry-Run

def run_dry(args) -> None:
    device = args.device or pick_device()
    print(f"\n[DRY-RUN] Queries fuer Korpus unter {args.corpus_dir}")
    print(f"  Output:    {args.output_dir or (args.corpus_dir / 'queries')}")
    print(f"  Modell:    {EMBEDDING_MODEL}  (lokal, kein API-Key)")
    print(f"  Device:    {device}")
    print(f"  N Queries: {args.n_queries}")
    print(f"  Top-k:     {args.top_k}")
    print(f"  Held-Out-Offset: {args.query_offset:,}")
    print(f"  Query-Instruction: {BGE_QUERY_INSTRUCTION!r}")
    print()
    print("Kein API-Key noetig. Modell-Erstdownload ~1.3 GB nach ~/.cache/huggingface/.")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = args.output_dir or (args.corpus_dir / "queries")

    if args.dry_run:
        run_dry(args)
        return

    if not args.corpus_dir.exists():
        sys.exit(f"Korpus nicht gefunden: {args.corpus_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.corpus_dir / "corpus_meta.json"
    if not meta_path.exists():
        sys.exit(f"corpus_meta.json fehlt unter {args.corpus_dir} — Korpus "
                 f"unvollstaendig? Erst `load.py` zu Ende laufen lassen.")
    meta = json.loads(meta_path.read_text())
    if meta.get("embedding_dim") != EMBEDDING_DIM:
        sys.exit(
            f"Korpus-dim {meta.get('embedding_dim')} != {EMBEDDING_DIM} "
            f"({EMBEDDING_MODEL}). Falscher oder veralteter Korpus. "
            f"Loeschen oder neu bauen."
        )

    qvec_path = out_dir / "queries.npy"
    qmeta_path = out_dir / "queries.parquet"
    if qvec_path.exists() and qmeta_path.exists():
        print(f"Resume: queries.npy + queries.parquet vorhanden, ueberspringe Embedding.")
        queries = np.load(qvec_path)
        if queries.shape[1] != EMBEDDING_DIM:
            sys.exit(
                f"Vorhandene queries.npy hat dim {queries.shape[1]}, erwartet "
                f"{EMBEDDING_DIM}. Bitte {qvec_path} loeschen und neu erzeugen."
            )
    else:
        print(f"Held-Out-Sampling ({args.n_queries} Queries)...", flush=True)
        query_items = sample_query_texts(
            meta, args.n_queries, args.query_offset,
            MIN_REVIEW_CHARS, args.max_chars, args.seed,
        )
        if len(query_items) == 0:
            sys.exit("Keine Query-Texte gefunden — Held-Out-Offset zu gross?")

        embedder = BGEEmbedder(batch_size=args.batch_size, device=args.device)
        print(f"Embedding {len(query_items)} Query-Texte "
              f"(mit BGE-Instruction-Prefix)...", flush=True)
        # WICHTIG: Query-Instruction nur hier, NICHT im Korpus.
        prefixed = [BGE_QUERY_INSTRUCTION + it["review_text"] for it in query_items]
        queries = embedder.encode(prefixed)

        np.save(qvec_path, queries)
        write_queries_parquet(out_dir, query_items)
        print(f"  -> {qvec_path.name}, {qmeta_path.name}")

    # Ground Truth
    gt_ids_path = out_dir / "ground_truth_ids.npy"
    gt_scores_path = out_dir / "ground_truth_scores.npy"
    gt_parq_path = out_dir / "queries_groundtruth.parquet"

    have_gt = gt_ids_path.exists() and gt_scores_path.exists() and gt_parq_path.exists()
    if have_gt and not args.force_gt:
        # Pruefen ob Korpus juenger ist als das vorhandene GT
        chunk_mtimes = [p.stat().st_mtime for p in args.corpus_dir.glob("chunk_*.parquet")]
        gt_mtime = gt_ids_path.stat().st_mtime
        if chunk_mtimes and max(chunk_mtimes) > gt_mtime:
            print("Ground Truth aelter als Korpus-Chunks — Neuberechnung.")
        else:
            print("Resume: Ground Truth bereits vorhanden, ueberspringe.")
            return

    gt_ids, gt_scores = brute_force_gt(
        queries, args.corpus_dir, args.top_k, args.gt_batch,
    )
    np.save(gt_ids_path, gt_ids)
    np.save(gt_scores_path, gt_scores)
    write_gt_parquet(out_dir, gt_ids, gt_scores)

    print(f"\nOutput in {out_dir}")
    print(f"  queries.npy                  shape={queries.shape}")
    print(f"  ground_truth_ids.npy         shape={gt_ids.shape}")
    print(f"  ground_truth_scores.npy      shape={gt_scores.shape}")
    print(f"  queries_groundtruth.parquet")


if __name__ == "__main__":
    main()
