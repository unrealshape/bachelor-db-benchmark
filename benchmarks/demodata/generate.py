#!/usr/bin/env python3
"""
Erzeugt synthetische Embedding-Datensaetze fuer den Benchmark.

Output: Parquet-Chunks im --output-dir.
Schema: id (int64), embedding (FixedSizeList<float32>[dim])

Vektoren sind aus einer Standard-Normalverteilung gezogen und auf die
Einheits-Sphaere normalisiert -- damit ist Cosine-Aehnlichkeit aequivalent
zum inneren Produkt. Mit festem Seed reproduzierbar.

Beispiel:
    python generate.py --output-dir ./out/S --size S
    python generate.py --output-dir ./out/custom --n-records 1000000 --dim 768
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Records pro Stufe -- Thesis 5.1.3 (Amazon Product Reviews, bge-large-en-v1.5 1024-dim).
# Synthese-Daten hier sind nur fuer Pipeline-Tests; offizielle Mess-Laeufe
# nutzen den realen Datensatz.
SIZE_PRESETS = {
    "S":    100_000,   # Thesis Stufe 1, ~0.5 GB
    "M":    500_000,   # Thesis Stufe 2, ~2.5 GB
    "L":  1_000_000,   # Thesis Stufe 3, ~5 GB
    "XL": 5_000_000,   # Thesis Stufe 4, ~25 GB
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, type=Path)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--size", choices=list(SIZE_PRESETS),
                     help="Preset-Groesse (S/M/L/XL)")
    grp.add_argument("--n-records", type=int,
                     help="explizite Anzahl Datensaetze")
    p.add_argument("--dim", type=int, default=1024,
                   help="Embedding-Dimension (default 1024, BAAI/bge-large-en-v1.5)")
    p.add_argument("--chunk-records", type=int, default=500_000,
                   help="Records pro Parquet-Chunk")
    p.add_argument("--seed", type=int, default=42,
                   help="Random-Seed fuer Reproduzierbarkeit")
    p.add_argument("--compression", default="zstd",
                   choices=["zstd", "snappy", "none"],
                   help="Parquet-Compression (zstd ist Standard)")
    return p.parse_args()


def gen_chunk(rng, n, dim, id_start):
    """Erzeugt n normalisierte Embeddings + zugehoerige IDs."""
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    np.divide(vecs, np.where(norms == 0, 1.0, norms), out=vecs)
    ids = np.arange(id_start, id_start + n, dtype=np.int64)
    return ids, vecs


def write_chunk(path, ids, vecs, compression):
    """Schreibt einen Chunk als Parquet."""
    n, dim = vecs.shape
    # Flach + FixedSizeList -- so liest der Reader effizient zurueck.
    flat = pa.array(vecs.reshape(-1), type=pa.float32())
    embedding = pa.FixedSizeListArray.from_arrays(flat, dim)
    table = pa.Table.from_arrays(
        [pa.array(ids, type=pa.int64()), embedding],
        names=["id", "embedding"],
    )
    comp = None if compression == "none" else compression
    pq.write_table(table, path, compression=comp)


def main():
    args = parse_args()
    n_total = args.n_records if args.n_records else SIZE_PRESETS[args.size]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    chunks = (n_total + args.chunk_records - 1) // args.chunk_records

    print(f"Ziel: {n_total:,} Vektoren, {args.dim}-dim, "
          f"{chunks} Chunks, seed={args.seed}", flush=True)

    t0 = time.time()
    for i in range(chunks):
        start = i * args.chunk_records
        end = min(start + args.chunk_records, n_total)
        ids, vecs = gen_chunk(rng, end - start, args.dim, start)
        path = out / f"chunk_{i:04d}.parquet"
        write_chunk(path, ids, vecs, args.compression)
        rate = end / max(time.time() - t0, 1e-6)
        print(f"  [{i+1:>3}/{chunks}]  {end:>12,}/{n_total:,}  "
              f"{rate:>10,.0f} vec/s", flush=True)

    total_time = time.time() - t0
    total_bytes = sum(p.stat().st_size for p in out.glob("chunk_*.parquet"))
    print(f"\nFertig in {total_time:.1f}s")
    print(f"Auf Disk: {total_bytes / 2**30:.2f} GiB  ({total_bytes:,} bytes)")


if __name__ == "__main__":
    main()
