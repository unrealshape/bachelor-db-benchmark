# Demodata-Generator

**Nur für Pipeline-Smokes auf Stufe T / T2.** Offizielle Thesis-Läufe nutzen Amazon Product Reviews mit MiniLM-Embeddings, nicht die Synthese-Daten hier.

Vier Thesis-Stufen, plus zwei Dev-Stufen:

| Stufe | Records   | On Disk (zstd) | Zweck |
|-------|-----------|----------------|-------|
| T     | 20.000    | ~100 MB        | Pipeline-Smoke |
| T2    | 100.000   | ~560 MB        | Stabilisierungs-Run |
| S     | 100.000   | ~0,5 GB        | Thesis Stufe 1 |
| M     | 500.000   | ~2,5 GB        | Thesis Stufe 2 |
| L     | 1.000.000 | ~5 GB          | Thesis Stufe 3 |
| XL    | 5.000.000 | ~25 GB         | Thesis Stufe 4 |

Format: Parquet-Chunks à 500.000 Records mit `id` (int64) und `embedding` (FixedSizeList<float32>[dim]). Vektoren aus Standard-Normalverteilung, auf Einheits-Sphäre normalisiert. Deterministisch über `--seed`. Default-Dimension 384 (MiniLM).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 oder neuer.

## Daten erzeugen

Eine Stufe:

```bash
python generate.py --output-dir ./out/S --size S
```

Alle: `make all OUT=./out`.

Eigene Größe geht auch:

```bash
python generate.py --output-dir ./out/custom --n-records 250000 --dim 384
```

## Queries und Ground-Truth

Für Recall@k über Brute-Force k-NN:

```bash
make queries STUFE=S OUT=./out
```

Output unter `out/<Stufe>/queries/`: `queries.npy`, `ground_truth_ids.npy`, `ground_truth_scores.npy`. Default 1.000 Queries (Thesis-Minimum), Top 100.

Grobe CPU-Laufzeiten bei 384-dim: S Sekunden, M ~1 min, L ~5 min, XL ~30 min. Für XL auf den echten Reviews-Daten lohnt sich GPU.

Daten nicht ins Repo. Lokal nach `~/.cache/bachelor-db-benchmark/`, im Cluster auf PVC.
