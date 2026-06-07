# reviewdata/

Loader für den Amazon Product Reviews-Datensatz (McAuley & Leskovec 2013) — die echte Datenbasis der Thesis-Mess-Läufe. Im Gegensatz zu `benchmarks/demodata/` (synthetische Vektoren, nur Pipeline-Smoke auf T/T2) entsteht hier der Korpus, der in Kapitel 6/7 ausgewertet wird.

## Quelle

[`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) auf HuggingFace. Direkt-Download der `raw/review_categories/*.jsonl`-Dateien (kein Loading-Script, kein `trust_remote_code` — die `datasets`-Library blockiert das ab 4.x). Felder mappen 1:1 ins Thesis-Schema 5.1.

Default-Kategorien: `Home_and_Kitchen, Clothing_Shoes_and_Jewelry, Electronics, Books, Tools_and_Home_Improvement, …` — zusammen genug Volumen für alle Stufen bis 100 GB Embeddings.

## Embedding

[`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5), **1024 dim, L2-normalisiert**. MIT-Lizenz, MTEB-State-of-the-Art im Open-Source-Bereich, läuft lokal via `sentence-transformers`.

**Kein API-Key nötig.** Beim ersten Lauf zieht `sentence-transformers` das Modell (~1,3 GB) nach `~/.cache/huggingface/`. Danach läuft alles offline.

Passages (Korpus-Texte) werden ohne Prefix embedded. Queries bekommen die BGE-spezifische Instruction `"Represent this sentence for searching relevant passages: "` vorangestellt — so will es das BGE-Paper, und das macht spürbaren Recall-Unterschied.

Device-Wahl automatisch: CUDA > MPS (Apple Silicon) > CPU. Override via `--device`. Batch-Size via `BENCH_EMBED_BATCH` (Default 64 auf CPU, 256 auf GPU/MPS) oder `--batch-size`.

## Stufen

Gemessen am reinen Embedding-Volumen (1024 x float32 = 4096 Byte pro Vektor):

| Stufe | Ziel-GB | ca. Reviews   |
|-------|---------|---------------|
| S     | 10 GB   | ~2,62 Mio.    |
| M     | 20 GB   | ~5,24 Mio.    |
| L     | 40 GB   | ~10,49 Mio.   |
| XL    | 80 GB   | ~20,97 Mio.   |
| XXL   | 100 GB  | ~26,21 Mio.   |

Plattenplatz auf Disk inkl. Metadaten und Parquet-zstd ~40 % höher.

## Setup

```bash
pip install -r requirements.txt
```

Bei `torch>=2.0` ohne explizite CUDA-Wheels wird automatisch die CPU-Variante installiert; für GPU bitte das passende Wheel von [pytorch.org](https://pytorch.org/get-started/locally/) wählen.

Cache-Verzeichnis für den Korpus: `$BENCH_CACHE_DIR` oder Default `~/.cache/bachelor-db-benchmark/reviewdata/`. Modell-Cache liegt unter `~/.cache/huggingface/`.

## Korpus erzeugen

Dry-Run (kein Modell-Download, zeigt Plan + geschätztes Volumen):

```bash
python load.py --stage S --dry-run
```

Echter Lauf:

```bash
python load.py --stage S
```

Argumente:

- `--stage S|M|L|XL|XXL` — Pflicht.
- `--categories ...` — eigene Kategorien-Auswahl (Default: 15 große Kategorien).
- `--batch-size` — Forward-Pass-Batch (Default aus `BENCH_EMBED_BATCH`).
- `--device cuda|mps|cpu` — Erzwingt ein Device (Default auto).
- `--chunk-records 50000` — Reviews pro Parquet-Chunk.
- `--max-chars 2000` — Trunkierung vor Embedding (BGE-Large: 512 Token Limit).
- `--output-dir` — überschreibt Default-Pfad.

**Resumable.** Bei Abbruch einfach denselben Befehl nochmal aufrufen: der Loader liest `.progress.json` ein und embedded nur die fehlenden Reviews.

**Inkompatibler Cache.** Liegt in einem Stufen-Verzeichnis schon ein Korpus mit anderer Embedding-Dimension (etwa 1536 dim aus einer früheren OpenAI-Variante), bricht der Loader ab statt zu überschreiben. Verzeichnis manuell umbenennen oder löschen.

## Queries + Ground Truth

```bash
python gen_queries.py --corpus-dir ~/.cache/bachelor-db-benchmark/reviewdata/S
```

Default: 1.000 Queries (Thesis-Minimum), Top-100 Brute-Force-Ground-Truth via Cosine.

Queries kommen aus einer **Held-Out-Partition** der gleichen Kategorien (ab Zeile `--query-offset`, default 10 Mio. = sicher disjunkt vom Korpus). Damit kommen Queries aus derselben Verteilung wie der Korpus, ohne Identität — Recall@1 wird nicht trivial.

Die Query-Texte werden vor dem Embedden mit der BGE-Instruction prefixed; Korpus-Passages bekommen sie nicht. Wer `queries.npy` aus dem Cache liest und parallel den Korpus tauscht, sollte beide regenerieren.

Ebenfalls resumable: vorhandene `queries.npy` / `ground_truth_*.npy` werden übersprungen. Mit `--force-gt` erzwingt man eine Neuberechnung.

## Output-Schema

Pro Chunk (`chunk_NNNN.parquet`):

| Spalte         | Typ                          |
|----------------|------------------------------|
| `id`           | int64                        |
| `product_id`   | string (parent_asin)         |
| `product_title`| string                       |
| `user_id`      | string                       |
| `rating`       | int8                         |
| `review_text`  | string                       |
| `timestamp`    | string (ISO-Date YYYY-MM-DD) |
| `embedding`    | FixedSizeList<float32>[1024] |

Plus `corpus_meta.json` mit Stufe, Modell, Anzahl, Kategorien, Schema.

Queries-Verzeichnis (`<corpus-dir>/queries/`):

- `queries.parquet` — id, product_id, user_id, rating, review_text, category
- `queries.npy` — (Q, 1024) float32, L2-normalisiert
- `ground_truth_ids.npy` — (Q, 100) int64
- `ground_truth_scores.npy` — (Q, 100) float32
- `queries_groundtruth.parquet` — query_id, gt_ids (list<int64>), gt_scores (list<float32>)

## Daten nicht ins Repo

Cache liegt lokal (`~/.cache/bachelor-db-benchmark/`), im Cluster auf PVC. Modell genauso (`~/.cache/huggingface/`).
