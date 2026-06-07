# Nächster Run: HNSW-Tuning-Leiter auf Stufe S

Tuning-Sweep für Kapitel 7.7. Vier Mess-Läufe auf demselben Korpus (Stufe S, ca. 10 GB Embedding-Volumen) und denselben Queries — nur die HNSW-Parameter ändern sich. Workload ist `topk` mit Concurrency 1, damit der Vergleich sauber an den Index-Parametern hängt.

## Voraussetzungen

Vor dem ersten echten Lauf:

1. **Amazon Product Reviews auf Stufe S laden und Embeddings rechnen.** Das ist die Datenbasis aus Thesis 5.1.1. Solange die Pipeline fehlt, läuft die Leiter nur auf Synthese-Daten — die Ergebnisse sind dann **Pipeline-Smoke, kein Thesis-Material**.
2. Embedding-Modell ist `BAAI/bge-large-en-v1.5` mit nativer Dimension 1024, L2-normalisiert, lokal via `sentence-transformers` (Thesis 5.2). `dim` steht in allen Configs auf 1024. Query-Prefix: `"Represent this sentence for searching relevant passages: "`. Modell-Cache wird unter `~/.cache/huggingface/hub/` erwartet und in den Pod gemountet, damit das Modell nicht je Lauf neu geladen wird.
3. Cluster läuft, Weaviate ist deployed (`./setup.sh`).
4. DB-Pod vor jedem Lauf neu starten und OS-Cache leeren (Thesis 5.5).

Korpus + Queries vorbereiten:

```bash
# (echt) Amazon-Reviews-Loader sobald implementiert
# (smoke) Synthese-Daten:
python benchmarks/demodata/generate.py --output-dir ~/.cache/bachelor-db-benchmark/S --size S --dim 1024
python benchmarks/demodata/gen_queries.py --corpus-dir ~/.cache/bachelor-db-benchmark/S --output-dir ~/.cache/bachelor-db-benchmark/S/queries --dim 1024 --n-queries 1000
```

## Reihenfolge

| # | Config | ef | ef_construction | M | Zweck |
|---|--------|----|-----------------|---|-------|
| 1 | `weaviate-S-notune` | 10 | 32 | 8 | Referenz ohne Tuning |
| 2 | `weaviate-S-tune` | 64 | 128 | 16 | Standard-Tuning |
| 3 | `weaviate-S-tune-extended` | 128 | 256 | 32 | erweitertes Tuning |
| 4 | `weaviate-S-tune-max` | 256 | 512 | 48 | Maximaler Recall |

## Aufrufen

```bash
./benchmarks/runners/runner.py --config weaviate-S-notune
./benchmarks/runners/runner.py --config weaviate-S-tune
./benchmarks/runners/runner.py --config weaviate-S-tune-extended
./benchmarks/runners/runner.py --config weaviate-S-tune-max
```

Optional `--push` an den letzten Aufruf hängen, dann landen alle vier auf einmal im public Repo und das Dashboard zeigt sie zusammen.

## Erwartung

Über die vier Läufe sollte eine monoton steigende Kurve von Recall@10 rauskommen (von deutlich unter 0,9 bei `notune` bis nahe 1,0 bei `tune-max`), gleichzeitig mit steigender Index-Bauzeit und Query-Latenz. Das ist die klassische HNSW Recall-vs-Performance-Trade-off-Kurve und gehört in Kapitel 7.7. Latenz wird als Mittelwert, Median (p50), p95 und p99 berichtet (Thesis 5.5.3).

## Danach

Sobald die Reviews-Pipeline steht: gleiche Leiter analog für pgvector HNSW und pgvector IVFFlat (Thesis 4.3 und 7.6). Pinecone wird nicht getuned — der Tier `s1.x1` ist die einzige Stellschraube.
