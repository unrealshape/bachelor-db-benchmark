# Test-Konfigurationen

Jede `.json` hier ist ein Mess-Lauf — welche DB, welche Stufe, welcher Workload, welche Index-Parameter, welche Datenmodellierungs-Variante. Der Runner nimmt die Config und hängt einen Timestamp dran, Output landet unter `results/<timestamp>_<config-name>/`.

Stufen und Methodik liegen in der Thesis fest (siehe `CLAUDE.md` und `docs/benchmark-plan.md`). Neue Configs müssen dazu passen.

```bash
./run.sh --config weaviate-S-tune --dummy
./run.sh --config weaviate-S-tune --push
```

`--dummy` schreibt plausible Beispiel-Metriken (für Pipeline-Tests, kein echter Lauf). `--push` committet und pusht — die Sync-Action propagiert ins public Repo, das Dashboard sieht's beim nächsten Refresh.

## Schema

```json
{
  "name": "weaviate-S-tune",
  "description": "kurze Beschreibung",
  "db": "weaviate | pgvector | pinecone",
  "stufe": "T | T2 | S | M | L | XL | XXL",
  "dim": 1024,
  "variant": "A | B",
  "workload": "topk | filtered | batch | hybrid",
  "index": {
    "type": "hnsw | ivfflat",
    "params": {"ef": 64, "ef_construction": 128, "M": 16}
  },
  "queries": {"n": 1000, "concurrency": 1}
}
```

`dim` ist Pflicht und steht auf 1024 (`BAAI/bge-large-en-v1.5` in nativer Dimension, L2-normalisiert, lokal via `sentence-transformers`, Thesis 5.2). `variant` trennt Variante A (Embedding + Metadaten inline) und B (Embedding und Metadaten getrennt, per ID verknüpft) aus Thesis 5.3. `params` ist je nach Index-Typ unterschiedlich: HNSW nutzt `ef`, `ef_construction`, `M` — IVFFlat nutzt `lists`, `probes`. Bei Durchsatz-Läufen ist `queries.concurrency` einer aus 1, 4, 8, 16 (Thesis 5.5.4).

Die Stufen S / M / L / XL / XXL entsprechen den Embedding-Volumen 10 / 20 / 40 / 80 / 100 GB (Thesis 5.1.3). T und T2 sind Dev-Stufen mit Synthese-Daten für Pipeline-Smoke und gehen nicht in die Thesis ein.
