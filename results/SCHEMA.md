# Results-Schema

Pro Mess-Lauf ein Verzeichnis `results/<run-id>/`, Index über alle Läufe in `results/index.json`. Latenz-Felder und Stufen folgen der Thesis-Methodik (Kapitel 5.5).

Run-ID-Format: `<ISO-Timestamp>_<config-name>`, z. B. `2026-06-07T10-30-15Z_weaviate-S-tune`. Doppelpunkte im Zeitstempel sind durch Bindestriche ersetzt.

## summary.json

```json
{
  "run_id": "2026-06-07T10-30-15Z_weaviate-S-tune",
  "config_name": "weaviate-S-tune",
  "spec_version": "1024-bge-v1",
  "started_at": "2026-06-07T10:30:15Z",
  "finished_at": "2026-06-07T10:34:42Z",
  "duration_s": 267,
  "status": "ok",
  "db": {
    "name": "weaviate",
    "version": "1.37.6",
    "image": "semitechnologies/weaviate:1.37.6"
  },
  "dataset": {
    "size_label": "S",
    "n_vectors": 2400000,
    "dim": 1024,
    "variant": "A",
    "size_gb": 10.0
  },
  "index": {
    "type": "hnsw",
    "params": {"ef": 64, "ef_construction": 128, "M": 16},
    "build_time_s": 145,
    "size_on_disk_mb": 320
  },
  "workload": {
    "profile": "topk",
    "n_queries": 1000,
    "concurrency": 1
  },
  "metrics": {
    "throughput_qps": 412.3,
    "latency_ms_mean": 2.4,
    "latency_ms_p50": 2.1,
    "latency_ms_p95": 6.7,
    "latency_ms_p99": 8.2,
    "recall_at_1": 0.98,
    "recall_at_10": 0.95,
    "recall_at_100": 0.91,
    "precision_at_10": 0.95,
    "ndcg_at_10": 0.93
  },
  "resources": {
    "cpu_avg_cores": 1.4,
    "mem_avg_mb": 4280
  },
  "cluster": {
    "k8s_version": "v1.31.4+k3s1",
    "nodes": 4
  },
  "notes": {}
}
```

Felder dürfen `null` sein wenn nicht erhoben (Pinecone z. B. hat keine Resource-Metriken). `status` ist `ok`, `partial` oder `failed`. `dataset.variant` ist `"A"` oder `"B"` gemäß Thesis 5.3, `workload.profile` einer aus `topk | filtered | batch | hybrid`.

`spec_version` markiert den Methodik-Stand des Runs. Aktuell `1024-bge-v1` (1024 dim `BAAI/bge-large-en-v1.5` lokal, S/M/L/XL/XXL = 10/20/40/80/100 GB, p50/p95/p99, k3d, Pinecone `s1.x1`). Ältere Runs aus der Pre-Spec-Phase tragen `pre-1536` und sind kein Thesis-Material.

## index.json

```json
{
  "generated_at": "2026-06-07T10:35:00Z",
  "n_runs": 1,
  "runs": [
    {
      "id": "2026-06-07T10-30-15Z_weaviate-S-tune",
      "config_name": "weaviate-S-tune",
      "spec_version": "1024-bge-v1",
      "db": "weaviate",
      "stufe": "S",
      "workload": "topk",
      "status": "ok",
      "started_at": "2026-06-07T10:30:15Z",
      "throughput_qps": 412.3,
      "latency_ms_p50": 2.1,
      "latency_ms_p95": 6.7,
      "latency_ms_p99": 8.2,
      "recall_at_10": 0.95
    }
  ]
}
```

Vom Runner nach jedem Lauf neu geschrieben (kompletter Scan über `results/`).

## raw/

Optional pro Run: `results/<run-id>/raw/` mit Roh-Logs (Latenzen pro Query, Pod-Stats). Vom Dashboard nicht gerendert, aber für Reproduzierbarkeit wichtig.
