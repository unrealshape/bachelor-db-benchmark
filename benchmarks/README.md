# benchmarks/

`demodata/` ist der Generator für synthetische Embedding-Datensätze, `runners/` ist die Orchestrierung (Deploy → Run → Collect).

Eigener Python-Runner für alle drei DB-Clients, `ann-benchmarks` nur als Cross-Check.

Pro Lauf werden Latenz (p50/p95/p99), QPS, Recall@1/10/100, NDCG@10, Index-Bauzeit, Index-Größe und CPU/RAM pro Pod gemessen. Output: `results/<run-id>/` mit `raw/`, `summary.json`, `meta.yaml`. Schema-Entwurf in `docs/benchmark-plan.md`.
