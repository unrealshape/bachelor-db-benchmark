# PostgreSQL mit pgvector

Self-hosted über pure Manifeste, kein Helm. Postgres 17 mit pgvector 0.8 (`pgvector/pgvector:0.8.0-pg17`). HNSW als Hauptindex, IVFFlat zum Vergleich, Cosine-Distanz. 2 vCPU / 8 GB Limit (passt zu Pinecone `s1.x1`), StatefulSet mit 50 GB PVC. Tunings (`shared_buffers=2GB`, `maintenance_work_mem=1GB`, parallel workers) liegen in `manifests/02-init-sql.yaml`.

## Deploy

```bash
kubectl apply -f databases/pgvector/manifests/
kubectl rollout status sts/pgvector -n db-pgvector --timeout=5m
```

Macht `./setup.sh` automatisch.

## Smoke-Test

```bash
kubectl exec -n db-pgvector pgvector-0 -- \
  psql -U bench -d benchmark -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

## Benchmark-Schema

```sql
CREATE TABLE docs (
  doc_id    bigint PRIMARY KEY,
  embedding vector(384) NOT NULL
);

CREATE INDEX docs_hnsw_idx
  ON docs USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);
```

IVFFlat-Variante:

```sql
CREATE INDEX docs_ivf_idx
  ON docs USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 1000);
```

Wieder weg: `kubectl delete ns db-pgvector`.
