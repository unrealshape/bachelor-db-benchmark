# Weaviate

Self-hosted über das offizielle Helm Chart, Image gepinnt auf 1.37.6. HNSW mit Cosine, 1 Replica, 50 GB PVC auf `local-path`, 2 vCPU / 8 GB Limit (passt zu Pinecone `s1.x1`). Vektorizer aus — Embeddings kommen vom Benchmark, kein verstecktes Modell im Weg.

## Deploy

```bash
helm upgrade --install weaviate weaviate/weaviate \
  -n db-weaviate --create-namespace \
  -f databases/weaviate/values.yaml --wait
```

Macht `./setup.sh` automatisch.

## Smoke-Test

```bash
kubectl port-forward -n db-weaviate svc/weaviate 8080:80 &
curl -s http://localhost:8080/v1/.well-known/ready -o /dev/null -w "%{http_code}\n"
curl -s http://localhost:8080/v1/meta | jq '.version'
```

## Klasse für den Benchmark

```json
{
  "class": "Doc",
  "vectorizer": "none",
  "vectorIndexConfig": {
    "distance": "cosine",
    "ef": 64,
    "efConstruction": 128,
    "maxConnections": 16
  },
  "properties": [
    {"name": "doc_id", "dataType": ["int"]},
    {"name": "product_id", "dataType": ["text"]},
    {"name": "rating", "dataType": ["int"]},
    {"name": "review_text", "dataType": ["text"]}
  ]
}
```

Wieder weg:

```bash
helm uninstall weaviate -n db-weaviate
kubectl delete ns db-weaviate
```
