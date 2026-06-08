# bachelor-db-benchmark

Pinecone, Weaviate und pgvector im Vergleich für semantische Suche auf Amazon Product Reviews.

## Daten

Amazon Product Reviews (McAuley 2013), Kategorie `Office_Products`. Sample:

```json
{
  "id": 0,
  "product_id": "B01MZ3SD2X",
  "product_title": "",
  "user_id": "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ",
  "rating": 5,
  "review_text": "Lovely ink. Writes well. The right amount of wet/dry. ...",
  "timestamp": "2023-03-04"
}
```

Embeddings: `BAAI/bge-large-en-v1.5`, 1024 dim, L2-normalisiert.

## changelog

### 07.06.2026 — quick-sweep auf 50.000 Reviews

- corpus indexiert: 50.000 Office_Products Reviews, BGE-1024
- queries: 100 held-out Reviews, brute-force Top-100 als ground truth
- spezial-GT: rating>=4 (42.321 von 50.000), hybrid alpha=0.5 (BM25 + Vektor)

topk Variante A, Concurrency 1:

- weaviate (ef=64 M=16 ef_c=128): qps=190.5, p50=4.87ms, recall@10=0.968
- pgvector (m=16 ef_c=128 ef_search=64): qps=313.1, p50=3.20ms, recall@10=0.972

filtered (rating>=4, Filter-GT):

- weaviate: qps=247.9, p50=3.65ms, recall@10=0.970
- pgvector: qps=250.9, p50=3.86ms, recall@10=0.932

batch (Concurrency 8):

- weaviate: qps=862.8, p50=7.79ms, recall@10=0.969
- pgvector: qps=264.7, p50=29.76ms, recall@10=0.971

hybrid (alpha=0.5, Hybrid-GT):

- weaviate: qps=87.1, p50=8.52ms, recall@10=0.267
- pgvector: qps=116.5, p50=8.19ms, recall@10=0.156

Variante B (Embedding + Metadaten getrennt):

- weaviate: qps=262.4, p50=3.50ms, recall@10=0.967
- pgvector: qps=269.2, p50=3.68ms, recall@10=0.971

concurrency-sweep topk:

- weaviate c=4: qps=808.6, p50=4.04ms, recall@10=0.969
- weaviate c=16: qps=797.4, p50=18.74ms, recall@10=0.969
- pgvector c=4: qps=228.6, p50=17.17ms, recall@10=0.972
- pgvector c=16: qps=259.1, p50=60.66ms, recall@10=0.970

tuning weaviate HNSW (ef / ef_construction / M):

- notune 10/32/8: qps=324.1, p50=2.96ms, recall@10=0.870
- tune-low 32/64/8: qps=321.9, p50=2.99ms, recall@10=0.924
- tune-high 128/256/32: qps=225.2, p50=4.17ms, recall@10=0.990
- tune-max 256/512/48: qps=144.5, p50=6.50ms, recall@10=0.995

tuning pgvector HNSW (m / ef_construction / ef_search):

- notune 8/32/10: qps=479.2, p50=1.83ms, recall@10=0.578, build=9s
- tune-low 8/64/32: qps=308.9, p50=3.11ms, recall@10=0.843, build=14s
- tune-high 32/256/128: qps=102.2, p50=9.72ms, recall@10=0.995, build=94s
- tune-max 48/512/256: qps=61.7, p50=15.96ms, recall@10=0.999, build=192s
