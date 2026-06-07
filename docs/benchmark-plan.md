# Benchmark-Plan

Kurzfassung. Details in Kapitel 5 der Thesis. Plan ist gegenüber dem Exposé angepasst, Begründung steht in §5.1.

## Frage

Welches der drei Systeme — Pinecone, Weaviate, pgvector — taugt am besten für semantische Suche, wenn Qualität, Latenz und Skalierung zusammen reinkommen?

## Daten

Amazon Product Reviews (McAuley 2013). Schema: `id`, `product_id`, `product_title`, `user_id`, `rating`, `review_text`, `timestamp`, `embedding`. Synthese unter `benchmarks/demodata/` ist nur Pipeline-Smoke (T/T2), kein Thesis-Material.

## Stufen

Am Embedding-Volumen gemessen, nicht an der Doc-Zahl. Vom „passt komplett in RAM" bis dahin wo der Index drüber raus wächst.

| Label | Volumen | Vektoren bei 1024 dim |
|-------|---------|-----------------------|
| S     | 10 GB   | ~2,4 Mio. |
| M     | 20 GB   | ~4,9 Mio. |
| L     | 40 GB   | ~9,8 Mio. |
| XL    | 80 GB   | ~19,5 Mio. |
| XXL   | 100 GB  | ~24,4 Mio. |

Pro Stufe min. 1.000 Queries aus der gleichen Verteilung, Top-100 Brute-Force als Ground Truth.

## Embeddings

`BAAI/bge-large-en-v1.5`, 1024 dim, L2-normalisiert, lokal über `sentence-transformers` (MIT). Queries mit BGE-Prefix `"Represent this sentence for searching relevant passages: "`, Docs ohne.

Ein Modell. Die Exposé-Linien (MiniLM 384, OpenAI 768) sind nach den Pilots raus. Ein Zwischenstand mit OpenAI 1536 dim auch — zu teuer und noch eine Cloud neben Pinecone. 1024 stresst die Indizes ähnlich hart wie 1536. Cache liegt unter `~/.cache/huggingface/hub/`, im Pod als hostPath gemountet.

## Modellierung, Queries, Umgebung

A: Embedding + Metadaten zusammen. B: getrennt, per `id` verknüpft. Beide pro DB und Stufe.

Queries: `topk`, `filtered` (z. B. `rating >= 4`), `batch`, `hybrid` (BM25 + Vektor).

k3d auf der Workstation. Weaviate und pgvector je 2 vCPU / 8 GB Limit, 50 GB PVC. Pinecone im Pod-Tier `s1.x1` — bewusst kein Serverless, hardware-äquivalent. Pinecone geht übers Netz, deshalb Wall-Clock auf Client plus Server-Zeit aus dem Header `x-pinecone-request-latency-ms`. Netz wird am Ende rausgerechnet.

## Metriken

Index-Bauzeit, Latenz p50/p95/p99 (min. 1.000 Anfragen), QPS bei 1/4/8/16 Threads, Recall@1/10/100, Precision@k, NDCG@10, Index-Größe auf Disk, CPU/RAM pro Pod (bei Pinecone nicht).

## Ablauf

Cluster sauber → DB-Pod neu starten + OS-Cache leeren → Deployen → Importieren + Index bauen → 1.000 Warmup-Queries wegwerfen → Mess-Queries → Recall + Metriken einsammeln → Cleanup.

Output: `results/<run-id>/summary.json` + `config.json`, optional `raw/`.

## Tuning

Sweeps laufen auf echten Reviews auf Stufe S. `weaviate-S-notune` / `-tune` / `-tune-extended` / `-tune-max` als Vorlage, analog für pgvector HNSW und IVFFlat. IVFFlat bei pgvector ist Pflicht (Thesis 4.3). Pinecone tunen wir nicht — der Tier ist die einzige Stellschraube.

## Reproduzierbarkeit

Run-ID eindeutig, Image-Tags und Index-Parameter gepinnt, Demodata mit Seed, Pinecone-Konfig im Code.

## Offen

Adapter um Variante B und `filtered` / `batch` / `hybrid` ergänzen. Pre-Run-Hook für Cache-Drop und Pod-Restart automatisieren. Pinecone-Account + API-Key.
