# databases/

Pro DB ein Ordner mit allem zum Deployen: `weaviate/` (Helm values), `pgvector/` (Manifeste), `pinecone/` (Cloud-Konfig + Client).

Für alle drei das selbe Setup: eigener Namespace (`db-weaviate`, `db-pgvector`, `db-pinecone`), 2 vCPU / 8 GB Limit pro Pod (vergleichbar mit Pinecone `s1.x1`), Image-Tag immer gepinnt, 50 GB PVC auf `local-path`. Index-Parameter stehen im Schema, nicht im Code.
