# Pinecone

Managed Cloud, kein Self-Hosting. Konfig so dass sie zur lokalen Basis passt: Pod-Tier `s1.x1` (~2 vCPU / ~8 GB), erstmal 1 Pod (Stufe XL ggf. mehr), AWS in der Region nahe der Workstation. Dimension 384 (MiniLM, Thesis 5.2.1), Cosine, HNSW intern. Bewusst Pod-Modell statt Serverless, weil Serverless die Hardware-Klasse versteckt.

Netz-Latenz: Wall-Clock (mit Netz) und Server-Zeit aus dem Header `x-pinecone-request-latency-ms` (ohne Netz) landen im `summary.json` getrennt.

## Setup (sobald der Account da ist)

```bash
kubectl create ns db-pinecone
kubectl create secret generic pinecone-api \
  --from-literal=api-key="<DEIN-KEY>" -n db-pinecone
```

Index per Script anlegen (`client/create_index.py` kommt noch):

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.create_index(
    name="bachelor-bench",
    dimension=384,
    metric="cosine",
    spec=PodSpec(environment="us-east-1-aws", pod_type="s1.x1", pods=1),
)
```

Geplante Struktur:

```
databases/pinecone/
├── README.md
├── client/{create_index.py, smoke.py, requirements.txt, Dockerfile}
└── manifests/client-job.yaml
```

Wieder weg: `kubectl delete ns db-pinecone` und Index in der Pinecone-Konsole löschen — sonst läuft die Rechnung weiter.
