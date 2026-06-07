# Cluster Setup

`./setup.sh` macht alles: Tools prüfen, k3d-Cluster anlegen, Weaviate und pgvector deployen, optional Smoke-Test.

Überschreiben geht per Env-Vars: `CLUSTER_NAME` (default `dbbench`), `AGENTS` (default 3), `K3S_IMAGE` (default `rancher/k3s:v1.31.4-k3s1`).

Check ob's läuft:

```bash
kubectl get nodes
kubectl get pods -A
```

Cluster wieder weg: `k3d cluster delete dbbench`.

## Wenn was klemmt

- Docker läuft nicht → Docker Desktop oder OrbStack starten
- Pod bleibt `Pending` → Ressourcen-Limit im Docker hochziehen
- Port 80/443 belegt → das andere Programm beenden oder Ports im Script anpassen
