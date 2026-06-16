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

## Windows-Setup (WSL2 + CUDA-GPU)

Empfohlen wenn eine NVIDIA-GPU da ist — Embedding läuft 10-20× schneller als auf macOS MPS.

Voraussetzungen: Windows 11, WSL2, Docker Desktop mit WSL2-Backend, NVIDIA-Treiber + CUDA-Toolkit (für die GPU sichtbar in WSL2).

```bash
# In WSL2 (Ubuntu)
sudo apt update && sudo apt install -y git python3.12 python3.12-venv build-essential
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Tools
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/
# Helm: apt-key ist auf Ubuntu 24.04 raus, Skript-Install ist der einfachste Weg
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

`usermod -aG docker` greift erst nach neuem Login — in derselben Session `newgrp docker` oder kurz aus-/einloggen, sonst „permission denied" am Docker-Socket.

Repo + Python-Env:

```bash
git clone git@github.com:unrealshape/bachelor-db-benchmark-nopublic.git
cd bachelor-db-benchmark-nopublic
python3.12 -m venv .venv
.venv/bin/pip install -r benchmarks/runners/requirements.txt
.venv/bin/pip install -r benchmarks/reviewdata/requirements.txt
# CUDA-Wheel ueberschreiben (cu128 fuer RTX 5090, sonst cu124 fuer Ampere/Ada)
.venv/bin/pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

Smoke-Check GPU:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Cluster + Embedding:

```bash
./setup.sh
# Loader nutzt CUDA automatisch wenn verfuegbar
BENCH_EMBED_BATCH=512 .venv/bin/python benchmarks/reviewdata/load.py \
    --stage S --categories Office_Products --no-meta \
    --output-dir ~/.cache/bachelor-db-benchmark/S
.venv/bin/python benchmarks/reviewdata/gen_queries.py \
    --corpus-dir ~/.cache/bachelor-db-benchmark/S --n-queries 1000 --top-k 100
.venv/bin/python benchmarks/reviewdata/gen_special_gt.py \
    --corpus-dir ~/.cache/bachelor-db-benchmark/S --filter rating_gte=4
.venv/bin/python benchmarks/reviewdata/gen_special_gt.py \
    --corpus-dir ~/.cache/bachelor-db-benchmark/S --hybrid-alpha 0.5
```

Auf RTX 5090: Stufe S (2,6M Reviews) ist in 20-40 Minuten durch, statt 3-5 h auf macOS MPS.

Hinweis: die Embeddings vom Mac (MPS) und Windows (CUDA) sind nicht bit-identisch — sehr kleine Float-Differenzen. Fuer die Thesis-Reproduzierbarkeit auf einer Maschine bleiben.
