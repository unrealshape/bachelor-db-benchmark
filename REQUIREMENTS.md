# Voraussetzungen

Mindestens 4 CPU-Kerne, 16 GB RAM, 30 GB Disk. macOS oder Linux, Windows mit WSL2 nicht getestet.

Tools: `git`, `docker` (muss laufen), `kubectl`, `k3d`, `helm`. Optional `k9s` und `psql`. Fehlt was, fragt `setup.sh` ob er das per Homebrew nachzieht.

Für die Mess-Läufe braucht's einen Pinecone-Account, fürs lokale Entwickeln nicht. SSH-Key bei GitHub für die drei Repos sollte da sein.

Schneller Check:

```bash
docker info >/dev/null && echo ok
kubectl version --client
k3d version
helm version --short
```
