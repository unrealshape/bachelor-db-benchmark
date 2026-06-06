# bachelor-db-benchmark

Oeffentliches Begleit-Repo zu meiner Bachelorarbeit an der Hochschule Darmstadt. Hier landen die saubere Doku, die Konfigurationsdateien zum Reproduzieren der Tests und die Ergebnisse der Benchmark-Laeufe.

> **Hinweis:** Dieses Repo wird automatisch befuellt -- direkt manuell editieren bringt nichts, beim naechsten Sync wird's wieder ueberschrieben. Quelle ist mein privates Arbeitsrepo, gesynct ueber eine GitHub Action.

## Stand

Im Moment noch leer bzw. nur Skelett. Sobald die ersten Benchmark-Laeufe sauber durch sind, wandern die Resultate hier rein.

Erwartete Struktur (Auszug):

```
.
├── docs/                # Anleitungen und Erklaerungen
├── infrastructure/      # Cluster-Setup, Manifeste, Helm-Werte
├── databases/           # DB-Konfigurationen
├── benchmarks/          # Workloads und Test-Szenarien
└── results/             # Roh- und aggregierte Ergebnisse
```

## Dashboard

Visualisierung der Ergebnisse: [bachelor-db-benchmark-dashboard](https://github.com/unrealshape/bachelor-db-benchmark-dashboard). Das Dashboard liest die Daten aus diesem Repo.

## Reproduzieren

Sobald `docs/setup-cluster.md` und `REQUIREMENTS.md` hier liegen, ist das die Schritt-fuer-Schritt-Anleitung.
