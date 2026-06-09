#!/usr/bin/env python3
"""Live-Status für eine laufende Benchmark-Batch. Schätzt Fortschritt in
Prozent anhand:
- Anzahl bereits geschriebener summary.json-Dateien (fertige Configs)
- aktuell aktiver Python-Runner-Prozess (zeigt die aktive Config)
- Anzahl bereits geladener Vektoren in der Ziel-DB (Insert-Phase)
- Phasenanteile pro Config (setup/insert/build/query/write)

Aufruf:
    python3 benchmarks/runners/status.py
    python3 benchmarks/runners/status.py --batch weaviate-T2-latency,weaviate-T2-throughput,pgvector-T2-ivfflat,pgvector-T2-latency
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# Phasen-Anteile -- summieren sich auf 1.0. Werte aus Beobachtung der bisherigen
# Läufe abgeleitet, nicht streng kalibriert -- reichen aber für eine sinnvolle
# Prozentanzeige.
PHASE_WEIGHTS = {
    "setup": 0.05,
    "insert": 0.55,
    "build": 0.25,
    "query": 0.10,
    "write": 0.05,
}

# Default-Batch (das was wir aktuell fahren).
DEFAULT_BATCH = [
    "weaviate-T2-latency",
    "weaviate-T2-throughput",
    "pgvector-T2-ivfflat",
    "pgvector-T2-latency",
]


# ----- Helpers -------------------------------------------------------------

def list_done(batch: list[str]) -> list[str]:
    """Configs aus der Batch, für die schon ein summary.json existiert."""
    done = []
    for cfg in batch:
        # Pattern: <ts>_<cfg-name>/summary.json
        matches = list(RESULTS_DIR.glob(f"*_{cfg}/summary.json"))
        if matches:
            done.append(cfg)
    return done


def active_config(batch: list[str]) -> str | None:
    """Sucht im Prozess-Baum nach einem aktiven runner.py --config <X>."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "args"], text=True, timeout=5,
        )
    except Exception:
        return None
    for line in out.splitlines():
        m = re.search(r"runner\.py\s+--config\s+(\S+)", line)
        if m:
            cfg = m.group(1)
            if cfg in batch:
                return cfg
    return None


def db_for_config(cfg: str) -> str:
    return cfg.split("-")[0]  # weaviate-T2-latency -> weaviate


def _stufe_vectors() -> dict:
    """STUFE_VECTORS direkt aus runner.py -- single source of truth. Vorher lag
    hier eine veraltete Kopie (S=100k statt 2,4M, kein S0/S1), die die
    Fortschritts-Schaetzung verfaelscht hat."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import STUFE_VECTORS
    return STUFE_VECTORS


def target_n_vectors(cfg: str) -> int:
    """Lädt die Config, gibt die Soll-Vektorenzahl aus STUFE_VECTORS zurück."""
    cfg_path = REPO_ROOT / "benchmarks" / "configs" / f"{cfg}.json"
    if not cfg_path.exists():
        return 0
    data = json.loads(cfg_path.read_text())
    return _stufe_vectors().get(data["stufe"], 0)


def weaviate_count() -> int | None:
    """Aktuelle Object-Count in der Bench-Collection. Via HTTP-API auf den
    Cluster-Service, weil das port-forward-Setup des Runners auf 8080 hört."""
    try:
        import urllib.request
        url = "http://127.0.0.1:8080/v1/graphql"
        body = json.dumps({
            "query": "{ Aggregate { Bench { meta { count } } } }"
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read())
        return data["data"]["Aggregate"]["Bench"][0]["meta"]["count"]
    except Exception:
        return None


def pgvector_count() -> int | None:
    """Row-Count in bench_items via psql-Pod im Cluster."""
    try:
        out = subprocess.check_output(
            ["kubectl", "exec", "-n", "db-pgvector", "pgvector-0",
             "-c", "pgvector", "--",
             "psql", "-U", "bench", "-d", "benchmark", "-tAc",
             "SELECT COUNT(*) FROM bench_items"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        return int(out.strip())
    except Exception:
        # Fallback ohne -c (psql-Container ist Default)
        try:
            out = subprocess.check_output(
                ["kubectl", "exec", "-n", "db-pgvector", "pgvector-0", "--",
                 "psql", "-U", "bench", "-d", "benchmark", "-tAc",
                 "SELECT COUNT(*) FROM bench_items"],
                text=True, timeout=10, stderr=subprocess.DEVNULL,
            )
            return int(out.strip())
        except Exception:
            return None


def pod_top(namespace: str, pod: str) -> tuple[float, float] | None:
    try:
        out = subprocess.check_output(
            ["kubectl", "top", "pod", "-n", namespace, pod, "--no-headers"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        parts = out.split()
        if len(parts) < 3:
            return None
        # CPU in mCores, mem in Mi
        cpu = parts[1]
        mem = parts[2]
        cpu_v = float(cpu.rstrip("m")) / 1000.0 if cpu.endswith("m") else float(cpu)
        mem_v = float(mem.rstrip("Mi")) if mem.endswith("Mi") else float(mem.rstrip("Gi")) * 1024
        return cpu_v, mem_v
    except Exception:
        return None


# ----- Progress estimation -------------------------------------------------

def estimate_current_progress(cfg: str) -> tuple[float, str]:
    """Gibt (progress_in_0_to_1, phase_label) zurück. Phase wird heuristisch
    abgeleitet."""
    db = db_for_config(cfg)
    target = target_n_vectors(cfg) or 1
    if db == "weaviate":
        count = weaviate_count()
    elif db == "pgvector":
        count = pgvector_count()
    else:
        count = None

    cumul = PHASE_WEIGHTS["setup"]

    if count is None or count == 0:
        return cumul * 0.5, "setup/warm-up"

    if count < target:
        ratio = count / target
        prog = cumul + PHASE_WEIGHTS["insert"] * ratio
        return prog, f"insert {count:,}/{target:,} ({ratio*100:.1f}%)"

    # Insert fertig -- jetzt Build oder Query. Phase via kubectl top heuristik:
    # hohe CPU bei pg = HNSW-Bau läuft, niedrige CPU = Query-Phase
    cumul += PHASE_WEIGHTS["insert"]
    namespace, pod = (
        ("db-weaviate", "weaviate-0") if db == "weaviate" else ("db-pgvector", "pgvector-0")
    )
    top = pod_top(namespace, pod)
    if top is not None and top[0] > 0.5 and db == "pgvector":
        # Heuristisch: pgvector HNSW-Build glüht die CPU. Wir können
        # nicht genau wissen wie weit der Build ist -- 50% Schätzung.
        return cumul + PHASE_WEIGHTS["build"] * 0.5, f"build index (CPU {top[0]:.2f}c)"

    cumul += PHASE_WEIGHTS["build"]
    # Wenn wir hier sind: vermutlich query loop oder write
    return cumul + PHASE_WEIGHTS["query"] * 0.5, "query loop / write"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch",
        default=",".join(DEFAULT_BATCH),
        help="Komma-getrennte Liste der Configs im aktuellen Batch",
    )
    p.add_argument("--json", action="store_true", help="JSON statt human-readable")
    args = p.parse_args()

    batch = [c.strip() for c in args.batch.split(",") if c.strip()]
    done = list_done(batch)
    active = active_config(batch)

    # Pro-Batch-Progress
    n_done = len(done)
    n_total = len(batch)
    per_cfg = 1.0 / n_total

    overall = n_done * per_cfg
    current_prog = 0.0
    current_phase = ""
    if active:
        current_prog, current_phase = estimate_current_progress(active)
        overall += current_prog * per_cfg

    result = {
        "batch": batch,
        "done": done,
        "active": active,
        "current_phase": current_phase,
        "current_progress": round(current_prog * 100, 1),
        "overall_progress": round(overall * 100, 1),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Human output
    bar_width = 30
    filled = int(round(overall * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"Batch: {n_done}/{n_total} Configs fertig")
    print(f"Gesamt: [{bar}] {overall*100:.1f}%")
    print()
    for cfg in batch:
        if cfg in done:
            # Letztes Result für diese Config einlesen
            matches = sorted(RESULTS_DIR.glob(f"*_{cfg}/summary.json"))
            s = json.loads(matches[-1].read_text())
            m = s["metrics"]
            r = s["resources"]
            print(
                f"  ✓ {cfg:28s}  "
                f"qps={m.get('throughput_qps','—'):>6}  "
                f"p50={m.get('latency_ms_p50','—')}ms  "
                f"recall@10={(m.get('recall_at_10') or 0)*100:5.1f}%  "
                f"cpu={r.get('cpu_avg_cores','—')}c  "
                f"mem={r.get('mem_avg_mb','—')}MB"
            )
        elif cfg == active:
            cbar_filled = int(round(current_prog * bar_width))
            cbar = "█" * cbar_filled + "░" * (bar_width - cbar_filled)
            print(f"  → {cfg:28s}  [{cbar}] {current_prog*100:5.1f}%  {current_phase}")
        else:
            print(f"  · {cfg:28s}  ausstehend")


if __name__ == "__main__":
    main()
