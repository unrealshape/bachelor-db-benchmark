#!/usr/bin/env python3
"""Trigger für Mess-Läufe. Nimmt eine Config, führt entweder einen Dummy-
Run oder einen echten Lauf gegen den k3d-Cluster durch, schreibt summary.json
und rebuilded den Index. Optional commit + push.

Echter Lauf:
    runner.py --config weaviate-T-latency
Dummy:
    runner.py --config weaviate-S-latency --dummy
"""

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "benchmarks" / "configs"

# Spec-Version-Tag. Wird in jede summary.json geschrieben, damit Auswertungen
# zwischen Mess-Läufen unterschiedlicher Methodik-Stände unterscheiden können.
# "1024-bge-v2" = 1024 dim BAAI/bge-large-en-v1.5 (lokal). v2: feasible Stufen
# S0/S1/S/M, Warmup-Discard, Disk-I/O-Sampling, echter n_vectors, Speicherdruck-
# Achse (mem_limit_gb). L/XL/XXL als HW-Grenze dokumentiert (siehe thesis-redesign.md).
SPEC_VERSION = "1024-bge-v2"

# Vektor-Zahl pro Stufe. S0/S1/S/M sind die feasiblen Stufen (Redesign 2026-06).
# L/XL/XXL bleiben als Referenz, werden auf 32-GB-Hardware aber nicht gefahren.
# T/T2 sind reine Dev-Stufen mit Synthese-Daten. Wenn der Loader eine
# corpus_meta.json schreibt, hat sie Vorrang -- die Werte hier sind Richtgrößen.
STUFE_VECTORS = {
    "T":       20_000,   # Dev-Pipeline (Synthese, nicht in der Thesis)
    "T2":     100_000,   # Stabilisierungs-Runs (Synthese, nicht in der Thesis)
    "S0":     500_000,   # feasible Stufe (~2 GB) -- unterer Kurvenpunkt
    "S1":   1_200_000,   # feasible Stufe (~5 GB)
    "S":    2_400_000,   # feasible Stufe (~10 GB) -- Haupt-Stufe
    "M":    4_900_000,   # feasible Stufe (~20 GB) -- Ceiling + Druck-Achse
    "L":    9_800_000,   # HW-Grenze (~40 GB, nicht gefahren)
    "XL":  19_500_000,   # HW-Grenze (~80 GB, nicht gefahren)
    "XXL": 24_400_000,   # HW-Grenze (~100 GB, nicht gefahren)
}

# Grobe On-Disk-Größe pro Stufe (Parquet inkl. Index, bge-large-en-v1.5 1024 dim).
STUFE_GB = {
    "T":     0.10,
    "T2":    0.56,
    "S0":    2.0,
    "S1":    5.0,
    "S":    10.0,
    "M":    20.0,
    "L":    40.0,
    "XL":   80.0,
    "XXL": 100.0,
}

DEFAULT_DEMODATA_DIR = Path(
    os.environ.get(
        "BENCH_DEMODATA_DIR",
        Path.home() / ".cache" / "bachelor-db-benchmark",
    )
)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_id():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def load_config(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIGS_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise SystemExit(f"Config nicht gefunden: {name_or_path}")
    return json.loads(p.read_text())


def list_configs():
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))


# ----- Dummy-Path ----------------------------------------------------------

def fake_metrics(cfg):
    rng = random.Random(hash(f"{cfg['db']}{cfg['stufe']}{cfg['workload']}{cfg['name']}") & 0xFFFFFFFF)
    base_lat = {"weaviate": 2.0, "pgvector": 3.5, "pinecone": 18.0}[cfg["db"]]
    base_qps = {"weaviate": 450, "pgvector": 280, "pinecone": 600}[cfg["db"]]
    scale = {"T": 0.7, "T2": 0.85, "S": 1.0, "M": 1.15, "L": 1.35, "XL": 1.7}[cfg["stufe"]]
    conc = cfg.get("queries", {}).get("concurrency", 1)

    return {
        "throughput_qps": round((base_qps / scale) * min(conc, 4) + rng.uniform(-15, 15), 1),
        "latency_ms_mean": round(base_lat * scale * 1.2 + rng.uniform(-0.2, 0.2), 2),
        "latency_ms_p50": round(base_lat * scale + rng.uniform(-0.3, 0.3), 2),
        "latency_ms_p95": round(base_lat * scale * 2.5 + rng.uniform(-0.5, 0.5), 2),
        "latency_ms_p99": round(base_lat * scale * 4.0 + rng.uniform(-1, 1), 2),
        "recall_at_1": round(rng.uniform(0.97, 0.995), 4),
        "recall_at_10": round(rng.uniform(0.93, 0.985), 4),
        "recall_at_100": round(rng.uniform(0.88, 0.96), 4),
        "precision_at_10": round(rng.uniform(0.92, 0.985), 4),
        "ndcg_at_10": round(rng.uniform(0.90, 0.97), 4),
    }


# ----- Echter Lauf ---------------------------------------------------------

META_COLS = ("product_id", "product_title", "user_id",
             "rating", "review_text", "timestamp")


def detect_corpus_dim(stufe_dir: Path) -> int:
    """Liest die Embedding-Dimension aus dem ersten Parquet-Chunk -- robust
    gegen falsch konfigurierte dim-Werte in der Config."""
    import pyarrow.parquet as pq
    paths = sorted(stufe_dir.glob("chunk_*.parquet"))
    if not paths:
        return 0
    schema = pq.read_schema(paths[0])
    field = schema.field("embedding")
    # FixedSizeList trägt die list_size direkt
    if hasattr(field.type, "list_size"):
        return int(field.type.list_size)
    # Fallback: erste Zeile lesen
    tbl = pq.read_table(paths[0], columns=["embedding"]).slice(0, 1)
    return len(tbl["embedding"][0].as_py())


def load_corpus_chunks(stufe_dir: Path, dim: int):
    """Liefert eine Liste (ids, vecs, metadata|None) pro Parquet-Chunk.
    metadata ist None falls die Chunks nur id + embedding enthalten (Demodata)."""
    import numpy as np
    import pyarrow.parquet as pq

    paths = sorted(stufe_dir.glob("chunk_*.parquet"))
    if not paths:
        raise SystemExit(
            f"Keine corpus-chunks unter {stufe_dir}. "
            f"Stufen S/M/L/XL/XXL: python benchmarks/reviewdata/load.py "
            f"--stage <S|M|L|XL|XXL>. "
            f"Dev-Stufen T/T2: python benchmarks/demodata/generate.py "
            f"--output-dir {stufe_dir} --size <T|T2>."
        )
    out = []
    for p in paths:
        schema_cols = set(pq.read_schema(p).names)
        meta_cols = [c for c in META_COLS if c in schema_cols]
        cols = ["id", "embedding"] + meta_cols
        tbl = pq.read_table(p, columns=cols)
        ids = tbl["id"].to_numpy()
        flat = tbl["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False)
        emb = flat.astype(np.float32, copy=False).reshape(-1, dim)
        metadata = None
        if meta_cols:
            metadata = {c: tbl[c].to_pylist() for c in meta_cols}
        out.append((ids, emb, metadata))
    return out


def corpus_chunk_paths(stufe_dir: Path):
    return sorted(stufe_dir.glob("chunk_*.parquet"))


def stream_corpus_chunks(stufe_dir: Path, dim: int):
    """Wie load_corpus_chunks, aber als Generator -- haelt nur EINEN Chunk im
    RAM. Noetig fuer grosse Stufen (M+), deren Gesamtkorpus (20-100 GB) nicht in
    den Host-RAM passt. Liefert (ids, vecs, metadata|None) pro Parquet-Chunk."""
    import numpy as np
    import pyarrow.parquet as pq

    paths = corpus_chunk_paths(stufe_dir)
    if not paths:
        raise SystemExit(
            f"Keine corpus-chunks unter {stufe_dir}. "
            f"Stufen S/M/L/XL/XXL: python benchmarks/reviewdata/load.py "
            f"--stage <S|M|L|XL|XXL>."
        )
    for p in paths:
        schema_cols = set(pq.read_schema(p).names)
        meta_cols = [c for c in META_COLS if c in schema_cols]
        cols = ["id", "embedding"] + meta_cols
        tbl = pq.read_table(p, columns=cols)
        ids = tbl["id"].to_numpy()
        flat = tbl["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False)
        emb = flat.astype(np.float32, copy=False).reshape(-1, dim)
        metadata = None
        if meta_cols:
            metadata = {c: tbl[c].to_pylist() for c in meta_cols}
        yield ids, emb, metadata


def load_queries(stufe_dir: Path, cfg: dict | None = None):
    """Laed queries.npy und die zur Workload passende Ground Truth.

    Default: ground_truth_ids.npy (Brute-Force-Cosine ueber den ganzen Korpus).
    Bei cfg.filter setzt: ground_truth_filter_<spec>_ids.npy
    Bei cfg.hybrid gesetzt: ground_truth_hybrid_alpha_<NN>_ids.npy
    Fehlt die spezialisierte GT, faellt's auf die Default-GT zurueck und
    setzt eine Warnung in den Notes (Konsumenten muessen das interpretieren).
    """
    import numpy as np
    qd = stufe_dir / "queries"
    q = np.load(qd / "queries.npy")

    gt_path = qd / "ground_truth_ids.npy"
    note = None
    if cfg:
        f = cfg.get("filter") or {}
        if f:
            spec_parts = [f"{k}_{v}" for k, v in sorted(f.items())]
            suffix = "_".join(spec_parts)
            cand = qd / f"ground_truth_filter_{suffix}_ids.npy"
            if cand.exists():
                gt_path = cand
            else:
                note = f"filter-spezifische GT fehlt ({cand.name}) -- nutze Default-GT"
        h = cfg.get("hybrid") or {}
        if h:
            alpha = h.get("alpha", 0.5)
            suffix = f"alpha_{int(round(alpha*100)):02d}"
            cand = qd / f"ground_truth_hybrid_{suffix}_ids.npy"
            if cand.exists():
                gt_path = cand
            else:
                note = f"hybrid-spezifische GT fehlt ({cand.name}) -- nutze Default-GT"

    gt = np.load(gt_path)
    return q, gt, gt_path.name, note


def load_query_meta(stufe_dir: Path) -> dict | None:
    """Lädt Query-Metadaten (Text, Rating, …) für filtered/hybrid Workloads.
    Gibt None zurück wenn queries.parquet fehlt (alter Demodata-Stand)."""
    import pyarrow.parquet as pq
    qp = stufe_dir / "queries" / "queries.parquet"
    if not qp.exists():
        return None
    tbl = pq.read_table(qp)
    out = {}
    for c in ("id", "rating", "review_text", "product_id", "product_title"):
        if c in tbl.column_names:
            out[c] = tbl[c].to_pylist()
    return out


DB_POD = {
    "weaviate": ("db-weaviate", "weaviate-0"),
    "pgvector": ("db-pgvector", "pgvector-0"),
}

# StatefulSet / Deployment Namen pro DB fuer den Pre-Run-Hook.
DB_WORKLOAD = {
    "weaviate": ("statefulset", "weaviate"),
    "pgvector": ("statefulset", "pgvector"),
}


def pre_run_reset(db: str, mem_limit_gb: int | float | None = None,
                  timeout_s: int = 180) -> dict:
    """Pod-Restart vor jedem echten Lauf, damit der Index aus einem definierten
    Zustand neu aufgebaut wird (Thesis 5: Caches geleert, DB neugestartet).

    OS-Page-Cache laesst sich in k3d nicht zuverlaessig droppen -- dokumentiert
    als Limitation. Effekt: DB-internen Cache nullen wir ueber den Pod-Restart.

    mem_limit_gb (Speicherdruck-Achse): patcht das Container-Memory-Limit der
    StatefulSet vor dem Restart. So wird derselbe M-Korpus unter sinkendem
    Pod-RAM gemessen -> 'Index waechst raus'-Degradation (dokumentierte
    Abweichung von der 8-GB-Paritaet)."""
    namespace, _ = DB_POD[db]
    kind, name = DB_WORKLOAD[db]
    notes = {"pre_run_reset": "rollout-restart", "timeout_s": timeout_s}
    if mem_limit_gb:
        patch = [{
            "op": "replace",
            "path": "/spec/template/spec/containers/0/resources/limits/memory",
            "value": f"{mem_limit_gb}Gi",
        }]
        print(f"  pre-run: patch {kind}/{name} memory-limit -> {mem_limit_gb}Gi",
              flush=True)
        subprocess.run(
            ["kubectl", "-n", namespace, "patch", kind, name,
             "--type=json", "-p", json.dumps(patch)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        notes["mem_limit_gb"] = mem_limit_gb
    print(f"  pre-run reset: rollout restart {kind}/{name}", flush=True)
    subprocess.run(
        ["kubectl", "-n", namespace, "rollout", "restart", f"{kind}/{name}"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["kubectl", "-n", namespace, "rollout", "status", f"{kind}/{name}",
         f"--timeout={timeout_s}s"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    return notes


def real_run(cfg: dict, demodata_dir: Path, dim: int, run_id: str | None = None):
    """Führt einen echten Run aus, gibt (metrics_dict, build_time_s, size_mb,
    resources_dict, cluster_dict, notes) zurück.

    Mit BENCH_INCLUSTER=1 laeuft der Query-Loop als Pod im Cluster (ClusterIP,
    kein port-forward im Mess-Pfad); insert + build_index bleiben host-seitig."""
    # Dependencies werden lazy importiert -- pyarrow/numpy braucht der Dummy-
    # Path nicht.
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "runners"))
    from adapters import get_adapter
    from cluster_metrics import ResourceSampler, cluster_info

    import pyarrow.parquet as pq

    stufe_dir = demodata_dir / cfg["stufe"]
    queries, ground_truth, gt_file, gt_note = load_queries(stufe_dir, cfg)

    # Streaming-Insert: nicht den ganzen Korpus in den RAM laden (M+ = 20-100 GB).
    chunk_paths = corpus_chunk_paths(stufe_dir)
    if not chunk_paths:
        raise SystemExit(f"Keine corpus-chunks unter {stufe_dir}.")
    first_cols = set(pq.read_schema(chunk_paths[0]).names)
    has_metadata = any(c in first_cols for c in META_COLS)
    n_chunks = len(chunk_paths)
    n_query = cfg["queries"]["n"]
    concurrency = cfg["queries"]["concurrency"]
    k_max = 100

    AdapterCls = get_adapter(cfg["db"])
    adapter = AdapterCls(cfg, dim=dim)

    # Cluster-Stammdaten einmalig, Resource-Sampler über die ganze Lauf-Dauer.
    cluster = cluster_info()
    namespace, pod = DB_POD[cfg["db"]]
    sampler = ResourceSampler(namespace=namespace, pod=pod, interval_s=2.0)

    notes = {}
    # Pre-Run-Hook: Pod-Restart, sofern nicht in der Config abgewaehlt.
    if cfg.get("pre_run_reset", True):
        try:
            notes.update(pre_run_reset(cfg["db"], mem_limit_gb=cfg.get("mem_limit_gb")))
        except subprocess.CalledProcessError as e:
            notes["pre_run_reset_error"] = str(e)
            print(f"  pre-run reset fehlgeschlagen: {e} -- fahre fort", flush=True)

    try:
        print(f"  setup ({adapter.db_name})...", flush=True)
        adapter.setup()

        sampler.start()
        print(f"  insert (streaming) in {n_chunks} Chunks"
              + (" + Metadaten" if has_metadata else "") + "...", flush=True)
        t0 = time.time()
        n_vec = 0
        for ids, vecs, metadata in stream_corpus_chunks(stufe_dir, dim):
            n_vec += len(ids)
            adapter.insert(ids, vecs, metadata=metadata)
        insert_s = time.time() - t0
        notes["insert_time_s"] = round(insert_s, 2)
        notes["n_vectors_actual"] = n_vec  # echter Count, nicht der Stufen-Zielwert
        notes["has_metadata"] = has_metadata
        notes["gt_file"] = gt_file
        if gt_note:
            notes["gt_note"] = gt_note

        print(f"  build index...", flush=True)
        build_s = adapter.build_index()

        # In-Cluster-Messung: Query-Loop laeuft als Pod via ClusterIP (kein
        # port-forward im Mess-Pfad). Host bleibt Orchestrator (insert/build/
        # Resource-Sampling laufen weiter).
        if os.environ.get("BENCH_INCLUSTER") == "1":
            from incluster import run_incluster_measure
            print("  in-cluster measure (Job)...", flush=True)
            out = run_incluster_measure(cfg, run_id, cfg["stufe"])
            metrics = out["metrics"]
            notes["measured"] = "in-cluster"
            notes["n_queries_executed"] = out.get("n_queries_executed")
            notes["n_warmup"] = out.get("n_warmup")
            notes["gt_file"] = out.get("gt_file")
            if out.get("gt_note"):
                notes["gt_note"] = out["gt_note"]
            size_mb = adapter.index_size_mb()
            avg = sampler.stop()
            resources = {
                "cpu_avg_cores": avg.cpu_avg_cores,
                "mem_avg_mb": avg.mem_avg_mb,
                "cpu_peak_cores": avg.cpu_peak_cores,
                "mem_peak_mb": avg.mem_peak_mb,
                "disk_read_mb": avg.disk_read_mb,
                "disk_write_mb": avg.disk_write_mb,
                "disk_read_ios": avg.disk_read_ios,
                "disk_write_ios": avg.disk_write_ios,
                "samples": avg.n_samples,
            }
            return metrics, round(build_s, 2), size_mb, resources, cluster, notes

        # Effektive Anzahl Queries (kann kleiner sein als queries.npy enthält)
        n_query = min(n_query, queries.shape[0])

        print(f"  query loop  n={n_query}  conc={concurrency}...", flush=True)
        latencies_ms = []
        recalls_1, recalls_10, recalls_100 = [], [], []
        precisions_10 = []
        ndcgs_10 = []

        workload = cfg.get("workload", "topk")
        filter_spec = cfg.get("filter", {})
        hybrid_alpha = cfg.get("hybrid", {}).get("alpha", 0.5)
        query_meta = load_query_meta(stufe_dir)
        query_texts = (query_meta or {}).get("review_text") if query_meta else None

        def do_query(i: int, v: np.ndarray) -> list[int]:
            if workload == "topk":
                return adapter.query(v, k_max)
            if workload == "filtered":
                return adapter.query_filtered(v, k_max, filter_spec)
            if workload == "batch":
                # Batch-Charakteristik kommt aus concurrency > 1 weiter unten.
                # Pro Einzelquery ruft uns ThreadPoolExecutor parallel auf.
                return adapter.query(v, k_max)
            if workload == "hybrid":
                if not query_texts:
                    raise SystemExit(
                        "hybrid braucht queries.parquet mit review_text -- "
                        "reviewdata/load.py + reviewdata/gen_queries.py "
                        "liefern das automatisch mit"
                    )
                return adapter.query_hybrid(
                    v, query_texts[i], k_max, alpha=hybrid_alpha,
                )
            raise ValueError(f"unbekannter workload: {workload}")

        def one_query(i: int):
            v = queries[i]
            t = time.perf_counter()
            retrieved = do_query(i, v)
            dt_ms = (time.perf_counter() - t) * 1000.0
            truth = ground_truth[i]
            return (
                dt_ms,
                adapter.recall_at_k(retrieved, truth, 1),
                adapter.recall_at_k(retrieved, truth, 10),
                adapter.recall_at_k(retrieved, truth, 100),
                adapter.precision_at_k(retrieved, truth, 10),
                adapter.ndcg_at_k(retrieved, truth, 10),
            )

        # Warmup: erste Queries fuellen DB-/OS-Cache, Timings verworfen, damit die
        # Messung nicht kalt-kontaminiert ist (Thesis 5.5.1). Sequentiell, gedeckelt
        # auf die verfuegbaren Queries.
        n_warmup = min(cfg["queries"].get("n_warmup", 1000), n_query)
        for i in range(n_warmup):
            do_query(i, queries[i])
        notes["n_warmup"] = n_warmup

        t_qstart = time.time()
        if concurrency <= 1:
            for i in range(n_query):
                dt, r1, r10, r100, p10, ndcg = one_query(i)
                latencies_ms.append(dt)
                recalls_1.append(r1)
                recalls_10.append(r10)
                recalls_100.append(r100)
                precisions_10.append(p10)
                ndcgs_10.append(ndcg)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                for dt, r1, r10, r100, p10, ndcg in ex.map(one_query, range(n_query)):
                    latencies_ms.append(dt)
                    recalls_1.append(r1)
                    recalls_10.append(r10)
                    recalls_100.append(r100)
                    precisions_10.append(p10)
                    ndcgs_10.append(ndcg)
        wall_s = time.time() - t_qstart
        qps = n_query / wall_s if wall_s > 0 else 0.0

        def pct(xs, p):
            xs = sorted(xs)
            k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
            return xs[k]

        metrics = {
            "throughput_qps": round(qps, 1),
            "latency_ms_mean": round(statistics.mean(latencies_ms), 2),
            "latency_ms_p50": round(pct(latencies_ms, 50), 2),
            "latency_ms_p95": round(pct(latencies_ms, 95), 2),
            "latency_ms_p99": round(pct(latencies_ms, 99), 2),
            "recall_at_1": round(statistics.mean(recalls_1), 4),
            "recall_at_10": round(statistics.mean(recalls_10), 4),
            "recall_at_100": round(statistics.mean(recalls_100), 4),
            "precision_at_10": round(statistics.mean(precisions_10), 4),
            "ndcg_at_10": round(statistics.mean(ndcgs_10), 4),
        }
        size_mb = adapter.index_size_mb()
        notes["n_queries_executed"] = n_query

        # Pinecone-Adapter liefert Server-Latenz aus dem x-pinecone-request-latency-ms
        # Header. Wenn vorhanden, neben den Client-Latenzen in den Notes ausweisen,
        # damit der Netz-Hop zur Cloud aus dem Vergleich herausgerechnet werden kann.
        if hasattr(adapter, "server_latency_summary"):
            try:
                srv = adapter.server_latency_summary()
                if srv:
                    notes["server_latency_ms"] = srv
            except Exception as e:
                notes["server_latency_error"] = str(e)

        avg = sampler.stop()
        resources = {
            "cpu_avg_cores": avg.cpu_avg_cores,
            "mem_avg_mb": avg.mem_avg_mb,
            "cpu_peak_cores": avg.cpu_peak_cores,
            "mem_peak_mb": avg.mem_peak_mb,
            "samples": avg.n_samples,
        }

        return metrics, round(build_s, 2), size_mb, resources, cluster, notes
    finally:
        sampler.stop()
        adapter.teardown()


# ----- Output --------------------------------------------------------------

def write_summary(run_dir, cfg, metrics, started_at, *,
                  build_time_s=None, size_on_disk_mb=None,
                  resources=None, cluster=None, dim_used=None,
                  notes_dict=None, status="ok"):
    finished_at = now_iso()
    duration_s = int(time.time() - time.mktime(time.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ")))
    summary = {
        "run_id": run_dir.name,
        "config_name": cfg["name"],
        "spec_version": SPEC_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": status,
        "db": {"name": cfg["db"], "version": None, "image": None},
        "dataset": {
            "size_label": cfg["stufe"],
            "n_vectors": (notes_dict or {}).get("n_vectors_actual") or STUFE_VECTORS.get(cfg["stufe"]),
            "dim": dim_used if dim_used is not None else cfg.get("dim", 1024),
            "variant": cfg.get("variant"),
            "size_gb": STUFE_GB.get(cfg["stufe"]),
        },
        "index": {
            "type": cfg["index"]["type"],
            "params": cfg["index"]["params"],
            "build_time_s": build_time_s,
            "size_on_disk_mb": size_on_disk_mb,
        },
        "workload": {
            "profile": cfg["workload"],
            "n_queries": cfg["queries"]["n"],
            "concurrency": cfg["queries"]["concurrency"],
        },
        "metrics": metrics,
        "resources": resources or {"cpu_avg_cores": None, "mem_avg_mb": None},
        "cluster": cluster or {"k8s_version": None, "nodes": None},
        "notes": notes_dict or {},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return summary


def rebuild_index():
    runs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "summary.json"
        if not sf.exists():
            continue
        try:
            s = json.loads(sf.read_text())
        except json.JSONDecodeError:
            continue
        s_notes = s.get("notes") if isinstance(s.get("notes"), dict) else {}
        s_res = s.get("resources") or {}
        runs.append({
            "id": s["run_id"],
            "config_name": s.get("config_name"),
            "spec_version": s.get("spec_version", "pre-1536"),
            "db": s["db"]["name"],
            "stufe": s["dataset"]["size_label"],
            "workload": s["workload"]["profile"],
            "status": s.get("status", "ok"),
            "started_at": s["started_at"],
            "throughput_qps": s["metrics"].get("throughput_qps"),
            "latency_ms_p50": s["metrics"].get("latency_ms_p50"),
            "latency_ms_p95": s["metrics"].get("latency_ms_p95") or s["metrics"].get("latency_ms_p90"),
            "latency_ms_p99": s["metrics"].get("latency_ms_p99"),
            "recall_at_10": s["metrics"].get("recall_at_10"),
            "size_gb": s["dataset"].get("size_gb"),
            "index_type": s["index"].get("type"),
            # Parametrisierungen -- damit das Dashboard danach filtern kann.
            "variant": s["dataset"].get("variant"),
            "dim": s["dataset"].get("dim"),
            "n_vectors": s["dataset"].get("n_vectors"),
            "concurrency": s["workload"].get("concurrency"),
            "build_time_s": s["index"].get("build_time_s"),
            "mem_limit_gb": s_notes.get("mem_limit_gb"),
            "n_warmup": s_notes.get("n_warmup"),
            "disk_read_mb": s_res.get("disk_read_mb"),
            "repeat_group": s_notes.get("repeat_group"),
            "repeat_index": s_notes.get("repeat_index"),
        })
    runs.sort(key=lambda r: r["started_at"], reverse=True)
    index = {"generated_at": now_iso(), "n_runs": len(runs), "runs": runs}
    (RESULTS_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def git_commit_push(run_id):
    subprocess.run(["git", "add", "results/"], cwd=REPO_ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode == 0:
        print("nichts zu committen")
        return
    msg = f"feat(results): add run {run_id}"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True)


# ----- CLI -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="Name (z.B. 'weaviate-T-latency') oder Pfad zur Config")
    p.add_argument("--list", action="store_true", help="verfügbare Configs zeigen")
    p.add_argument("--dummy", action="store_true", help="plausible Beispiel-Metriken statt echtem Lauf")
    p.add_argument("--demodata-dir", type=Path, default=DEFAULT_DEMODATA_DIR,
                   help="Basisverzeichnis für demodata (default: ~/.cache/bachelor-db-benchmark)")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="nach dem Schreiben NICHT committen + pushen. "
                        "Default ist push fuer echte Runs auf Stufe S/M/L/XL/XXL.")
    p.add_argument("--push", dest="push", action="store_true",
                   help="explizit committen + pushen (auch auf T/T2 oder bei --dummy).")
    p.add_argument("--repeat", type=int, default=1,
                   help="Lauf N-mal wiederholen (Varianz-Analyse). Jede Wdh. "
                        "schreibt eine eigene summary.json, getaggt mit "
                        "repeat_group/repeat_index in den Notes.")
    p.set_defaults(push=None)
    args = p.parse_args()

    # Auto-Push-Logik: Default ist push fuer echte Runs auf den Prod-Stufen.
    # Dummy-Laufe und Dev-Stufen T/T2 werden nicht gepusht, ausser --push explizit.
    if args.push is None:
        try:
            _cfg_peek = load_config(args.config) if args.config else {}
        except SystemExit:
            _cfg_peek = {}
        prod_stage = _cfg_peek.get("stufe") in {"S0", "S1", "S", "M", "L", "XL", "XXL"}
        args.push = bool(prod_stage and not args.dummy)

    if args.list:
        for name in list_configs():
            print(name)
        return

    if not args.config:
        sys.exit("--config <name> oder --list")

    cfg = load_config(args.config)
    n_repeat = max(1, args.repeat)
    # Gemeinsamer Tag, der alle Wiederholungen desselben Aufrufs verknuepft.
    repeat_group = f"{now_id()}_{cfg['name']}" if n_repeat > 1 else None
    n_failed = 0

    for rep in range(n_repeat):
        started_at = now_iso()
        run_id = f"{now_id()}_{cfg['name']}"
        if n_repeat > 1:
            run_id += f"-r{rep + 1}"
        run_dir = RESULTS_DIR / run_id
        tag = f"  (Wdh {rep + 1}/{n_repeat})" if n_repeat > 1 else ""
        print(f"▸ Run: {run_id}{tag}")

        build_s = None
        size_mb = None
        resources = None
        cluster = None
        notes_dict = {}

        dim_used = cfg.get("dim", 1024)
        if args.dummy:
            metrics = fake_metrics(cfg)
            notes_dict["mode"] = "dummy"
        else:
            detected = detect_corpus_dim(args.demodata_dir / cfg["stufe"])
            dim_used = detected or cfg.get("dim", 1024)
            if detected and detected != cfg.get("dim", detected):
                print(f"  Hinweis: Config-dim {cfg.get('dim')} != Korpus-dim {detected} -- benutze {detected}", flush=True)
            try:
                metrics, build_s, size_mb, resources, cluster, adapter_notes = real_run(
                    cfg, args.demodata_dir, dim=dim_used, run_id=run_id,
                )
            except Exception as e:
                # Ein fehlgeschlagener Repeat darf die uebrigen Wiederholungen
                # nicht killen (Varianz-Analyse braucht alle Datenpunkte). Lauf
                # ueberspringen, naechsten Repeat versuchen.
                n_failed += 1
                print(f"  ✗ Run fehlgeschlagen ({type(e).__name__}: {e}) -- "
                      f"ueberspringe Wdh {rep + 1}/{n_repeat}", flush=True)
                continue
            notes_dict.update(adapter_notes)
            notes_dict["mode"] = "real"

        if n_repeat > 1:
            notes_dict["repeat_group"] = repeat_group
            notes_dict["repeat_index"] = rep + 1
            notes_dict["repeat_total"] = n_repeat

        write_summary(
            run_dir, cfg, metrics, started_at,
            build_time_s=build_s, size_on_disk_mb=size_mb,
            resources=resources, cluster=cluster, dim_used=dim_used,
            notes_dict=notes_dict,
        )
        print(
            f"  qps={metrics['throughput_qps']}  "
            f"p50={metrics['latency_ms_p50']}ms  "
            f"p95={metrics['latency_ms_p95']}ms  "
            f"recall@10={metrics['recall_at_10']}"
        )
        if build_s is not None:
            print(f"  build_time={build_s}s  index_size={size_mb} MB")

        index = rebuild_index()
        print(f"  Index: {index['n_runs']} Runs gesamt")

        if args.push:
            git_commit_push(run_id)
            print("  gepusht")

    if n_repeat > 1:
        print(f"= {cfg['name']}: {n_repeat - n_failed}/{n_repeat} Wdh ok"
              + (f", {n_failed} fehlgeschlagen" if n_failed else ""))


if __name__ == "__main__":
    main()
