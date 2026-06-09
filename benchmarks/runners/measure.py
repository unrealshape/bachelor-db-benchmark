#!/usr/bin/env python3
"""In-Cluster Mess-Harness.

Laeuft als Pod im selben Cluster wie die DB und verbindet via ClusterIP (kein
kubectl port-forward, kein Userspace-Proxy). Der Host-Runner hat vorher
insert + build_index gemacht; dieser Prozess macht NUR den Query-Loop und misst
Latenz/Throughput/Recall -- so ist die gemessene Latenz die DB-Roundtrip-Zeit
aus dem Cluster heraus, nicht Python+port-forward auf dem Host.

Aufruf (im Pod):
    measure.py --config /data/cfg/<name>.json \
               --data-dir /data/<stufe> --out /data/results_incluster/<run>.json

Erwartet BENCH_IN_CLUSTER=1 in der Umgebung -> Adapter nutzen ClusterIP-DNS.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import get_adapter
from runner import load_queries, load_query_meta, detect_corpus_dim


def pct(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Pfad zur Config-JSON")
    ap.add_argument("--data-dir", required=True, help="Stufen-Verzeichnis (chunks + queries/)")
    ap.add_argument("--out", required=True, help="Output-JSON-Pfad")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    stufe_dir = Path(args.data_dir)
    dim = detect_corpus_dim(stufe_dir)
    if dim <= 0:
        raise SystemExit(f"keine corpus-chunks unter {stufe_dir}")

    queries, ground_truth, gt_file, gt_note = load_queries(stufe_dir, cfg)
    query_meta = load_query_meta(stufe_dir)
    query_texts = (query_meta or {}).get("review_text") if query_meta else None

    adapter = get_adapter(cfg["db"])(cfg, dim=dim)
    adapter.attach()

    try:
        n_query = min(cfg["queries"]["n"], queries.shape[0])
        concurrency = cfg["queries"]["concurrency"]
        k_max = 100
        workload = cfg.get("workload", "topk")
        filter_spec = cfg.get("filter", {})
        hybrid_alpha = cfg.get("hybrid", {}).get("alpha", 0.5)

        def do_query(i, v):
            if workload in ("topk", "batch"):
                return adapter.query(v, k_max)
            if workload == "filtered":
                return adapter.query_filtered(v, k_max, filter_spec)
            if workload == "hybrid":
                if not query_texts:
                    raise SystemExit("hybrid braucht queries.parquet mit review_text")
                return adapter.query_hybrid(v, query_texts[i], k_max, alpha=hybrid_alpha)
            raise ValueError(f"unbekannter workload: {workload}")

        def one(i):
            v = queries[i]
            t = time.perf_counter()
            retrieved = do_query(i, v)
            dt = (time.perf_counter() - t) * 1000.0
            truth = ground_truth[i]
            return (
                dt,
                adapter.recall_at_k(retrieved, truth, 1),
                adapter.recall_at_k(retrieved, truth, 10),
                adapter.recall_at_k(retrieved, truth, 100),
                adapter.precision_at_k(retrieved, truth, 10),
                adapter.ndcg_at_k(retrieved, truth, 10),
            )

        # Warmup: erste Queries fuellen DB-/OS-Cache; Timings werden verworfen,
        # damit die Messung nicht kalt-kontaminiert ist (Thesis 5.5.1). n_warmup
        # aus der Config, gedeckelt auf die verfuegbaren Queries. Sequentiell --
        # es geht nur ums Cache-Warmziehen, nicht um Zeiten.
        n_warmup = min(cfg["queries"].get("n_warmup", 1000), n_query)
        for i in range(n_warmup):
            do_query(i, queries[i])

        lat, r1, r10, r100, p10, nd = [], [], [], [], [], []
        t0 = time.time()
        if concurrency <= 1:
            for i in range(n_query):
                d, a, b, c, p, n = one(i)
                lat.append(d); r1.append(a); r10.append(b)
                r100.append(c); p10.append(p); nd.append(n)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                for d, a, b, c, p, n in ex.map(one, range(n_query)):
                    lat.append(d); r1.append(a); r10.append(b)
                    r100.append(c); p10.append(p); nd.append(n)
        wall = time.time() - t0

        metrics = {
            "throughput_qps": round(n_query / wall, 1) if wall > 0 else 0.0,
            "latency_ms_mean": round(statistics.mean(lat), 2),
            "latency_ms_p50": round(pct(lat, 50), 2),
            "latency_ms_p95": round(pct(lat, 95), 2),
            "latency_ms_p99": round(pct(lat, 99), 2),
            "recall_at_1": round(statistics.mean(r1), 4),
            "recall_at_10": round(statistics.mean(r10), 4),
            "recall_at_100": round(statistics.mean(r100), 4),
            "precision_at_10": round(statistics.mean(p10), 4),
            "ndcg_at_10": round(statistics.mean(nd), 4),
        }
        out = {
            "metrics": metrics,
            "measured": "in-cluster",
            "n_queries_executed": n_query,
            "n_warmup": n_warmup,
            "concurrency": concurrency,
            "gt_file": gt_file,
            "gt_note": gt_note,
            "wall_s": round(wall, 2),
        }
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2) + "\n")
        print("MEASURE_OK " + json.dumps(metrics), flush=True)
    finally:
        adapter.teardown()


if __name__ == "__main__":
    main()
