#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CLUSTER="${BENCH_CLUSTER:-dbbench-demo}"
CACHE="${BENCH_CACHE_DIR:-$HOME/.cache/bachelor-db-benchmark}"
PY="${BENCH_PY:-$HOME/bench-venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

for bin in k3d kubectl docker helm; do
  command -v "$bin" >/dev/null 2>&1 || { echo "fehlt: $bin"; exit 1; }
done
"$PY" -c "import psycopg, pyarrow, numpy, weaviate" 2>/dev/null || { echo "Python-Deps fehlen (psycopg/pyarrow/numpy/weaviate) in $PY"; exit 1; }

print_result() {
  local res
  res="$(ls -td "$ROOT"/results/*"$1"* 2>/dev/null | head -1)"
  [ -n "$res" ] && "$PY" -c "import json,sys; m=json.load(open(sys.argv[1]+'/summary.json'))['metrics']; print('  %-10s p50=%.2fms p95=%.2fms p99=%.2fms qps=%.1f recall@10=%.3f' % (sys.argv[2],m['latency_ms_p50'],m['latency_ms_p95'],m['latency_ms_p99'],m['throughput_qps'],m['recall_at_10']))" "$res" "$2"
}

echo "[1/6] k3d-Cluster '$CLUSTER'"
k3d cluster delete "$CLUSTER" >/dev/null 2>&1 || true
k3d cluster create "$CLUSTER" --agents 1 --wait

echo "[2/6] Synthetischer Demo-Korpus (5000 x 1024) + Queries"
rm -rf "$CACHE/DEMO"
"$PY" benchmarks/demodata/generate.py --output-dir "$CACHE/DEMO" --n-records 5000 --dim 1024 --seed 42
"$PY" benchmarks/demodata/gen_queries.py --corpus-dir "$CACHE/DEMO" --output-dir "$CACHE/DEMO/queries" --n-queries 500 --dim 1024 --top-k 100 --seed 42

echo "[3/6] pgvector deployen"
kubectl apply -f databases/pgvector/manifests/
kubectl -n db-pgvector rollout status statefulset/pgvector --timeout=300s

echo "[4/6] pgvector-Lauf"
cat > benchmarks/configs/pgvector-DEMO-latency.json <<'JSON'
{
  "name": "pgvector-DEMO-latency",
  "description": "Live-Demo pgvector HNSW, synthetischer Mini-Korpus",
  "db": "pgvector",
  "stufe": "DEMO",
  "dim": 1024,
  "variant": "A",
  "workload": "topk",
  "index": { "type": "hnsw", "params": { "m": 16, "ef_construction": 128, "ef_search": 64 } },
  "queries": { "n": 500, "concurrency": 1 },
  "pre_run_reset": false
}
JSON
"$PY" benchmarks/runners/runner.py --config pgvector-DEMO-latency --demodata-dir "$CACHE" --no-push

echo "[5/6] weaviate deployen + Lauf"
(
  set -e
  helm repo add weaviate https://weaviate.github.io/weaviate-helm >/dev/null 2>&1 || true
  helm repo update >/dev/null
  helm upgrade --install weaviate weaviate/weaviate \
    --namespace db-weaviate --create-namespace \
    --values databases/weaviate/values-demo.yaml --wait --timeout 6m
  kubectl -n db-weaviate rollout status statefulset/weaviate --timeout=300s
  cat > benchmarks/configs/weaviate-DEMO-latency.json <<'JSON'
{
  "name": "weaviate-DEMO-latency",
  "description": "Live-Demo weaviate HNSW, synthetischer Mini-Korpus",
  "db": "weaviate",
  "stufe": "DEMO",
  "dim": 1024,
  "variant": "A",
  "workload": "topk",
  "index": { "type": "hnsw", "params": { "ef": 64, "ef_construction": 128, "M": 16 } },
  "queries": { "n": 500, "concurrency": 1 },
  "pre_run_reset": false
}
JSON
  "$PY" benchmarks/runners/runner.py --config weaviate-DEMO-latency --demodata-dir "$CACHE" --no-push
) || echo "  weaviate-Block uebersprungen (pgvector-Ergebnis oben gilt)"

echo
echo "[6/6] Ergebnisse (Stufe DEMO, 5k x 1024, topk):"
print_result "pgvector-DEMO-latency" "pgvector"
print_result "weaviate-DEMO-latency" "weaviate"
echo
echo "Cluster aufraeumen: k3d cluster delete $CLUSTER"
