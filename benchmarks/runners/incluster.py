"""Host-seitige Orchestrierung des In-Cluster-Mess-Pods.

Der Host-Runner macht insert + build_index (via port-forward). Danach ruft er
run_incluster_measure() auf: rendert einen k8s-Job mit dem schlanken
bench-measure-Image, der via ClusterIP den Query-Loop fährt und Latenz/
Throughput/Recall server-nah misst (kein port-forward im Mess-Pfad). Output
kommt über das gemeinsame hostPath-Volume (~/.cache == /data) zurück.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

# ~/.cache/bachelor-db-benchmark ist via k3d --volume als /data in den Nodes
# gemountet; der Pod hostPath-mountet /data.
HOST_CACHE = Path.home() / ".cache" / "bachelor-db-benchmark"
POD_DATA = "/data"
IMAGE = "bench-measure:1"
NAMESPACE = "default"


def _san(s: str) -> str:
    s = re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")
    return (s[:48] or "run").strip("-")


def _render_job(job_name: str, cfg_name: str, stufe: str, out_name: str) -> str:
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {NAMESPACE}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  activeDeadlineSeconds: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: measure
          image: {IMAGE}
          imagePullPolicy: IfNotPresent
          args:
            - "--config"
            - "{POD_DATA}/cfg/{cfg_name}.json"
            - "--data-dir"
            - "{POD_DATA}/{stufe}"
            - "--out"
            - "{POD_DATA}/results_incluster/{out_name}"
          volumeMounts:
            - name: data
              mountPath: {POD_DATA}
      volumes:
        - name: data
          hostPath:
            path: {POD_DATA}
            type: Directory
"""


def _kubectl(args: list[str], **kw):
    return subprocess.run(["kubectl", "-n", NAMESPACE, *args], **kw)


def run_incluster_measure(cfg: dict, run_id: str, stufe: str,
                          timeout_s: int = 3600) -> dict:
    (HOST_CACHE / "cfg").mkdir(parents=True, exist_ok=True)
    (HOST_CACHE / "results_incluster").mkdir(parents=True, exist_ok=True)
    (HOST_CACHE / "cfg" / f"{cfg['name']}.json").write_text(json.dumps(cfg))

    out_name = f"{run_id}.json"
    out_host = HOST_CACHE / "results_incluster" / out_name
    if out_host.exists():
        out_host.unlink()

    job_name = "m-" + _san(run_id)
    _kubectl(["delete", "job", job_name, "--ignore-not-found"],
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    yaml = _render_job(job_name, cfg["name"], stufe, out_name)
    ap = subprocess.run(["kubectl", "apply", "-f", "-"], input=yaml,
                        text=True, capture_output=True)
    if ap.returncode != 0:
        raise RuntimeError(f"Job-apply fehlgeschlagen: {ap.stderr.strip()}")

    deadline = time.time() + timeout_s
    status = None
    while time.time() < deadline:
        r = _kubectl(["get", "job", job_name, "-o", "json"],
                     capture_output=True, text=True)
        if r.returncode == 0:
            st = json.loads(r.stdout).get("status", {})
            if st.get("succeeded", 0) >= 1:
                status = "complete"
                break
            if st.get("failed", 0) >= 1:
                status = "failed"
                break
        time.sleep(2)

    logs = _kubectl(["logs", f"job/{job_name}", "--tail=30"],
                    capture_output=True, text=True).stdout

    if status != "complete":
        raise RuntimeError(
            f"In-Cluster-Mess-Job {status or 'timeout'}.\nLogs:\n{logs}")
    if not out_host.exists():
        raise RuntimeError(f"kein Output {out_host}.\nLogs:\n{logs}")

    out = json.loads(out_host.read_text())
    _kubectl(["delete", "job", job_name, "--ignore-not-found"],
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out
