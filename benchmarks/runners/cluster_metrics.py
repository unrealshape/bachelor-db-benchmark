"""Cluster- und Ressourcen-Metriken für den Runner. Liest kubectl-Output
(Version, Nodes, `top pod`) und stellt einen ResourceSampler-Thread bereit,
der CPU- und Memory-Werte über die Lauf-Dauer mittelt."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass


# ----- statische Cluster-Infos --------------------------------------------

def cluster_info() -> dict:
    """k8s-Version + Anzahl Ready-Nodes. Fällt auf None zurück wenn kubectl
    nicht erreichbar ist."""
    info = {"k8s_version": None, "nodes": None}
    try:
        out = subprocess.check_output(
            ["kubectl", "version", "-o", "json"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        server = data.get("serverVersion", {}).get("gitVersion")
        if server:
            info["k8s_version"] = server
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "nodes", "-o", "json"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        info["nodes"] = len(data.get("items", []))
    except Exception:
        pass
    return info


# ----- Pod-Ressourcen ------------------------------------------------------

_CPU_RE = re.compile(r"^(\d+(?:\.\d+)?)([mun]?)$")
_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)(Ki|Mi|Gi)?$")


def _parse_cpu(s: str) -> float:
    """`8m` -> 0.008 cores, `2` -> 2.0 cores, `300n` -> 0.0000003. Robust
    gegen kubectl-Quirks bei sehr kleinen Werten."""
    s = s.strip()
    m = _CPU_RE.match(s)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    if unit == "m":
        return val / 1000.0
    if unit == "u":
        return val / 1_000_000.0
    if unit == "n":
        return val / 1_000_000_000.0
    return val


def _parse_mem_mb(s: str) -> float:
    """`336Mi` -> 336.0, `1Gi` -> 1024.0, `42Ki` -> 0.041. Default: bytes."""
    s = s.strip()
    m = _MEM_RE.match(s)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    if unit == "Gi":
        return val * 1024.0
    if unit == "Mi":
        return val
    if unit == "Ki":
        return val / 1024.0
    # bytes
    return val / (1024.0 * 1024.0)


def sample_pod(namespace: str, pod: str) -> tuple[float, float] | None:
    """Eine `kubectl top pod`-Probe. Gibt (cpu_cores, mem_mb) zurück oder
    None wenn metrics-server nicht antwortet."""
    try:
        out = subprocess.check_output(
            ["kubectl", "top", "pod", "-n", namespace, pod, "--no-headers"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    parts = out.split()
    if len(parts) < 3:
        return None
    # POD CPU MEM
    return _parse_cpu(parts[1]), _parse_mem_mb(parts[2])


def _kubectl_exec_cat(namespace: str, pod: str, path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["kubectl", "exec", "-n", namespace, pod, "--", "cat", path],
            text=True, timeout=8, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def sample_pod_io(namespace: str, pod: str) -> dict | None:
    """Kumulative Block-I/O-Zaehler des Pod-Containers aus dem cgroup.

    cgroup v2 (`io.stat`) bevorzugt, Fallback v1 (`blkio...io_service_bytes`).
    Summiert ueber alle Devices. `read_bytes` = echte Disk-Reads (Page-Cache-
    Misses) -> Paging-Signal fuer 'Index passt nicht mehr in RAM' (Thesis Kap 5,
    Speicherdruck-Achse). None wenn nicht lesbar (kein io-Controller delegiert,
    kein shell-loser Container etc.) -- dann bleiben die Disk-Felder None.
    """
    out = _kubectl_exec_cat(namespace, pod, "/sys/fs/cgroup/io.stat")
    if out:
        rb = wb = ri = wi = 0
        found = False
        for line in out.splitlines():
            for tok in line.split():
                if tok.startswith("rbytes="):
                    rb += int(tok[7:]); found = True
                elif tok.startswith("wbytes="):
                    wb += int(tok[7:])
                elif tok.startswith("rios="):
                    ri += int(tok[5:])
                elif tok.startswith("wios="):
                    wi += int(tok[5:])
        if found:
            return {"read_bytes": rb, "write_bytes": wb,
                    "read_ios": ri, "write_ios": wi}
    out = _kubectl_exec_cat(
        namespace, pod, "/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes")
    if out:
        rb = wb = 0
        found = False
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("Read", "Write"):
                found = True
                if parts[1] == "Read":
                    rb += int(parts[2])
                else:
                    wb += int(parts[2])
        if found:
            return {"read_bytes": rb, "write_bytes": wb,
                    "read_ios": None, "write_ios": None}
    return None


@dataclass
class ResourceAverages:
    cpu_avg_cores: float | None
    mem_avg_mb: float | None
    cpu_peak_cores: float | None
    mem_peak_mb: float | None
    n_samples: int
    # Disk-I/O ueber die Lauf-Dauer (Delta der kumulativen cgroup-Zaehler).
    # read_mb = echte Disk-Reads -> Paging-Beleg. None wenn nicht ermittelbar.
    disk_read_mb: float | None = None
    disk_write_mb: float | None = None
    disk_read_ios: int | None = None
    disk_write_ios: int | None = None


class ResourceSampler:
    """Hintergrund-Thread der periodisch `kubectl top pod` aufruft. start()
    beginnt das Sampling, stop() liefert die gemittelten Werte."""

    def __init__(self, namespace: str, pod: str, interval_s: float = 2.0):
        self.namespace = namespace
        self.pod = pod
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu: list[float] = []
        self._mem: list[float] = []
        # Disk-I/O: nur an den Lauf-Grenzen gelesen (2 execs ausserhalb des
        # Mess-Fensters), damit der Mess-Pod nicht durch periodisches exec
        # verfaelscht wird. Counters sind kumulativ -> Delta = Lauf-I/O.
        self._io_start: dict | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = sample_pod(self.namespace, self.pod)
            if s is not None:
                self._cpu.append(s[0])
                self._mem.append(s[1])
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._io_start = sample_pod_io(self.namespace, self.pod)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _disk_delta(self) -> dict:
        io_stop = sample_pod_io(self.namespace, self.pod)
        if not self._io_start or not io_stop:
            return {}
        d = {}
        rb = io_stop["read_bytes"] - self._io_start["read_bytes"]
        wb = io_stop["write_bytes"] - self._io_start["write_bytes"]
        d["disk_read_mb"] = round(max(0, rb) / (1024 * 1024), 1)
        d["disk_write_mb"] = round(max(0, wb) / (1024 * 1024), 1)
        if io_stop.get("read_ios") is not None and self._io_start.get("read_ios") is not None:
            d["disk_read_ios"] = max(0, io_stop["read_ios"] - self._io_start["read_ios"])
            d["disk_write_ios"] = max(0, io_stop["write_ios"] - self._io_start["write_ios"])
        return d

    def stop(self) -> ResourceAverages:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        disk = self._disk_delta()
        if not self._cpu:
            return ResourceAverages(None, None, None, None, 0, **disk)
        return ResourceAverages(
            cpu_avg_cores=round(statistics.mean(self._cpu), 4),
            mem_avg_mb=round(statistics.mean(self._mem), 1),
            cpu_peak_cores=round(max(self._cpu), 4),
            mem_peak_mb=round(max(self._mem), 1),
            n_samples=len(self._cpu),
            **disk,
        )
