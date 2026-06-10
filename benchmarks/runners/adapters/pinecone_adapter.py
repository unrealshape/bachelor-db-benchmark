"""Pinecone-Adapter. Spricht den Managed-Service ueber die offizielle
`pinecone` Python-SDK (v3+ / v5+, Pod-basiert) an.

Bewusst Pod-Tier `s1.x1` statt Serverless, damit Pinecone hardware-aequivalent
zu den self-hosted DBs ist (Thesis 5: Vergleich auf vergleichbarer Klasse).

Latenz-Erfassung:
    - `client_latency_ms` -- Wall-Clock vom Adapter aus gemessen, inkl.
      Netz-Hop zur Pinecone-Cloud.
    - `server_latency_ms` -- aus dem Antwort-Header
      `x-pinecone-request-latency-ms`, ohne Netz.
Beide Werte werden pro Query gesammelt und in `notes` ausgewiesen, damit die
Auswertung den Netz-Aufschlag (Pinecone vs. self-hosted) sichtbar macht.

Achtung: Der eigentliche Runner serialisiert nur die ueblichen Latenz-
Metriken; der Adapter haelt die Server-Zeiten zusaetzlich in
`self.server_latencies_ms` bzw. liefert sie ueber `last_server_latency_ms`
zurueck, damit der Runner sie spaeter aufnehmen kann ohne dass diese Datei
das Runner-Interface bricht.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress

import numpy as np

from .base import Adapter


# Defaults, falls die Config nichts vorgibt.
DEFAULT_POD_TYPE = "s1.x1"
DEFAULT_PODS = 1
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_METRIC = "cosine"

# Antwort-Header, in dem Pinecone die Server-Zeit (ohne Netz) liefert.
SERVER_LATENCY_HEADER = "x-pinecone-request-latency-ms"


def _split_region(region: str) -> tuple[str, str]:
    """Akzeptiert sowohl 'aws-us-east-1' (alt) als auch 'us-east-1' (neu).
    Liefert (cloud, region)."""
    region = region.strip()
    for cloud in ("aws", "gcp", "azure"):
        if region.startswith(cloud + "-"):
            return cloud, region[len(cloud) + 1:]
    return DEFAULT_CLOUD, region


class PineconeAdapter(Adapter):
    """Pinecone Pod-Tier Adapter.

    Anders als bei Weaviate/pgvector gibt es keine kubectl-port-forwards: der
    Index lebt in der Pinecone-Cloud. Auch `index_size_mb` ist nicht direkt
    messbar -- Pinecone exponiert das nicht. Wir geben None zurueck und
    notieren die Pod-Konfig stattdessen in `notes`.
    """

    db_name = "pinecone"

    def __init__(self, cfg: dict, dim: int) -> None:
        super().__init__(cfg, dim)
        self._pc = None
        self._index = None
        self._index_name: str = cfg.get("index_name") or cfg["name"]
        self._build_s: float | None = None
        # Pod-Konfig aus index.params, mit env-Fallbacks fuer Region/Cloud.
        p = self.index_params
        self._pod_type: str = p.get("pod_type", DEFAULT_POD_TYPE)
        self._pods: int = int(p.get("pods", DEFAULT_PODS))
        # `region` in der Config kann "aws-us-east-1" oder "us-east-1" sein.
        cloud_cfg = p.get("cloud")
        region_cfg = p.get("region") or os.environ.get("PINECONE_REGION", DEFAULT_REGION)
        if cloud_cfg:
            self._cloud = cloud_cfg
            self._region = region_cfg
        else:
            self._cloud, self._region = _split_region(region_cfg)
        # Pinecone-Umgebungs-String fuer Pod-Spec: "<region>-<cloud>" (legacy).
        self._environment: str = p.get(
            "environment", f"{self._region}-{self._cloud}"
        )
        self._metric: str = p.get("metric", DEFAULT_METRIC)
        # Header-Latenz: thread-local Speicher, damit parallele Queries
        # (concurrency > 1) sich nicht ueberschreiben.
        self._tls = threading.local()
        # Aggregat aller Server-Latenzen; der Runner kann das nach dem Lauf
        # abgreifen (z.B. ueber adapter.server_latencies_ms).
        self.server_latencies_ms: list[float] = []
        self._server_lat_lock = threading.Lock()

    # ---- Header-Hook ------------------------------------------------------

    def _install_header_hook(self) -> None:
        """Setzt einen response-Hook auf die requests-Session der Pinecone-
        REST-Connection, damit jeder query-Response den Header
        `x-pinecone-request-latency-ms` in thread-local Storage stellt.

        Pinecone SDK v3+ verwendet intern `urllib3` ueber den generierten
        api_client. Wir hangeln uns defensiv durch das Objekt: wenn die
        interne Struktur sich aendert, geht der Hook leise verloren, das
        Wall-Clock-Messung bleibt aber unberuehrt.
        """
        idx = self._index
        # Drei Pfade die in verschiedenen SDK-Versionen vorkommen:
        candidates = []
        with suppress(Exception):
            candidates.append(getattr(idx, "_vector_api", None))
        with suppress(Exception):
            candidates.append(getattr(idx, "_api_client", None))
        with suppress(Exception):
            inner = getattr(idx, "_vector_api", None)
            if inner is not None:
                candidates.append(getattr(inner, "api_client", None))

        adapter = self

        def _capture(headers) -> None:
            try:
                v = headers.get(SERVER_LATENCY_HEADER)
                if v is None:
                    return
                ms = float(v)
            except (TypeError, ValueError):
                return
            adapter._tls.last_server_ms = ms

        for cand in candidates:
            if cand is None:
                continue
            # urllib3-PoolManager hat keinen Hook-Mechanismus wie requests.
            # Wir monkey-patchen die Methode, die letztlich die Response
            # zurueckgibt. Falls schon gepatcht: ueberspringen.
            for attr in ("rest_client", "rest_client_instance"):
                rest = getattr(cand, attr, None)
                if rest is None:
                    continue
                if getattr(rest, "_bench_hooked", False):
                    return
                orig = getattr(rest, "request", None)
                if orig is None:
                    continue

                def wrapped(*args, _orig=orig, **kwargs):
                    resp = _orig(*args, **kwargs)
                    try:
                        # urllib3-Response hat .getheaders() bzw. .headers
                        headers = getattr(resp, "headers", None)
                        if headers is not None:
                            _capture(headers)
                    except Exception:
                        pass
                    return resp

                rest.request = wrapped
                rest._bench_hooked = True
                return

    def _pop_server_latency(self) -> float | None:
        v = getattr(self._tls, "last_server_ms", None)
        if v is not None:
            self._tls.last_server_ms = None
        return v

    # ---- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        from pinecone import Pinecone, PodSpec

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY ist nicht gesetzt. "
                "Pinecone-Runs brauchen einen API-Key in der Umgebung."
            )

        self._pc = Pinecone(api_key=api_key)

        # Idempotent: alten Index wegraeumen, neu anlegen.
        existing = {ix.name for ix in self._pc.list_indexes()}
        if self._index_name in existing:
            self._pc.delete_index(self._index_name)
            # Pinecone braucht ein paar Sekunden bis das wirklich weg ist.
            deadline = time.time() + 120
            while time.time() < deadline:
                if self._index_name not in {ix.name for ix in self._pc.list_indexes()}:
                    break
                time.sleep(2.0)

        spec = PodSpec(
            environment=self._environment,
            pod_type=self._pod_type,
            pods=self._pods,
        )
        self._pc.create_index(
            name=self._index_name,
            dimension=self.dim,
            metric=self._metric,
            spec=spec,
        )

        # Auf "ready" warten: Pod-Indizes brauchen ~30-90s bis sie ready sind.
        deadline = time.time() + 600
        while time.time() < deadline:
            desc = self._pc.describe_index(self._index_name)
            status = getattr(desc, "status", None) or {}
            ready = bool(getattr(status, "ready", None)
                          if not isinstance(status, dict)
                          else status.get("ready"))
            if ready:
                break
            time.sleep(3.0)
        else:
            raise RuntimeError(
                f"pinecone-index {self._index_name} ist nach 10min nicht ready"
            )

        self._index = self._pc.Index(self._index_name)
        self._install_header_hook()

    # ---- inserts ----------------------------------------------------------

    def insert(self, ids: np.ndarray, vecs: np.ndarray,
                metadata: dict | None = None) -> None:
        """Upsert in Batches a 100 Vektoren (Pinecone-Empfehlung). Variante B
        unterstuetzen wir hier nicht: Pinecone hat keine separate Metadaten-
        Collection, die getrennt joinbar waere. Variante A: Metadaten landen
        als Filter-Felder am Vektor.
        """
        if self.variant == "B":
            raise NotImplementedError(
                "pinecone unterstuetzt keine getrennte Metadaten-Collection -- "
                "Variante B ist hier nicht definiert."
            )

        n = len(ids)
        BATCH = 100
        ids_list = ids.tolist()

        def _meta_row(j: int) -> dict | None:
            if metadata is None:
                return None
            row: dict = {}
            for c in ("product_id", "product_title", "user_id",
                       "rating", "review_text", "timestamp"):
                col = metadata.get(c)
                if col is None or j >= len(col):
                    continue
                v = col[j]
                if v is None:
                    continue
                # Pinecone-Metadaten erlauben str/number/bool/list[str].
                if c == "rating":
                    with suppress(Exception):
                        row[c] = float(v)
                else:
                    row[c] = str(v)
            return row or None

        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            batch = []
            for j in range(start, end):
                rec = {
                    "id": str(int(ids_list[j])),
                    "values": vecs[j].tolist(),
                }
                meta = _meta_row(j)
                if meta is not None:
                    rec["metadata"] = meta
                batch.append(rec)
            self._index.upsert(vectors=batch)

    def build_index(self, build_text_index: bool | None = None) -> float:
        """Pinecone baut den Index waehrend des Upserts; wir warten nur bis
        `vector_count` stabil ist und returnen die dafuer noetige Zeit.
        build_text_index ist ohne Wirkung (kein separater Text-Index).
        Pinecone braucht typischerweise ein paar Sekunden bis die Vektoren
        nach dem letzten Upsert tatsaechlich queryable sind.
        """
        t0 = time.time()
        deadline = t0 + 600
        last_count = -1
        stable_since: float | None = None
        while time.time() < deadline:
            try:
                stats = self._index.describe_index_stats()
                count = int(
                    getattr(stats, "total_vector_count", None)
                    if not isinstance(stats, dict)
                    else stats.get("total_vector_count", 0)
                )
            except Exception:
                count = -1
            if count > 0 and count == last_count:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= 5.0:
                    break
            else:
                stable_since = None
                last_count = count
            time.sleep(1.0)
        self._build_s = time.time() - t0
        return self._build_s

    # ---- queries ----------------------------------------------------------

    def query(self, vec: np.ndarray, k: int) -> list[int]:
        """Top-k Similarity Search. Misst zusaetzlich die Server-Zeit aus
        dem Antwort-Header und legt sie in `self.server_latencies_ms` ab.
        Wall-Clock laeuft beim Runner.
        """
        t0 = time.perf_counter()
        res = self._index.query(
            vector=vec.tolist(),
            top_k=k,
            include_values=False,
            include_metadata=False,
        )
        client_ms = (time.perf_counter() - t0) * 1000.0

        server_ms = self._pop_server_latency()
        if server_ms is not None:
            with self._server_lat_lock:
                self.server_latencies_ms.append(server_ms)
        # Schoenen Seiteneffekt: client-Latenz als Attribut, damit Tests
        # daran ablesen koennen ohne den Runner zu kennen.
        self.last_client_latency_ms = client_ms
        self.last_server_latency_ms = server_ms

        matches = (
            res.get("matches", []) if isinstance(res, dict)
            else getattr(res, "matches", [])
        )
        out: list[int] = []
        for m in matches:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            if mid is None:
                continue
            with suppress(ValueError, TypeError):
                out.append(int(mid))
        return out

    def query_filtered(self, vec: np.ndarray, k: int,
                        filters: dict) -> list[int]:
        """Pinecone hat ein natives Filter-DSL: `{"rating": {"$gte": 4}}`."""
        pc_filter: dict = {}
        if "rating_gte" in filters:
            pc_filter["rating"] = {"$gte": float(filters["rating_gte"])}
        if "product_id" in filters:
            pc_filter["product_id"] = {"$eq": str(filters["product_id"])}

        t0 = time.perf_counter()
        res = self._index.query(
            vector=vec.tolist(),
            top_k=k,
            filter=pc_filter or None,
            include_values=False,
            include_metadata=False,
        )
        self.last_client_latency_ms = (time.perf_counter() - t0) * 1000.0
        server_ms = self._pop_server_latency()
        if server_ms is not None:
            with self._server_lat_lock:
                self.server_latencies_ms.append(server_ms)
        self.last_server_latency_ms = server_ms

        matches = (
            res.get("matches", []) if isinstance(res, dict)
            else getattr(res, "matches", [])
        )
        out: list[int] = []
        for m in matches:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            if mid is None:
                continue
            with suppress(ValueError, TypeError):
                out.append(int(mid))
        return out

    # query_hybrid: Pinecone Pod-Tier unterstuetzt Sparse-Dense Hybrid nur
    # bei bestimmten Pod-Typen (s1, p1 mit sparse-Vektor). Wir lassen den
    # Default aus base.Adapter: NotImplementedError, der Runner faengt das ab.

    # ---- meta -------------------------------------------------------------

    def index_size_mb(self) -> float | None:
        """Pinecone exponiert keine Index-Groesse auf Disk -- managed Cloud.
        Stattdessen geben wir None zurueck und notieren die Pod-Konfig in
        den Run-Notes (siehe `notes` unten)."""
        return None

    def server_latency_summary(self) -> dict | None:
        """Aggregat der Server-Header-Latenzen ueber den ganzen Lauf.
        Der Runner kann das nach `query()`-Loop abrufen und in `summary.json`
        unter `notes.server_latency_ms_*` ablegen.
        """
        xs = list(self.server_latencies_ms)
        if not xs:
            return None
        xs_sorted = sorted(xs)
        n = len(xs_sorted)

        def pct(p: float) -> float:
            k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return round(xs_sorted[k], 2)

        return {
            "samples": n,
            "mean": round(sum(xs_sorted) / n, 2),
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
        }

    @property
    def notes(self) -> dict:
        """Lauf-Notizen die der Runner in `summary.json.notes` aufnehmen kann."""
        return {
            "pod_type": self._pod_type,
            "pods": self._pods,
            "cloud": self._cloud,
            "region": self._region,
            "environment": self._environment,
            "metric": self._metric,
            "managed": True,
            "server_latency_header": SERVER_LATENCY_HEADER,
        }

    def teardown(self) -> None:
        """Index am Ende loeschen, damit die Rechnung nicht weiterlaeuft.
        Per Env-Var `PINECONE_KEEP_INDEX=1` deaktivierbar (Debug)."""
        if self._pc is None or self._index_name is None:
            return
        if os.environ.get("PINECONE_KEEP_INDEX") == "1":
            return
        with suppress(Exception):
            self._pc.delete_index(self._index_name)
