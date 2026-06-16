"""Weaviate-Adapter. Spricht den Cluster über kubectl port-forward an.

Schema (Thesis 3.1):
    Variante A:  Collection "Bench" mit Embedding + allen Metadaten + BM25
                 auf review_text.
    Variante B:  Collection "BenchVecs" (Embedding + ext_id) und Collection
                 "BenchMeta" (Metadaten + ext_id). Query verknüpft über ext_id.
"""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import suppress

import numpy as np

from .base import Adapter


COLLECTION = "Bench"           # Variante A
COLLECTION_VECS = "BenchVecs"  # Variante B -- nur Vektor + ext_id
COLLECTION_META = "BenchMeta"  # Variante B -- nur Metadaten + ext_id

WEAVIATE_NAMESPACE = "db-weaviate"
WEAVIATE_HTTP_SERVICE = "weaviate"
WEAVIATE_GRPC_SERVICE = "weaviate-grpc"
WEAVIATE_HTTP_PORT = 80
WEAVIATE_GRPC_PORT = 50051

# Prometheus-Metrics-Endpoint (PROMETHEUS_MONITORING_ENABLED in values.yaml).
# Server-seitige Query-Latenz: Histogramm `queries_durations_ms` (Labels
# class_name, query_type). In-Cluster ueber Pod-DNS direkt am Container-Port
# erreichbar (keine Service-Port-Abhaengigkeit). Per env ueberschreibbar.
WEAVIATE_METRICS_PORT = int(os.environ.get("WEAVIATE_METRICS_PORT", "2112"))
WEAVIATE_METRICS_HOST_INCLUSTER = os.environ.get(
    "WEAVIATE_METRICS_HOST",
    f"weaviate-0.weaviate-headless.{WEAVIATE_NAMESPACE}.svc.cluster.local")
QUERY_DURATION_METRIC = "queries_durations_ms"


def _parse_query_durations(text: str, class_name: str) -> dict | None:
    """Parst das `queries_durations_ms`-Histogramm aus dem /metrics-Text der
    gegebenen class_name, **getrennt nach query_type**. Liefert
    {query_type: {sum, count, buckets:{le->count}}} oder None, wenn die Metrik fehlt.

    Die Trennung nach query_type ist wichtig: ueber das Mess-Fenster koennen neben
    der gemessenen Workload (z.B. get_graphql) auch andere/langsamere Query-Typen
    laufen; ein Aggregat ueber alle wuerde die Perzentile verfaelschen. server_latency_
    summary waehlt den dominanten query_type (groesster Delta-Count). Zeilen ohne
    class_name-Label zaehlen als Fallback mit; Zeilen anderer class_name werden uebersprungen."""
    import re
    cls_re = re.compile(r'class_name="([^"]*)"')
    qt_re = re.compile(r'query_type="([^"]*)"')
    le_re = re.compile(r'le="([^"]*)"')
    out: dict[str, dict] = {}
    found = False

    for line in text.splitlines():
        if not line or line[0] == "#" or not line.startswith(QUERY_DURATION_METRIC):
            continue
        head = line.split("{", 1)[0] if "{" in line else line.split(" ", 1)[0]
        suffix = head[len(QUERY_DURATION_METRIC):]  # "_sum" | "_count" | "_bucket" | ""
        m = cls_re.search(line)
        if m and m.group(1) != class_name:
            continue
        qt_m = qt_re.search(line)
        qt = qt_m.group(1) if qt_m else ""
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        d = out.setdefault(qt, {"sum": 0.0, "count": 0.0, "buckets": {}})
        if suffix == "_sum":
            d["sum"] += val; found = True
        elif suffix == "_count":
            d["count"] += val; found = True
        elif suffix == "_bucket":
            le = le_re.search(line)
            if le:
                key = float("inf") if le.group(1) in ("+Inf", "Inf") else float(le.group(1))
                d["buckets"][key] = d["buckets"].get(key, 0.0) + val
                found = True
    return out if found else None


def _fetch_metrics_text(host: str, port: int, timeout_s: float = 5.0) -> str | None:
    """GET http://host:port/metrics -> Body oder None bei Fehler/Non-200."""
    import urllib.request
    import urllib.error
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            if r.status != 200:
                return None
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _histogram_quantile(buckets: dict, q: float) -> float | None:
    """Prometheus-style histogram_quantile ueber kumulative {le->count}-Buckets
    (bereits als Delta ueber das Mess-Fenster uebergeben). Lineare Interpolation
    innerhalb des Treffer-Buckets. q in [0,1]. None wenn leer/degeneriert."""
    if not buckets:
        return None
    items = sorted(buckets.items(), key=lambda kv: kv[0])  # +Inf sortiert zuletzt
    total = items[-1][1]  # kumulativer Count beim hoechsten le (= +Inf)
    if total <= 0:
        return None
    rank = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, cum in items:
        if cum >= rank:
            if le == float("inf"):
                return prev_le if prev_le > 0 else None
            if cum <= prev_count:
                return le
            frac = (rank - prev_count) / (cum - prev_count)
            return prev_le + frac * (le - prev_le)
        if le != float("inf"):
            prev_le = le
        prev_count = cum
    return prev_le or None


def _port_forward(local_http: int, local_grpc: int) -> list[subprocess.Popen]:
    procs = []
    for svc, lport, rport in (
        (WEAVIATE_HTTP_SERVICE, local_http, WEAVIATE_HTTP_PORT),
        (WEAVIATE_GRPC_SERVICE, local_grpc, WEAVIATE_GRPC_PORT),
    ):
        cmd = [
            "kubectl", "port-forward",
            "-n", WEAVIATE_NAMESPACE,
            f"svc/{svc}",
            f"{lport}:{rport}",
        ]
        procs.append(subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        ))
    return procs


def _wait_for(host: str, port: int, timeout_s: float = 15.0) -> None:
    import socket
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"timeout waiting for {host}:{port}")


def _wait_http_ready(host: str, port: int, timeout_s: float = 60.0) -> None:
    """Pollt /v1/.well-known/ready bis Weaviate echte 200er gibt -- Pod-Restart
    braucht ~5-15s bis HNSW-Engine bereit ist."""
    import urllib.request, urllib.error
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/.well-known/ready"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"weaviate http nicht ready an {host}:{port}")


def _meta_properties():
    """Property-Definitionen für die Amazon-Reviews-Metadaten."""
    from weaviate.classes.config import DataType, Property
    return [
        Property(name="ext_id", data_type=DataType.INT),
        Property(name="product_id", data_type=DataType.TEXT),
        Property(name="product_title", data_type=DataType.TEXT),
        Property(name="user_id", data_type=DataType.TEXT),
        Property(name="rating", data_type=DataType.NUMBER),
        Property(name="review_text", data_type=DataType.TEXT),
        Property(name="timestamp", data_type=DataType.TEXT),
    ]


class WeaviateAdapter(Adapter):
    db_name = "weaviate"

    def __init__(self, cfg: dict, dim: int) -> None:
        super().__init__(cfg, dim)
        self._pfs: list[subprocess.Popen] = []
        self._client = None
        self._coll = None       # Variante A: Bench
        self._coll_vecs = None  # Variante B: BenchVecs
        self._coll_meta = None  # Variante B: BenchMeta
        self._local_http = int(os.environ.get("WEAVIATE_LOCAL_HTTP", "8080"))
        self._local_grpc = int(os.environ.get("WEAVIATE_LOCAL_GRPC", "50051"))
        self._build_s: float | None = None
        # In-Cluster-Modus: Mess-Pod verbindet via ClusterIP-DNS, kein port-forward.
        self._in_cluster = os.environ.get("BENCH_IN_CLUSTER") == "1"
        # Vorher-Snapshot des queries_durations_ms-Histogramms (begin_server_metrics).
        self._srv_snapshot: dict | None = None
        # Build/Query-Cache (in setup() aus index.params gesetzt).
        self._build_cache: int | None = None
        self._query_cache: int | None = None

    def _connect_client(self):
        import weaviate
        # Nach einem pre-run reset (rollout restart) kann der gRPC-Health-Check beim
        # Connect fehlschlagen, obwohl /v1/.well-known/ready (HTTP) schon 200 gibt --
        # gRPC kommt minimal spaeter hoch. Bis ~60s retryen.
        last_exc = None
        deadline = time.time() + 60.0
        while time.time() < deadline:
            try:
                if self._in_cluster:
                    http_host = os.environ.get(
                        "WEAVIATE_HTTP_HOST",
                        f"{WEAVIATE_HTTP_SERVICE}.{WEAVIATE_NAMESPACE}.svc.cluster.local")
                    grpc_host = os.environ.get(
                        "WEAVIATE_GRPC_HOST",
                        f"{WEAVIATE_GRPC_SERVICE}.{WEAVIATE_NAMESPACE}.svc.cluster.local")
                    return weaviate.connect_to_custom(
                        http_host=http_host, http_port=WEAVIATE_HTTP_PORT, http_secure=False,
                        grpc_host=grpc_host, grpc_port=WEAVIATE_GRPC_PORT, grpc_secure=False,
                    )
                return weaviate.connect_to_local(
                    host="127.0.0.1", port=self._local_http, grpc_port=self._local_grpc,
                )
            except Exception as e:
                last_exc = e
                time.sleep(2.0)
        raise last_exc

    def attach(self) -> None:
        """Verbindet zu bereits befuellten Collections ohne Drop/Create
        (Mess-Pod im Cluster). Setzt nur die Collection-Handles."""
        if not self._in_cluster and os.environ.get("WEAVIATE_SKIP_PORTFORWARD") != "1":
            self._pfs = _port_forward(self._local_http, self._local_grpc)
            _wait_for("127.0.0.1", self._local_http)
            _wait_for("127.0.0.1", self._local_grpc)
            _wait_http_ready("127.0.0.1", self._local_http)
        self._client = self._connect_client()
        if self.variant == "B":
            self._coll_vecs = self._client.collections.get(COLLECTION_VECS)
            self._coll_meta = self._client.collections.get(COLLECTION_META)
        else:
            self._coll = self._client.collections.get(COLLECTION)

    # ---- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        import weaviate
        from weaviate.classes.config import (
            Configure, DataType, Property, VectorDistances,
        )

        if not self._in_cluster and os.environ.get("WEAVIATE_SKIP_PORTFORWARD") != "1":
            self._pfs = _port_forward(self._local_http, self._local_grpc)
            _wait_for("127.0.0.1", self._local_http)
            _wait_for("127.0.0.1", self._local_grpc)
            _wait_http_ready("127.0.0.1", self._local_http)

        self._client = self._connect_client()

        # Idempotent: alle alten Collections wegräumen.
        for name in (COLLECTION, COLLECTION_VECS, COLLECTION_META):
            with suppress(Exception):
                self._client.collections.delete(name)

        p = self.index_params
        # Build/Query-RAM-Entkopplung (Thesis Kap 5): der HNSW-Graph wird mit
        # vectorCacheMaxObjects=BUILD (gross genug, dass die Stufe in den Cache passt,
        # z.B. 3M @S) auf einem Build-Pod mit genug RAM (build_mem_gb) gebaut -> kein
        # Disk-Thrash. Nach dem Build senkt build_index() den Cache auf den QUERY-Wert
        # (Tier, z.B. 1M ≈ 8-GiB-Budget); danach resized der Runner den Pod auf das
        # Serving-Budget (Pinecone s1.x1) -> ab S spillt der Cache von Disk = "Index
        # waechst aus RAM" wird an der Query-Seite gemessen. weaviate-Storage ist
        # persistent -> der Pod-Resize zwischen Build und Query verliert keine Daten.
        self._build_cache = int(p.get("vector_cache_max_objects", 3_000_000))
        self._query_cache = int(p.get("query_vector_cache_max_objects", 1_000_000))
        hnsw = Configure.VectorIndex.hnsw(
            ef=p.get("ef", 64),
            ef_construction=p.get("ef_construction", 128),
            max_connections=p.get("M", 16),
            vector_cache_max_objects=self._build_cache,
            distance_metric=VectorDistances.COSINE,
        )

        if self.variant == "B":
            # Variante B: zwei separate Collections.
            self._client.collections.create(
                name=COLLECTION_VECS,
                properties=[Property(name="ext_id", data_type=DataType.INT)],
                vector_index_config=hnsw,
            )
            self._client.collections.create(
                name=COLLECTION_META,
                properties=_meta_properties(),
                # Keine HNSW-Konfig, BenchMeta hält keine Vektoren.
            )
            self._coll_vecs = self._client.collections.get(COLLECTION_VECS)
            self._coll_meta = self._client.collections.get(COLLECTION_META)
        else:
            # Variante A: eine Collection mit allem.
            self._client.collections.create(
                name=COLLECTION,
                properties=_meta_properties(),
                vector_index_config=hnsw,
            )
            self._coll = self._client.collections.get(COLLECTION)

    # ---- inserts ----------------------------------------------------------

    def insert(self, ids: np.ndarray, vecs: np.ndarray,
                metadata: dict | None = None) -> None:
        """Variante A: Vektor + Metadaten in einem Batch.
        Variante B: Vektor in BenchVecs, Metadaten in BenchMeta."""
        ids_list = ids.tolist()

        if self.variant == "B":
            # BenchVecs: nur ext_id + vector
            with self._coll_vecs.batch.fixed_size(batch_size=500) as batch:
                for ext_id, vec in zip(ids_list, vecs):
                    batch.add_object(
                        properties={"ext_id": int(ext_id)},
                        vector=vec.tolist(),
                    )
            if self._coll_vecs.batch.failed_objects:
                raise RuntimeError(
                    f"{len(self._coll_vecs.batch.failed_objects)} failed vec-inserts"
                )
            # BenchMeta: alle Properties, ohne Vektor
            if metadata is not None:
                with self._coll_meta.batch.fixed_size(batch_size=500) as batch:
                    for j, ext_id in enumerate(ids_list):
                        batch.add_object(properties=_props_dict(metadata, j, ext_id))
                if self._coll_meta.batch.failed_objects:
                    raise RuntimeError(
                        f"{len(self._coll_meta.batch.failed_objects)} failed meta-inserts"
                    )
            return

        # Variante A: alles zusammen in Bench
        with self._coll.batch.fixed_size(batch_size=500) as batch:
            for j, (ext_id, vec) in enumerate(zip(ids_list, vecs)):
                if metadata is not None:
                    props = _props_dict(metadata, j, ext_id)
                else:
                    props = {"ext_id": int(ext_id)}
                batch.add_object(properties=props, vector=vec.tolist())
        if self._coll.batch.failed_objects:
            raise RuntimeError(
                f"{len(self._coll.batch.failed_objects)} failed inserts"
            )

    def _vector_queue_len(self) -> int | None:
        """Summe der noch nicht in den HNSW-Index aufgenommenen Vektoren ueber
        alle Shards (ASYNC_INDEXING). None wenn nicht ermittelbar -> dann nicht
        blockieren. Bei synchronem Indexing immer 0."""
        cls = COLLECTION_VECS if self.variant == "B" else COLLECTION
        try:
            nodes = self._client.cluster.nodes(collection=cls, output="verbose")
        except Exception:
            return None
        total = 0
        found = False
        for n in nodes:
            for sh in (getattr(n, "shards", None) or []):
                # weaviate-client v4 Shard-Modell: snake_case vector_queue_length.
                q = getattr(sh, "vector_queue_length", None)
                if q is not None:
                    total += int(q)
                    found = True
        return total if found else None

    def build_index(self, build_text_index: bool | None = None) -> float:
        # build_text_index ignoriert: weaviate haelt den BM25-/inverted-Index
        # automatisch auf den Text-Properties (bei setup angelegt). Ein Ingest
        # bedient damit ohnehin alle Workloads inkl. hybrid.
        coll = self._coll_vecs if self.variant == "B" else self._coll
        t0 = time.time()
        target = len(coll)
        deadline = t0 + 600
        while time.time() < deadline:
            agg = coll.aggregate.over_all(total_count=True)
            if agg.total_count >= target:
                time.sleep(1.0)
                break
            time.sleep(0.5)
        # ASYNC_INDEXING: Objekte sind gespeichert, der HNSW-Index baut aber noch
        # im Hintergrund aus der Queue. Erst wenn die Queue leer ist, ist der Index
        # vollstaendig -> sonst misst der Measure einen halb gebauten Index (Recall
        # zu niedrig). Unter 8-GiB-Budget kann der gebatchte Build dauern -> grosszuegiges
        # Deadline. None (nicht ermittelbar / sync indexing) -> nicht blockieren.
        idx_deadline = time.time() + 8 * 3600
        stable_zero = 0
        while time.time() < idx_deadline:
            q = self._vector_queue_len()
            if q is None:
                break
            if q == 0:
                stable_zero += 1
                if stable_zero >= 3:
                    break
            else:
                stable_zero = 0
            time.sleep(5.0)
        self._build_s = time.time() - t0

        # Index ist vollstaendig gebaut. Jetzt den vectorCacheMaxObjects auf den
        # QUERY-Wert senken (Build/Query-Entkopplung): danach resized der Runner den
        # Pod auf das 8-GiB-Serving-Budget; beim Reload prefillt weaviate nur noch den
        # Query-Cache (passt in 8 GiB), der Rest spillt von Disk = Mess-Phaenomen.
        # Muss VOR dem Pod-Resize passieren, sonst OOMt der 8-GiB-Pod am Build-Cache-Prefill.
        if self._query_cache and self._query_cache != self._build_cache:
            from weaviate.classes.config import Reconfigure
            target_coll = self._coll_vecs if self.variant == "B" else self._coll
            with suppress(Exception):
                target_coll.config.update(
                    vector_index_config=Reconfigure.VectorIndex.hnsw(
                        vector_cache_max_objects=self._query_cache,
                    )
                )
        return self._build_s

    # ---- queries ----------------------------------------------------------

    def query(self, vec: np.ndarray, k: int) -> list[int]:
        from weaviate.classes.query import MetadataQuery
        coll = self._coll_vecs if self.variant == "B" else self._coll
        res = coll.query.near_vector(
            near_vector=vec.tolist(),
            limit=k,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["ext_id"],
        )
        return [int(o.properties["ext_id"]) for o in res.objects]

    def query_filtered(self, vec: np.ndarray, k: int, filters: dict) -> list[int]:
        """Variante A: Filter direkt im near_vector. Variante B: zwei-Stufen-
        Query -- erst Vektor-Top-K' (mit K' > K), dann nach ext_id in
        BenchMeta filtern, dann auf K trimmen."""
        from weaviate.classes.query import Filter, MetadataQuery

        wv_filter = _filter_from_dict(filters)

        if self.variant == "A":
            coll = self._coll
            res = coll.query.near_vector(
                near_vector=vec.tolist(),
                limit=k,
                filters=wv_filter,
                return_properties=["ext_id"],
            )
            return [int(o.properties["ext_id"]) for o in res.objects]

        # Variante B: overshoot, dann in BenchMeta filtern.
        oversample = k * 10
        vec_res = self._coll_vecs.query.near_vector(
            near_vector=vec.tolist(),
            limit=oversample,
            return_properties=["ext_id"],
        )
        candidate_ids = [int(o.properties["ext_id"]) for o in vec_res.objects]
        if not candidate_ids:
            return []
        # In BenchMeta filtern, Reihenfolge der Kandidaten beibehalten.
        f = Filter.by_property("ext_id").contains_any(candidate_ids)
        if wv_filter is not None:
            f = Filter.all_of([f, wv_filter])
        meta_res = self._coll_meta.query.fetch_objects(
            filters=f, limit=oversample, return_properties=["ext_id"],
        )
        allowed = {int(o.properties["ext_id"]) for o in meta_res.objects}
        ranked = [eid for eid in candidate_ids if eid in allowed][:k]
        return ranked

    def query_hybrid(self, vec: np.ndarray, text: str, k: int,
                      alpha: float = 0.5) -> list[int]:
        """Hybrid = BM25(review_text) + Vektor. Weaviate hat ein natives
        hybrid()-API mit alpha. Nur Variante A unterstützt das direkt --
        Variante B muss erst Texte aus BenchMeta holen."""
        from weaviate.classes.query import HybridFusion

        if self.variant == "B":
            # Variante B: vector-only Top-K' und dann text-rerank wäre die
            # saubere Variante. Wir machen den einfachen Fall: vector first,
            # dann in BenchMeta die Treffer nach BM25-Score sortieren.
            # Für die Thesis interessant: zeigt den Overhead. Hier vereinfacht
            # auf reine Vektor-Suche, weil Re-Rank logik gehört in die
            # Auswertung.
            return self.query(vec, k)

        # Ranked-Fusion = Reciprocal Rank Fusion (RANK_CONSTANT=60), passend zur
        # RRF-GT (gen_special_gt) und zur pgvector-RRF-SQL.
        coll = self._coll
        res = coll.query.hybrid(
            query=text,
            vector=vec.tolist(),
            alpha=alpha,
            fusion_type=HybridFusion.RANKED,
            query_properties=["review_text"],  # GT-BM25 nutzt nur review_text
            limit=k,
            return_properties=["ext_id"],
        )
        return [int(o.properties["ext_id"]) for o in res.objects]

    # ---- meta -------------------------------------------------------------

    def index_size_mb(self) -> float | None:
        try:
            cmd = [
                "kubectl", "exec", "-n", WEAVIATE_NAMESPACE,
                "weaviate-0", "--", "sh", "-c",
                "du -sb /var/lib/weaviate 2>/dev/null | awk '{print $1}'",
            ]
            out = subprocess.check_output(
                cmd, text=True, timeout=20, stderr=subprocess.DEVNULL,
            )
            bytes_ = int(out.strip().splitlines()[-1])
            return round(bytes_ / (1024 * 1024), 1)
        except Exception:
            return None

    # ---- server-seitige Latenz (Prometheus /metrics) ----------------------

    def _metrics_endpoint(self) -> tuple[str, int] | None:
        """Host:Port des Prometheus-/metrics-Endpoints. In-Cluster ueber Pod-DNS;
        lokal nur wenn WEAVIATE_METRICS_HOST_LOCAL gesetzt ist (eigenes
        port-forward auf 2112). Sonst None -> Server-Latenz wird uebersprungen."""
        if self._in_cluster:
            return (WEAVIATE_METRICS_HOST_INCLUSTER, WEAVIATE_METRICS_PORT)
        host = os.environ.get("WEAVIATE_METRICS_HOST_LOCAL")
        if host:
            return (host, WEAVIATE_METRICS_PORT)
        return None

    def _scrape_durations(self) -> dict | None:
        ep = self._metrics_endpoint()
        if ep is None:
            return None
        text = _fetch_metrics_text(*ep)
        if text is None:
            return None
        cls = COLLECTION_VECS if self.variant == "B" else COLLECTION
        return _parse_query_durations(text, cls)

    def begin_server_metrics(self) -> None:
        """Vorher-Snapshot des Histogramms -- nach Warmup, vor dem Mess-Fenster."""
        self._srv_snapshot = self._scrape_durations()

    def server_latency_summary(self) -> dict | None:
        """Server-seitige Query-Latenz aus dem queries_durations_ms-Histogramm, als
        Delta gegen den begin_server_metrics-Snapshot (= reines Mess-Fenster, Warmup
        raus), ohne Client/gRPC/Netz. Pro query_type getrennt; Perzentile aus dem
        DOMINANTEN query_type (groesster Delta-Count = gemessene Workload), damit
        langsamere interne Query-Typen die p95/p99 nicht verfaelschen."""
        now = self._scrape_durations()
        if now is None:
            return None
        snap = self._srv_snapshot or {}
        deltas = {}
        total_count = 0.0
        for qt, d in now.items():
            s = snap.get(qt, {})
            dc = d["count"] - s.get("count", 0.0)
            if dc <= 0:
                continue
            ds = d["sum"] - s.get("sum", 0.0)
            snap_b = s.get("buckets", {})
            db = {le: cnt - snap_b.get(le, 0.0) for le, cnt in d["buckets"].items()}
            deltas[qt] = {"count": dc, "sum": ds, "buckets": db}
            total_count += dc
        if not deltas:
            return None
        dom_qt = max(deltas, key=lambda k: deltas[k]["count"])
        dom = deltas[dom_qt]
        cls = COLLECTION_VECS if self.variant == "B" else COLLECTION
        out = {
            "source": "weaviate_prometheus",
            "metric": QUERY_DURATION_METRIC,
            "class_name": cls,
            "query_type": dom_qt or "(none)",
            "count": int(dom["count"]),
            "count_all_types": int(total_count),
            "mean_ms": round(dom["sum"] / dom["count"], 3),
            "windowed": self._srv_snapshot is not None,
        }
        for q, name in ((0.50, "p50_ms"), (0.95, "p95_ms"), (0.99, "p99_ms")):
            val = _histogram_quantile(dom["buckets"], q)
            if val is not None:
                out[name] = round(val, 3)
        return out

    def teardown(self) -> None:
        if self._client is not None:
            with suppress(Exception):
                self._client.close()
        for pf in self._pfs:
            with suppress(Exception):
                pf.terminate()
                pf.wait(timeout=5)
        self._pfs = []


# ----- Helpers -------------------------------------------------------------

def _props_dict(metadata: dict, j: int, ext_id: int) -> dict:
    """Stellt ein Properties-Dict aus dem metadata-Bundle für Zeile j zusammen."""
    def safe(key: str, default):
        v = metadata.get(key)
        return v[j] if v is not None and j < len(v) else default

    return {
        "ext_id": int(ext_id),
        "product_id": str(safe("product_id", "") or ""),
        "product_title": str(safe("product_title", "") or ""),
        "user_id": str(safe("user_id", "") or ""),
        "rating": float(safe("rating", 0.0) or 0.0),
        "review_text": str(safe("review_text", "") or ""),
        "timestamp": str(safe("timestamp", "") or ""),
    }


def _filter_from_dict(filters: dict):
    """Übersetzt das Runner-Filter-Dict in einen Weaviate-Filter.
    Unterstützt rating_gte, product_id."""
    from weaviate.classes.query import Filter

    parts = []
    if "rating_gte" in filters:
        parts.append(Filter.by_property("rating").greater_or_equal(float(filters["rating_gte"])))
    if "product_id" in filters:
        parts.append(Filter.by_property("product_id").equal(str(filters["product_id"])))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return Filter.all_of(parts)
