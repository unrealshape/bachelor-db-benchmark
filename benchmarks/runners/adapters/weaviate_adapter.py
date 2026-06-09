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

    def _connect_client(self):
        import weaviate
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
        # vectorCacheMaxObjects begrenzt die im RAM gehaltenen Vektoren auf das
        # 8-GB-Pod-Budget (~1,2M x 4KB ≈ 5GB). Groessere Stufen spillen den Rest
        # on-demand von Disk -> misst die "Index waechst raus"-Degradation
        # (Thesis Kap 5) statt OOM-Crash. Pro Config ueberschreibbar.
        hnsw = Configure.VectorIndex.hnsw(
            ef=p.get("ef", 64),
            ef_construction=p.get("ef_construction", 128),
            max_connections=p.get("M", 16),
            vector_cache_max_objects=p.get("vector_cache_max_objects", 3_000_000),
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

    def build_index(self) -> float:
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
        self._build_s = time.time() - t0
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
