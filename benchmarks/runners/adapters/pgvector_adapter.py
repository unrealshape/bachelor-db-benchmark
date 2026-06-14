"""pgvector-Adapter. Spricht Postgres im Cluster über kubectl port-forward
an. Bulk-Insert via COPY FROM STDIN, dann CREATE INDEX (IVFFlat oder HNSW
je nach Config), dann Query-Loop."""

from __future__ import annotations

import io
import os
import subprocess
import time
from contextlib import suppress

import numpy as np

from .base import Adapter


PG_NAMESPACE = "db-pgvector"
PG_SERVICE = "pgvector"
PG_PORT = 5432

# Variante A: alles in einer Tabelle (TABLE).
# Variante B: Vektoren in TABLE_VECS, Metadaten in TABLE_META, JOIN via id.
TABLE = "bench_items"
TABLE_VECS = "bench_vecs"
TABLE_META = "bench_meta"
INDEX = "bench_items_emb_idx"
DB_NAME = os.environ.get("PG_DB", "benchmark")
DB_USER = os.environ.get("PG_USER", "bench")
DB_PASS = os.environ.get("PG_PASSWORD", "benchlocal")

META_COLS_SQL = (
    "product_id TEXT, product_title TEXT, user_id TEXT, "
    "rating SMALLINT, review_text TEXT, timestamp TEXT"
)


def _port_forward(local_port: int) -> subprocess.Popen:
    cmd = [
        "kubectl", "port-forward",
        "-n", PG_NAMESPACE,
        f"svc/{PG_SERVICE}",
        f"{local_port}:{PG_PORT}",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for(host: str, port: int, timeout_s: float = 120.0) -> None:
    import socket
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"timeout waiting for {host}:{port}")


def _establish_port_forward(local_port: int, attempts: int = 8) -> subprocess.Popen:
    """Port-forward auf svc/pgvector robust aufbauen. Direkt nach einem rollout-restart
    hat der Service noch keine ready Endpoints -> `kubectl port-forward svc/...` stirbt
    sofort und der lokale Port bindet nie (_wait_for liefe dann nur in den Timeout).
    Daher den Forward mehrfach neu aufsetzen, bis der Port wirklich steht."""
    last = None
    for _ in range(attempts):
        pf = _port_forward(local_port)
        try:
            _wait_for("127.0.0.1", local_port, timeout_s=15.0)
            return pf
        except RuntimeError as e:
            last = e
            try:
                pf.terminate()
            except Exception:
                pass
            time.sleep(4)
    raise RuntimeError(f"port-forward auf 127.0.0.1:{local_port} nicht aufbaubar: {last}")


def _vec_to_pg(vec: np.ndarray) -> str:
    """Serialisiert einen Vektor in das pgvector-Textformat: '[1,2,3]'."""
    # repr ohne Wissenschaft: explizit f-format
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


class PgvectorAdapter(Adapter):
    db_name = "pgvector"

    def __init__(self, cfg: dict, dim: int) -> None:
        super().__init__(cfg, dim)
        self._pf: subprocess.Popen | None = None
        self._conn = None
        self._local_port = int(os.environ.get("PG_LOCAL_PORT", "55432"))
        self._index_type = cfg["index"]["type"]  # 'hnsw' oder 'ivfflat'
        self._build_s: float | None = None
        # In-Cluster-Modus: Mess-Pod verbindet via ClusterIP-DNS, kein port-forward.
        self._in_cluster = os.environ.get("BENCH_IN_CLUSTER") == "1"

    def _connect(self):
        import psycopg
        if self._in_cluster:
            host = os.environ.get(
                "PG_HOST", f"{PG_SERVICE}.{PG_NAMESPACE}.svc.cluster.local")
            port = int(os.environ.get("PG_PORT", str(PG_PORT)))
        else:
            host = "127.0.0.1"
            port = self._local_port
        # Nach einem pre-run reset (rollout restart) bindet der port-forward sofort
        # lokal, aber Postgres im frischen Pod nimmt erst nach einigen Sekunden an
        # -> _wait_for (nur TCP-Bind) reicht nicht. Bis ~45s auf echte Annahme warten.
        deadline = time.time() + 45.0
        last_exc = None
        while time.time() < deadline:
            try:
                return psycopg.connect(
                    host=host,
                    port=port,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASS,
                    autocommit=True,
                )
            except psycopg.OperationalError as e:
                last_exc = e
                time.sleep(1.5)
        raise last_exc

    def _apply_search_params(self) -> None:
        """ef_search / probes sind Session-Settings -- pro Connection neu setzen
        (im Mess-Pod, der eine eigene Connection hat)."""
        p = self.index_params
        with self._conn.cursor() as cur:
            if self._index_type == "hnsw":
                cur.execute(f"SET hnsw.ef_search = {p.get('ef_search', 64)}")
            elif self._index_type == "ivfflat":
                cur.execute(f"SET ivfflat.probes = {p.get('probes', 10)}")

    def attach(self) -> None:
        """Verbindet zu einer bereits befuellten DB ohne Schema-Drop (Mess-Pad
        im Cluster). Setzt Such-Parameter fuer die neue Session."""
        if not self._in_cluster and os.environ.get("PG_SKIP_PORTFORWARD") != "1":
            self._pf = _establish_port_forward(self._local_port)
        self._conn = self._connect()
        self._apply_search_params()

    # ---- lifecycle --------------------------------------------------------

    @property
    def _vec_table(self) -> str:
        return TABLE_VECS if self.variant == "B" else TABLE

    def setup(self) -> None:
        if not self._in_cluster and os.environ.get("PG_SKIP_PORTFORWARD") != "1":
            self._pf = _establish_port_forward(self._local_port)

        self._conn = self._connect()
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_VECS}")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_META}")
            # UNLOGGED: kein WAL -> deutlich schnellerer Bulk-COPY + kein
            # WAL-Disk-Druck (bei 2.65M+ sonst PVC-/Crash-Risiko). Fuer einen
            # Benchmark unkritisch -- die Daten werden vor jedem Lauf neu geladen.
            if self.variant == "B":
                # Vektoren und Metadaten getrennt, verknuepft ueber id (Thesis 5.3 B).
                cur.execute(
                    f"CREATE UNLOGGED TABLE {TABLE_VECS} "
                    f"(id BIGINT PRIMARY KEY, embedding vector({self.dim}))"
                )
                cur.execute(
                    f"CREATE UNLOGGED TABLE {TABLE_META} "
                    f"(id BIGINT PRIMARY KEY, {META_COLS_SQL})"
                )
            else:
                # Variante A: alles inline.
                cur.execute(
                    f"CREATE UNLOGGED TABLE {TABLE} "
                    f"(id BIGINT PRIMARY KEY, embedding vector({self.dim}), "
                    f"{META_COLS_SQL})"
                )

    def insert(self, ids: np.ndarray, vecs: np.ndarray,
                metadata: dict | None = None) -> None:
        """Variante A: alles in einer COPY-Aktion in TABLE.
        Variante B: vec-Spalten in TABLE_VECS, Metadaten in TABLE_META.

        Bei grossen Eingaben wird intern in Sub-Batches gechunked, damit der
        StringIO-Puffer im Client und das COPY-Statement im Server beherrschbar
        bleiben (1024-dim Vektoren als TEXT sind ~10 KB/Row)."""
        meta_cols = ["product_id", "product_title", "user_id", "rating",
                     "review_text", "timestamp"]
        n = len(ids)
        sub = 5000  # Sub-Batch-Groesse

        def esc(s) -> str:
            if s is None:
                return ""
            t = str(s)
            return (
                t.replace("\\", "\\\\")
                 .replace("\t", " ")
                 .replace("\n", " ")
                 .replace("\r", " ")
            )

        def _copy_chunk(table: str, cols_sql: str, render):
            buf = io.StringIO()
            for j in range(n):
                if j > 0 and j % sub == 0:
                    buf.seek(0)
                    with self._conn.cursor() as cur:
                        with cur.copy(f"COPY {table} ({cols_sql}) FROM STDIN") as cp:
                            cp.write(buf.read())
                    self._conn.commit()
                    buf = io.StringIO()
                buf.write(render(j) + "\n")
            buf.seek(0)
            data = buf.read()
            if data:
                with self._conn.cursor() as cur:
                    with cur.copy(f"COPY {table} ({cols_sql}) FROM STDIN") as cp:
                        cp.write(data)
                self._conn.commit()

        if self.variant == "B":
            _copy_chunk(
                TABLE_VECS, "id, embedding",
                lambda j: f"{int(ids[j])}\t{_vec_to_pg(vecs[j])}",
            )
            if metadata is not None:
                def render_meta(j: int) -> str:
                    row = [str(int(ids[j]))]
                    for c in meta_cols:
                        v = metadata.get(c, [""] * n)[j] if metadata.get(c) is not None else ""
                        if c == "rating":
                            row.append(str(int(v) if v not in (None, "") else 0))
                        else:
                            row.append(esc(v))
                    return "\t".join(row)
                _copy_chunk(TABLE_META, "id, " + ", ".join(meta_cols), render_meta)
            return

        if metadata is None:
            _copy_chunk(
                TABLE, "id, embedding",
                lambda j: f"{int(ids[j])}\t{_vec_to_pg(vecs[j])}",
            )
            return

        cols = ["id", "embedding"] + meta_cols
        def render_full(j: int) -> str:
            row = [str(int(ids[j])), _vec_to_pg(vecs[j])]
            for c in meta_cols:
                v = metadata.get(c, [""] * n)[j] if metadata.get(c) is not None else ""
                if c == "rating":
                    row.append(str(int(v) if v not in (None, "") else 0))
                else:
                    row.append(esc(v))
            return "\t".join(row)
        _copy_chunk(TABLE, ", ".join(cols), render_full)

    def build_index(self, build_text_index: bool | None = None) -> float:
        p = self.index_params
        t0 = time.time()
        tbl = self._vec_table
        with self._conn.cursor() as cur:
            # Build-Speicher pro Lauf hochsetzen, wenn die Config es vorgibt: der
            # IVFFlat-Build skaliert mit n (Stufe M @5,25M braucht ~5,7 GB, der
            # globale Default 2 GB reicht nur bis ~S). Session-GUC -> nur dieser
            # Ingest, beeinflusst die Query-RAM-Paritaet nicht. Tier-abhaengig in
            # der Config (8-GiB-Tier kann den Build-Speicher nicht stellen -> Befund).
            mwm = self.cfg.get("maintenance_work_mem_mb")
            if mwm:
                cur.execute(f"SET maintenance_work_mem = '{int(mwm)}MB'")
            if self._index_type == "hnsw":
                m = p.get("m", 16)
                efc = p.get("ef_construction", 128)
                cur.execute(
                    f"CREATE INDEX {INDEX} ON {tbl} USING hnsw "
                    f"(embedding vector_cosine_ops) "
                    f"WITH (m = {m}, ef_construction = {efc})"
                )
                ef_search = p.get("ef_search", 64)
                cur.execute(f"SET hnsw.ef_search = {ef_search}")
            elif self._index_type == "ivfflat":
                lists = p.get("lists", 1000)
                probes = p.get("probes", 10)
                cur.execute(
                    f"CREATE INDEX {INDEX} ON {tbl} USING ivfflat "
                    f"(embedding vector_cosine_ops) WITH (lists = {lists})"
                )
                cur.execute(f"SET ivfflat.probes = {probes}")
            else:
                raise ValueError(f"unbekannter index type: {self._index_type}")
            # Hybrid braucht einen GIN-Index auf dem tsvector, sonst macht die
            # Text-Komponente pro Query einen Seqscan. Im gekoppelten Modus nur
            # fuer hybrid bauen, damit topk/filtered/batch ihre build_time nicht
            # verfaelscht kriegen. Im entkoppelten Ingest (build_text_index=True)
            # immer -- EIN Index bedient dann alle vier Workloads.
            want_text = (build_text_index if build_text_index is not None
                         else self.cfg.get("workload") == "hybrid")
            if want_text:
                txt_tbl = TABLE_META if self.variant == "B" else tbl
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {txt_tbl}_ts_gin ON {txt_tbl} "
                    f"USING gin (to_tsvector('english', review_text))"
                )
            cur.execute(f"ANALYZE {tbl}")
            if self.variant == "B":
                cur.execute(f"ANALYZE {TABLE_META}")
        self._build_s = time.time() - t0
        return self._build_s

    def query(self, vec: np.ndarray, k: int) -> list[int]:
        vec_str = _vec_to_pg(vec)
        tbl = self._vec_table
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec_str, k),
            )
            return [row[0] for row in cur.fetchall()]

    def query_filtered(self, vec: np.ndarray, k: int, filters: dict) -> list[int]:
        """filters: {"rating_gte": int, "product_id": str}. Andere Keys werden
        ignoriert. Variante B macht den JOIN, Variante A filtert inline."""
        vec_str = _vec_to_pg(vec)
        clauses: list[str] = []
        params: list = []
        if "rating_gte" in filters:
            clauses.append("m.rating >= %s" if self.variant == "B" else "rating >= %s")
            params.append(int(filters["rating_gte"]))
        if "product_id" in filters:
            clauses.append("m.product_id = %s" if self.variant == "B" else "product_id = %s")
            params.append(filters["product_id"])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        if self.variant == "B":
            sql = (
                f"SELECT v.id FROM {TABLE_VECS} v "
                f"JOIN {TABLE_META} m USING (id)"
                f"{where} "
                f"ORDER BY v.embedding <=> %s::vector LIMIT %s"
            )
        else:
            sql = (
                f"SELECT id FROM {TABLE}"
                f"{where} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s"
            )
        params.extend([vec_str, k])
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return [row[0] for row in cur.fetchall()]

    def query_hybrid(self, vec: np.ndarray, text: str, k: int,
                      alpha: float = 0.5) -> list[int]:
        """Native Reciprocal Rank Fusion in EINER SQL (server-seitig, kein
        Client-Merge) -- passt zur RRF-GT (gen_special_gt) und zu Weaviates
        Ranked-Fusion (RANK_CONSTANT=60):

            score = alpha/(K + vrank) + (1-alpha)/(K + trank)

        Kandidaten = Top-pool nach Vektor UNION Top-pool nach Text. Dokumente
        ohne Rang in einer Liste tragen dort ~0 bei (COALESCE auf 1e9).
        pgvector nutzt ts_rank_cd statt echtem BM25 -- bauartbedingte Differenz
        zur BM25-GT, fairer Befund (kein natives BM25 in pgvector)."""
        vec_str = _vec_to_pg(vec)
        q = text or ""
        pool = max(k * 5, 500)
        K = 60
        vec_tbl = TABLE_VECS if self.variant == "B" else TABLE
        txt_tbl = TABLE_META if self.variant == "B" else TABLE
        sql = (
            f"WITH vc AS ("
            f"  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vrank "
            f"  FROM {vec_tbl} ORDER BY embedding <=> %s::vector LIMIT %s"
            f"), tc AS ("
            f"  SELECT id, ROW_NUMBER() OVER ("
            f"    ORDER BY ts_rank_cd(to_tsvector('english', review_text), "
            f"             plainto_tsquery('english', %s)) DESC) AS trank "
            f"  FROM {txt_tbl} "
            f"  WHERE to_tsvector('english', review_text) @@ plainto_tsquery('english', %s) "
            f"  ORDER BY ts_rank_cd(to_tsvector('english', review_text), "
            f"           plainto_tsquery('english', %s)) DESC LIMIT %s"
            f"), u AS (SELECT id FROM vc UNION SELECT id FROM tc) "
            f"SELECT u.id FROM u "
            f"LEFT JOIN vc ON vc.id = u.id LEFT JOIN tc ON tc.id = u.id "
            f"ORDER BY %s / ({K} + COALESCE(vc.vrank, 1e9)) "
            f"       + %s / ({K} + COALESCE(tc.trank, 1e9)) DESC "
            f"LIMIT %s"
        )
        params = (vec_str, vec_str, pool, q, q, q, pool,
                  alpha, 1.0 - alpha, k)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [row[0] for row in cur.fetchall()]

    def index_size_mb(self) -> float | None:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT pg_relation_size('{INDEX}')")
                bytes_ = cur.fetchone()[0]
            return bytes_ / (1024 * 1024)
        except Exception:
            return None

    def teardown(self) -> None:
        if self._conn is not None:
            with suppress(Exception):
                self._conn.close()
        if self._pf is not None:
            with suppress(Exception):
                self._pf.terminate()
                self._pf.wait(timeout=5)
