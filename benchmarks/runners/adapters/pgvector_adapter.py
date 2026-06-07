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

    def _connect(self):
        import psycopg
        return psycopg.connect(
            host="127.0.0.1",
            port=self._local_port,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            autocommit=True,
        )

    # ---- lifecycle --------------------------------------------------------

    @property
    def _vec_table(self) -> str:
        return TABLE_VECS if self.variant == "B" else TABLE

    def setup(self) -> None:
        if os.environ.get("PG_SKIP_PORTFORWARD") != "1":
            self._pf = _port_forward(self._local_port)
            _wait_for("127.0.0.1", self._local_port)

        self._conn = self._connect()
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_VECS}")
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_META}")
            if self.variant == "B":
                # Vektoren und Metadaten getrennt, verknuepft ueber id (Thesis 5.3 B).
                cur.execute(
                    f"CREATE TABLE {TABLE_VECS} "
                    f"(id BIGINT PRIMARY KEY, embedding vector({self.dim}))"
                )
                cur.execute(
                    f"CREATE TABLE {TABLE_META} "
                    f"(id BIGINT PRIMARY KEY, {META_COLS_SQL})"
                )
            else:
                # Variante A: alles inline.
                cur.execute(
                    f"CREATE TABLE {TABLE} "
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

    def build_index(self) -> float:
        p = self.index_params
        t0 = time.time()
        tbl = self._vec_table
        with self._conn.cursor() as cur:
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
        """Hybrid = BM25 (tsvector + plainto_tsquery) + Vektor (cosine).
        Wir nutzen Reciprocal Rank Fusion (RRF) als Score, weil die zwei
        Räume nicht direkt vergleichbar sind. alpha steuert das Gewicht
        des Vektor-Rangs gegen den Text-Rang (1.0 = pur Vektor, 0.0 = pur Text).
        Variante B: erst Kandidaten aus Vektoren, dann in TABLE_META re-ranken."""
        vec_str = _vec_to_pg(vec)
        oversample = k * 10
        if self.variant == "B":
            # B: vec-cands holen, dann gegen meta.review_text BM25 ranken
            sql = (
                f"WITH vec_cands AS ("
                f" SELECT id, embedding <=> %s::vector AS vd, "
                f"        ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vrank "
                f" FROM {TABLE_VECS} ORDER BY embedding <=> %s::vector LIMIT %s "
                f"),"
                f"text_score AS ("
                f" SELECT id, "
                f"        ts_rank_cd(to_tsvector('english', review_text), plainto_tsquery('english', %s)) AS ts "
                f" FROM {TABLE_META} WHERE id IN (SELECT id FROM vec_cands)"
                f")"
                f"SELECT v.id "
                f"FROM vec_cands v LEFT JOIN text_score t USING (id) "
                f"ORDER BY %s * (1.0 / NULLIF(v.vrank, 0)) + (1.0 - %s) * COALESCE(t.ts, 0) DESC "
                f"LIMIT %s"
            )
            params = (vec_str, vec_str, vec_str, oversample, text or "",
                      alpha, alpha, k)
        else:
            sql = (
                f"WITH vec_cands AS ("
                f" SELECT id, embedding <=> %s::vector AS vd, "
                f"        ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vrank, "
                f"        ts_rank_cd(to_tsvector('english', review_text), plainto_tsquery('english', %s)) AS ts "
                f" FROM {TABLE} ORDER BY embedding <=> %s::vector LIMIT %s "
                f")"
                f"SELECT id FROM vec_cands "
                f"ORDER BY %s * (1.0 / NULLIF(vrank, 0)) + (1.0 - %s) * COALESCE(ts, 0) DESC "
                f"LIMIT %s"
            )
            params = (vec_str, vec_str, text or "", vec_str, oversample,
                      alpha, alpha, k)
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
