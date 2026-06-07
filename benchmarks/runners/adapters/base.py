"""Adapter-Basisklasse. Jede DB implementiert dieselbe Schnittstelle, damit
der Runner DB-agnostisch laeuft."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class BenchmarkResult:
    # Index-Bau
    build_time_s: float
    size_on_disk_mb: float | None  # None = unbekannt

    # Latenz-Verteilung (ms) -- ein Eintrag pro Query
    latencies_ms: list[float]

    # Throughput-Modus: parallele Clients beobachtete QPS (über Gesamtlaufzeit)
    throughput_qps: float

    # Recall vs. Ground-Truth, einmal pro Query gesammelt (Sets aus Top-k IDs)
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float

    # Optionale Resourcen
    cpu_avg_cores: float | None = None
    mem_avg_mb: float | None = None

    # freie Notizen vom Adapter (Image, Version, etc.)
    notes: dict = field(default_factory=dict)


class Adapter(ABC):
    """Lifecycle: setup -> insert (batches) -> build_index -> query (loop) -> teardown.

    Variante A vs B (Thesis 5.3) wird ueber cfg["variant"] gesteuert:
      - "A": Embedding + Metadaten zusammen in einer Tabelle/Collection.
      - "B": Embedding und Metadaten getrennt, verknuepft ueber id. Adapter
        muss `insert_metadata` implementieren und im Query joinen.

    Query-Typen (Thesis Kap. 4):
      - query        -> Top-k Similarity Search (Pflicht)
      - query_filtered, query_batch, query_hybrid sind optional. Wenn ein
        Adapter sie nicht beherrscht, lassen wir sie hier eine
        NotImplementedError werfen -- der Runner faengt das ab.
    """

    db_name: str = "abstract"

    def __init__(self, cfg: dict, dim: int) -> None:
        self.cfg = cfg
        self.dim = dim
        self.variant = cfg.get("variant", "A")
        self.index_params = cfg["index"]["params"]

    @abstractmethod
    def setup(self) -> None:
        """Verbindung, Schema/Tabelle, leere Collection."""

    @abstractmethod
    def insert(self, ids: np.ndarray, vecs: np.ndarray,
                metadata: dict | None = None) -> None:
        """Einen Batch Vektoren einfügen. Wenn metadata != None liegt der
        Adapter selbst entscheidet, ob er Variante A (inline) oder B
        (separate Tabelle/Collection) bedient. Wird mehrfach aufgerufen."""

    def insert_metadata(self, ids: np.ndarray, metadata: dict) -> None:
        """Optional separater Metadaten-Pfad. Default: no-op -- Adapter
        sollten Metadaten direkt in `insert` einbringen, das ist effizienter."""

    @abstractmethod
    def build_index(self) -> float:
        """Index bauen (falls noch nicht). Gibt Bauzeit in Sekunden zurück."""

    @abstractmethod
    def query(self, vec: np.ndarray, k: int) -> list[int]:
        """Top-k Similarity Search. Pflichtmethode (Thesis 4.1)."""

    def query_filtered(self, vec: np.ndarray, k: int,
                        filters: dict) -> list[int]:
        """Top-k mit Metadatenfilter (Thesis 4.2). Beispiel-Filter:
        `{"rating_gte": 4}` oder `{"product_id": "B001"}`."""
        raise NotImplementedError(f"{self.db_name} unterstuetzt query_filtered nicht")

    def query_batch(self, vecs: np.ndarray, k: int) -> list[list[int]]:
        """Mehrere Queries in einem Call (Thesis 4.3). Default: Loop ueber
        query. Adapter mit nativen Batch-APIs sollten das ueberschreiben."""
        return [self.query(v, k) for v in vecs]

    def query_hybrid(self, vec: np.ndarray, text: str, k: int,
                      alpha: float = 0.5) -> list[int]:
        """BM25 + Vektor kombiniert (Thesis 4.4). `alpha` gewichtet Vektor
        gegen BM25 (1.0 = pur Vektor, 0.0 = pur Text)."""
        raise NotImplementedError(f"{self.db_name} unterstuetzt query_hybrid nicht")

    @abstractmethod
    def index_size_mb(self) -> float | None:
        """Index-Größe auf Disk (MB) wenn ermittelbar, sonst None."""

    def teardown(self) -> None:
        """Verbindung schließen. Default: no-op."""

    # ---- gemeinsame Helfer für Recall/NDCG -------------------------------

    @staticmethod
    def recall_at_k(retrieved: list[int], truth: np.ndarray, k: int) -> float:
        if len(retrieved) == 0:
            return 0.0
        ret_set = set(retrieved[:k])
        gt_set = set(truth[:k].tolist())
        if not gt_set:
            return 0.0
        return len(ret_set & gt_set) / len(gt_set)

    @staticmethod
    def precision_at_k(retrieved: list[int], truth: np.ndarray, k: int) -> float:
        ret = retrieved[:k]
        if not ret:
            return 0.0
        gt_set = set(truth[:k].tolist())
        return sum(1 for r in ret if r in gt_set) / len(ret)

    @staticmethod
    def ndcg_at_k(retrieved: list[int], truth: np.ndarray, k: int) -> float:
        """Binary-Relevance NDCG@k. Truth-Top-k zählt als relevant."""
        import math

        gt_set = set(truth[:k].tolist())
        dcg = 0.0
        for i, rid in enumerate(retrieved[:k]):
            if rid in gt_set:
                dcg += 1.0 / math.log2(i + 2)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gt_set))))
        return dcg / ideal if ideal > 0 else 0.0
