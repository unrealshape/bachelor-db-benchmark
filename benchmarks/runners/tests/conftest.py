"""Pytest-Fixtures fuer die Adapter-Smoke-Tests.

Die Tests laufen ohne API-Keys, ohne kubectl-port-forwards und ohne
laufende DB-Server. Wir pruefen nur das Interface (ABC-Vertrag,
Registry-Lookup, Konstruktoren) und Pinecone-eigene Hilfen, die rein in
Python ohne Netzwerk arbeiten (Region-Parser, notes-Property,
server_latency_summary auf leerem Zustand).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Damit `from adapters import ...` aus dem runners-Verzeichnis funktioniert,
# auch wenn pytest aus dem Repo-Root gestartet wird.
RUNNERS_DIR = Path(__file__).resolve().parents[1]
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))


TEST_DIM = 1024
TEST_N = 50
TEST_K = 10


@pytest.fixture(scope="session")
def test_dim() -> int:
    return TEST_DIM


@pytest.fixture(scope="session")
def test_corpus() -> tuple[np.ndarray, np.ndarray]:
    """50 random float32-Vektoren bei 1024 dim + zugehoerige Integer-IDs.

    Fester Seed, damit Tests deterministisch sind.
    """
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((TEST_N, TEST_DIM)).astype(np.float32)
    ids = np.arange(TEST_N, dtype=np.int64)
    return ids, vecs


@pytest.fixture(scope="session")
def test_queries() -> np.ndarray:
    """5 Query-Vektoren in derselben Dimension."""
    rng = np.random.default_rng(seed=7)
    return rng.standard_normal((5, TEST_DIM)).astype(np.float32)


@pytest.fixture
def dummy_config() -> dict:
    """Minimale Config, die alle drei Adapter-Konstruktoren akzeptieren.

    Wichtig:
        - `name`/`index_name` damit Pinecone nicht im __init__ stolpert.
        - `index.type` fuer pgvector (verzweigt auf hnsw/ivfflat).
        - `index.params` mit HNSW-Default-Keys, die alle drei lesen koennen.
    """
    return {
        "name": "bench-smoke",
        "index_name": "bench-smoke",
        "variant": "A",
        "index": {
            "type": "hnsw",
            "params": {
                # gemeinsame HNSW-Knoepfe
                "M": 16,
                "m": 16,
                "ef": 64,
                "ef_construction": 128,
                "ef_search": 64,
                # Pinecone Pod-Config
                "pod_type": "s1.x1",
                "pods": 1,
                "region": "us-east-1",
                "cloud": "aws",
                "metric": "cosine",
            },
        },
    }


@pytest.fixture
def dummy_config_variant_b(dummy_config: dict) -> dict:
    """Wie dummy_config, aber Variante B (getrennte Meta-Collection)."""
    cfg = dict(dummy_config)
    cfg["variant"] = "B"
    return cfg
