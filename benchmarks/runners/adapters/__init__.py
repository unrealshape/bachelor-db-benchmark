"""Adapter pro Vector-DB. Jeder Adapter folgt dem base.Adapter Interface."""
from .base import Adapter, BenchmarkResult
from .weaviate_adapter import WeaviateAdapter
from .pgvector_adapter import PgvectorAdapter

# pinecone optional -- der schlanke In-Cluster-Mess-Container hat das SDK nicht
# (und braucht es nicht).
try:
    from .pinecone_adapter import PineconeAdapter
except Exception:  # pragma: no cover
    PineconeAdapter = None

ADAPTERS = {
    "weaviate": WeaviateAdapter,
    "pgvector": PgvectorAdapter,
}
if PineconeAdapter is not None:
    ADAPTERS["pinecone"] = PineconeAdapter


def get_adapter(db_name: str) -> type[Adapter]:
    if db_name not in ADAPTERS:
        raise ValueError(f"Unbekannte DB: {db_name}")
    return ADAPTERS[db_name]
