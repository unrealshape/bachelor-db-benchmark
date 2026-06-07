"""Adapter pro Vector-DB. Jeder Adapter folgt dem base.Adapter Interface."""
from .base import Adapter, BenchmarkResult
from .weaviate_adapter import WeaviateAdapter
from .pgvector_adapter import PgvectorAdapter
from .pinecone_adapter import PineconeAdapter

ADAPTERS = {
    "weaviate": WeaviateAdapter,
    "pgvector": PgvectorAdapter,
    "pinecone": PineconeAdapter,
}


def get_adapter(db_name: str) -> type[Adapter]:
    if db_name not in ADAPTERS:
        raise ValueError(f"Unbekannte DB: {db_name}")
    return ADAPTERS[db_name]
