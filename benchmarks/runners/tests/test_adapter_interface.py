"""Smoke-Tests fuer den Adapter-Interface-Vertrag.

Ziel: zukuenftige Spec-Aenderungen (z.B. Dim 1024 -> 768, neue Pflicht-
methoden) brechen hier laut, nicht erst im echten Run gegen eine DB.

Kein Test ruft `setup()`, `insert()` oder `query()` -- diese brauchen einen
echten Server oder API-Key. Wir pruefen Konstruktion, ABC-Vertrag und
Registry-Lookup.
"""

from __future__ import annotations

import inspect

import pytest

from adapters import ADAPTERS, get_adapter
from adapters.base import Adapter, BenchmarkResult


# Methoden, die `base.Adapter` als `@abstractmethod` markiert.
EXPECTED_ABSTRACT = {"setup", "insert", "build_index", "query", "index_size_mb"}

# Optionale Methoden (Default in der Basis: NotImplementedError oder no-op).
EXPECTED_OPTIONAL = {
    "insert_metadata",
    "query_filtered",
    "query_batch",
    "query_hybrid",
    "teardown",
}


# ---- ABC contract -----------------------------------------------------------

def test_base_adapter_is_abstract():
    """`base.Adapter` darf nicht direkt instanziierbar sein."""
    assert inspect.isabstract(Adapter), (
        "Adapter sollte ABC mit mindestens einer abstrakten Methode bleiben"
    )


def test_base_adapter_exposes_expected_abstract_methods():
    """Alle Pflichtmethoden aus der Thesis-Spec sind als abstrakt markiert."""
    abstract = set(Adapter.__abstractmethods__)
    missing = EXPECTED_ABSTRACT - abstract
    assert not missing, (
        f"Pflichtmethoden fehlen im ABC: {missing}. "
        f"Aktuell abstrakt: {abstract}"
    )


def test_base_adapter_exposes_optional_methods():
    """Die optionalen Methoden existieren als Default-Implementierung."""
    for name in EXPECTED_OPTIONAL:
        assert hasattr(Adapter, name), (
            f"Basis-Adapter sollte optionale Methode `{name}` als Default haben"
        )


def test_benchmark_result_dataclass_fields():
    """BenchmarkResult hat die Felder, die `runner.py` und die Auswertung
    erwartet. Wenn hier etwas wegfaellt, scheitern die Runs spaeter beim
    Serialisieren der summary.json."""
    expected = {
        "build_time_s", "size_on_disk_mb", "latencies_ms", "throughput_qps",
        "recall_at_1", "recall_at_10", "recall_at_100", "ndcg_at_10",
        "cpu_avg_cores", "mem_avg_mb", "notes",
    }
    fields = set(BenchmarkResult.__dataclass_fields__.keys())
    missing = expected - fields
    assert not missing, f"BenchmarkResult fehlen Felder: {missing}"


# ---- Registry ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["weaviate", "pgvector", "pinecone"])
def test_get_adapter_returns_class(name):
    """`get_adapter` liefert die richtige Klasse fuer alle drei DB-Namen."""
    cls = get_adapter(name)
    assert inspect.isclass(cls), f"get_adapter({name!r}) ist keine Klasse"
    assert issubclass(cls, Adapter), (
        f"{cls.__name__} muss von base.Adapter erben"
    )
    assert cls.db_name == name, (
        f"db_name-Attribut ({cls.db_name!r}) passt nicht zum Registry-Key ({name!r})"
    )


def test_get_adapter_rejects_unknown():
    with pytest.raises(ValueError):
        get_adapter("nonexistent-db")


def test_adapters_registry_contains_three_dbs():
    """Wenn jemand einen Adapter hinzufuegt/entfernt, muss der Test
    bewusst nachgezogen werden."""
    assert set(ADAPTERS.keys()) == {"weaviate", "pgvector", "pinecone"}


# ---- Concrete implementations -----------------------------------------------

@pytest.mark.parametrize("name", ["weaviate", "pgvector", "pinecone"])
def test_concrete_adapter_implements_all_abstract_methods(name):
    """Jeder konkrete Adapter darf nicht selbst noch abstrakt sein --
    sonst wuerden `cls(cfg, dim)` an `TypeError: Can't instantiate
    abstract class` scheitern."""
    cls = get_adapter(name)
    assert not inspect.isabstract(cls), (
        f"{cls.__name__} ist noch abstrakt; offene Methoden: "
        f"{getattr(cls, '__abstractmethods__', set())}"
    )


@pytest.mark.parametrize("name", ["weaviate", "pgvector", "pinecone"])
def test_adapter_constructor_does_not_crash(name, dummy_config, test_dim):
    """Konstruktor darf mit der minimalen Config kein Setup auf einer echten
    DB triggern. Wir pruefen nur, dass Felder gesetzt sind."""
    cls = get_adapter(name)
    inst = cls(dummy_config, test_dim)
    assert isinstance(inst, Adapter)
    assert inst.dim == test_dim
    assert inst.variant == "A"
    assert inst.cfg is dummy_config
    # index_params muss durchgereicht sein
    assert inst.index_params == dummy_config["index"]["params"]


@pytest.mark.parametrize("name", ["weaviate", "pgvector", "pinecone"])
def test_adapter_variant_b_constructor(name, dummy_config_variant_b, test_dim):
    """Konstruktoren akzeptieren auch Variante B (split metadata)."""
    cls = get_adapter(name)
    inst = cls(dummy_config_variant_b, test_dim)
    assert inst.variant == "B"


@pytest.mark.parametrize("name", ["weaviate", "pgvector", "pinecone"])
def test_adapter_has_callable_query_methods(name, dummy_config, test_dim):
    """`query` und die optionalen Varianten existieren als aufrufbare
    Attribute -- der Runner ruft sie reflexiv ab."""
    cls = get_adapter(name)
    inst = cls(dummy_config, test_dim)
    for meth in ("setup", "insert", "build_index", "query",
                  "index_size_mb", "teardown",
                  "query_filtered", "query_batch", "query_hybrid"):
        assert callable(getattr(inst, meth)), (
            f"{cls.__name__}.{meth} ist nicht aufrufbar"
        )


@pytest.mark.parametrize("name", ["weaviate", "pgvector", "pinecone"])
def test_recall_and_ndcg_helpers_inherited(name, dummy_config, test_dim):
    """Die statischen Helfer aus Adapter (recall_at_k, ndcg_at_k) sind ueber
    die konkreten Klassen erreichbar -- der Runner verlaesst sich darauf."""
    import numpy as np

    cls = get_adapter(name)
    retrieved = [1, 2, 3, 4]
    truth = np.array([1, 2, 5, 6])
    assert cls.recall_at_k(retrieved, truth, k=4) == pytest.approx(0.5)
    # NDCG@k: ideal-DCG mit zwei Treffern, hier zwei Treffer in den
    # vorderen Plaetzen -- Score > 0.
    assert cls.ndcg_at_k(retrieved, truth, k=4) > 0.0
