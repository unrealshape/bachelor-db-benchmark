"""Tests fuer Pinecone-eigene Hilfen, die ohne API-Key/Netzwerk laufen.

Geprueft werden:
    - Region-Parser `_split_region`
    - `server_latency_summary()` bei leerem Header-Aggregat (None)
    - `notes` Property -- liefert die Pod-Konfig fuer summary.json
    - Variante B wirft `NotImplementedError` beim insert (Pinecone kennt
      keine getrennte Metadaten-Collection)
"""

from __future__ import annotations

import numpy as np
import pytest

from adapters.pinecone_adapter import (
    DEFAULT_CLOUD,
    SERVER_LATENCY_HEADER,
    PineconeAdapter,
    _split_region,
)


# ---- Region-Parser ----------------------------------------------------------

class TestSplitRegion:
    def test_legacy_aws_prefix(self):
        cloud, region = _split_region("aws-us-east-1")
        assert cloud == "aws"
        assert region == "us-east-1"

    def test_legacy_gcp_prefix(self):
        cloud, region = _split_region("gcp-europe-west4")
        assert cloud == "gcp"
        assert region == "europe-west4"

    def test_legacy_azure_prefix(self):
        cloud, region = _split_region("azure-eastus2")
        assert cloud == "azure"
        assert region == "eastus2"

    def test_new_style_no_prefix_falls_back_to_aws(self):
        """Neues Format ohne Cloud-Praefix -> Default-Cloud (aws)."""
        cloud, region = _split_region("us-east-1")
        assert cloud == DEFAULT_CLOUD
        assert region == "us-east-1"

    def test_whitespace_is_stripped(self):
        cloud, region = _split_region("  aws-us-west-2  ")
        assert cloud == "aws"
        assert region == "us-west-2"


# ---- Adapter-Helpers (ohne API-Call) ----------------------------------------

def test_server_latency_summary_empty_returns_none(dummy_config, test_dim):
    """Ohne erfasste Header-Latenzen (`server_latencies_ms` leer) darf der
    Summary-Helper nicht crashen und gibt None zurueck -- der Runner kann
    das gefahrlos in summary.json schreiben."""
    inst = PineconeAdapter(dummy_config, test_dim)
    assert inst.server_latencies_ms == []
    assert inst.server_latency_summary() is None


def test_server_latency_summary_with_samples(dummy_config, test_dim):
    """Mit ein paar Werten kommt ein dict mit den erwarteten Perzentilen."""
    inst = PineconeAdapter(dummy_config, test_dim)
    inst.server_latencies_ms.extend([1.0, 2.0, 3.0, 4.0, 5.0])
    summary = inst.server_latency_summary()
    assert summary is not None
    assert summary["samples"] == 5
    assert summary["mean"] == pytest.approx(3.0)
    # Perzentile sind monoton wachsend
    assert summary["p50"] <= summary["p90"] <= summary["p99"]


def test_notes_property_contains_pod_config(dummy_config, test_dim):
    """`notes` liefert die Pod-Konfig, die der Runner in summary.json.notes
    uebernimmt. Pflichtfelder fuer die Auswertung der Thesis."""
    inst = PineconeAdapter(dummy_config, test_dim)
    notes = inst.notes
    assert notes["pod_type"] == "s1.x1"
    assert notes["pods"] == 1
    assert notes["cloud"] == "aws"
    assert notes["region"] == "us-east-1"
    assert notes["metric"] == "cosine"
    assert notes["managed"] is True
    assert notes["server_latency_header"] == SERVER_LATENCY_HEADER


def test_notes_uses_legacy_region_string_when_only_region_given(test_dim):
    """Wenn die Config nur `region: 'aws-us-west-2'` enthaelt, wird das
    via `_split_region` zerlegt und beide Felder in `notes` erscheinen."""
    cfg = {
        "name": "x",
        "index_name": "x",
        "variant": "A",
        "index": {"type": "hnsw", "params": {"region": "aws-us-west-2"}},
    }
    inst = PineconeAdapter(cfg, test_dim)
    notes = inst.notes
    assert notes["cloud"] == "aws"
    assert notes["region"] == "us-west-2"


def test_variant_b_insert_raises_not_implemented(
    dummy_config_variant_b, test_dim, test_corpus,
):
    """Pinecone hat keine getrennte Metadaten-Collection. Variante B muss
    am insert sauber abbrechen, nicht erst beim ersten API-Call."""
    ids, vecs = test_corpus
    inst = PineconeAdapter(dummy_config_variant_b, test_dim)
    with pytest.raises(NotImplementedError):
        inst.insert(ids, vecs, metadata=None)


def test_index_size_mb_returns_none(dummy_config, test_dim):
    """Pinecone exponiert keine Disk-Groesse -- `None` ist der definierte
    Rueckgabewert, damit der Runner die Spalte als 'n/a' rendert."""
    inst = PineconeAdapter(dummy_config, test_dim)
    assert inst.index_size_mb() is None


def test_pop_server_latency_empty_returns_none(dummy_config, test_dim):
    """Ohne Header-Hit liefert der TLS-Pop None, kein Crash."""
    inst = PineconeAdapter(dummy_config, test_dim)
    assert inst._pop_server_latency() is None
