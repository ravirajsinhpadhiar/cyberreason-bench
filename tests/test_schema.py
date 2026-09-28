from datetime import date

import pytest

from cyberreason_bench.schema import (
    GraphStore, Node, Edge, NodeType, EdgeType, Provenance, ProvenanceKind,
)


def make_store():
    gs = GraphStore()
    gs.add_node(Node("CVE-2024-1234", NodeType.CVE, "CVE-2024-1234"))
    gs.add_node(Node("openssl-3.0.1", NodeType.PRODUCT_VERSION, "OpenSSL 3.0.1"))
    gs.add_node(Node("openssl-3.0.2", NodeType.PRODUCT_VERSION, "OpenSSL 3.0.2"))
    gs.add_edge(Edge(
        "CVE-2024-1234", "openssl-3.0.1", EdgeType.AFFECTS,
        Provenance(ProvenanceKind.EXPLICIT, source="NVD-2024-1234", asserted_on=date(2024, 3, 1)),
    ))
    gs.add_edge(Edge(
        "CVE-2024-1234", "openssl-3.0.2", EdgeType.AFFECTS,
        Provenance(ProvenanceKind.MODEL_PROPOSED, source="llm-guess", asserted_on=date(2024, 3, 2),
                   model_name="qwen3-8b", uncertainty=0.6),
    ))
    return gs


def test_expert_assessed_requires_reviewer():
    with pytest.raises(ValueError):
        Provenance(ProvenanceKind.EXPERT_ASSESSED, source="x", asserted_on=date.today())


def test_model_proposed_requires_model_name():
    with pytest.raises(ValueError):
        Provenance(ProvenanceKind.MODEL_PROPOSED, source="x", asserted_on=date.today())


def test_uncertainty_bounds():
    with pytest.raises(ValueError):
        Provenance(ProvenanceKind.EXPLICIT, source="x", asserted_on=date.today(), uncertainty=1.5)


def test_edge_requires_known_nodes():
    gs = GraphStore()
    gs.add_node(Node("a", NodeType.CVE, "a"))
    with pytest.raises(ValueError):
        gs.add_edge(Edge("a", "unknown", EdgeType.AFFECTS,
                          Provenance(ProvenanceKind.EXPLICIT, source="s", asserted_on=date.today())))


def test_neighbors_provenance_filter():
    gs = make_store()
    all_neighbors = gs.neighbors("CVE-2024-1234")
    assert len(all_neighbors) == 2

    high_trust_only = gs.neighbors(
        "CVE-2024-1234", min_provenance={ProvenanceKind.EXPLICIT, ProvenanceKind.DETERMINISTIC_DERIVED}
    )
    assert len(high_trust_only) == 1
    assert high_trust_only[0].dst == "openssl-3.0.1"


def test_temporal_validity():
    gs = GraphStore()
    gs.add_node(Node("a", NodeType.ADVISORY, "a"))
    gs.add_node(Node("b", NodeType.ADVISORY, "b"))
    gs.add_edge(Edge(
        "a", "b", EdgeType.SUPERSEDES,
        Provenance(ProvenanceKind.EXPLICIT, source="s", asserted_on=date(2024, 1, 1)),
        valid_from=date(2024, 1, 1), valid_until=date(2024, 6, 1),
    ))
    assert gs.neighbors("a", as_of=date(2024, 3, 1)) != []
    assert gs.neighbors("a", as_of=date(2025, 1, 1)) == []


def test_stats_counts_by_provenance_kind():
    gs = make_store()
    stats = gs.stats()
    assert stats["n_nodes"] == 3
    assert stats["n_edges"] == 2
    assert stats["edges_by_provenance_kind"]["explicit"] == 1
    assert stats["edges_by_provenance_kind"]["model_proposed"] == 1


def test_node_id_collision_with_different_content_rejected():
    gs = GraphStore()
    gs.add_node(Node("a", NodeType.CVE, "A"))
    with pytest.raises(ValueError):
        gs.add_node(Node("a", NodeType.CVE, "Different label"))
