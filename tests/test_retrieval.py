from cyberreason_bench.retrieval import (
    Document, HybridRetriever, extract_exact_ids, recall_at_k, evaluate_recall_at_k,
)


DOCS = [
    Document("d1", "CVE-2024-1234 is a remote code execution flaw in OpenSSL 3.0.1 TLS handshake handling."),
    Document("d2", "Buffer overflow in an unrelated logging library, CVE-2023-9999, affects version 1.2.0."),
    Document("d3", "General discussion of transport layer security best practices, no specific CVE mentioned."),
    Document("d4", "NIST control AC-2 covers account management; see also AC-2(1) for automated support."),
    Document("d5", "SC-7 boundary protection guidance for network segmentation in industrial control systems."),
]


def test_extract_exact_ids_finds_cve():
    ids = extract_exact_ids("See CVE-2024-1234 for details.")
    assert "CVE-2024-1234" in ids


def test_extract_exact_ids_finds_control_id():
    ids = extract_exact_ids("Reference control AC-2(1) in the policy.")
    assert "AC-2(1)" in ids


def test_hybrid_retriever_exact_id_boost_wins():
    retriever = HybridRetriever(DOCS, alpha=0.5, id_boost=1.0)
    results = retriever.search("What does CVE-2024-1234 affect?", k=5)
    top_id, _ = results[0]
    assert top_id == "d1"


def test_hybrid_retriever_does_not_confuse_similar_looking_cves():
    retriever = HybridRetriever(DOCS, alpha=0.5, id_boost=1.0)
    results = retriever.search("Tell me about CVE-2023-9999", k=5)
    top_id, _ = results[0]
    assert top_id == "d2"


def test_recall_at_k_basic():
    results = [("d1", 0.9), ("d2", 0.5), ("d3", 0.3)]
    assert recall_at_k(results, {"d1"}, k=3) == 1.0
    assert recall_at_k(results, {"d1"}, k=1) == 1.0
    assert recall_at_k(results, {"d4"}, k=3) == 0.0


def test_evaluate_recall_at_k_over_query_set():
    retriever = HybridRetriever(DOCS, alpha=0.5, id_boost=1.0)
    queries = [
        ("What does CVE-2024-1234 affect?", {"d1"}),
        ("Tell me about AC-2(1)", {"d4"}),
        ("boundary protection SC-7", {"d5"}),
    ]
    report = evaluate_recall_at_k(retriever, queries, k=3)
    assert report["n_queries"] == 3
    assert report["mean_recall_at_k"] == 1.0
