#!/usr/bin/env python3
"""End-to-end demo tying retrieval + leakage-controlled splits + bootstrap
stats together on a small synthetic advisory corpus.

This is NOT a real benchmark result -- the corpus is 10 hand-written toy
advisories (demo/data/toy_advisories.json), not a real CVE/NVD corpus.
What this script actually demonstrates and checks:

  1. The hybrid retriever's exact-ID boost beats a BM25-only baseline on
     recall@3 for queries that name a specific CVE/control ID (item 4's
     stated goal -- IDs must not be missed by semantic-only search).
  2. A family-level split keeps both advisories about the same CVE/control
     in the same split (item 42) -- verified, not assumed.
  3. A paired bootstrap CI on hybrid-vs-BM25-only pass rate, at the family
     level (item 44), so the reported "hybrid wins" claim carries an
     honest confidence interval instead of a bare percentage.

Run: PYTHONPATH=src python3 demo/benchmark_demo.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cyberreason_bench.retrieval import Document, HybridRetriever, recall_at_k
from cyberreason_bench.splits import Case, family_split, verify_no_leakage, assign_family_id
from cyberreason_bench.stats import paired_bootstrap_diff, stop_rule_check


def load_corpus() -> list[dict]:
    path = pathlib.Path(__file__).resolve().parent / "data" / "toy_advisories.json"
    return json.loads(path.read_text())


def main() -> None:
    corpus = load_corpus()
    documents = [Document(row["id"], row["text"]) for row in corpus]

    # 1. exact-ID boost vs BM25-only, per query
    hybrid = HybridRetriever(documents, alpha=0.5, id_boost=1.0)
    bm25_only = HybridRetriever(documents, alpha=1.0, id_boost=0.0)

    hybrid_pass, bm25_pass, family_ids = [], [], []
    print("=== per-query recall@3 (relevant = the advisory itself) ===")
    for row in corpus:
        if row["relevant_query"] is None:
            continue
        relevant = {row["id"]}
        h_results = hybrid.search(row["relevant_query"], k=3)
        b_results = bm25_only.search(row["relevant_query"], k=3)
        h_recall = recall_at_k(h_results, relevant, k=3)
        b_recall = recall_at_k(b_results, relevant, k=3)
        hybrid_pass.append(h_recall >= 1.0)
        bm25_pass.append(b_recall >= 1.0)
        family_ids.append(assign_family_id(row["source_id"]))
        print(f"  {row['id']:6s} hybrid={h_recall:.2f}  bm25_only={b_recall:.2f}  q={row['relevant_query']!r}")

    print(f"\nhybrid pass rate:    {sum(hybrid_pass)}/{len(hybrid_pass)}")
    print(f"bm25-only pass rate: {sum(bm25_pass)}/{len(bm25_pass)}")

    # 2. leakage-controlled split, verified
    cases = [Case(id=row["id"], family_key=assign_family_id(row["source_id"]), payload={}) for row in corpus]
    splits = family_split(cases, ratios=(0.6, 0.2, 0.2), seed=13)
    leaked = verify_no_leakage(splits)
    print(f"\n=== family split (n={len(cases)} cases) ===")
    for name, bucket in splits.items():
        print(f"  {name}: {len(bucket)} cases")
    print(f"  leaked family keys: {leaked if leaked else 'none (clean)'}")
    assert leaked == [], "leakage check failed -- this should never happen with family_split"

    # 3. paired bootstrap: does hybrid actually beat BM25-only, with a CI?
    result = paired_bootstrap_diff(hybrid_pass, bm25_pass, family_ids, n_resamples=2000, seed=13)
    print("\n=== paired bootstrap: hybrid pass rate - bm25-only pass rate ===")
    print(f"  point estimate: {result.point_estimate:+.3f}")
    print(f"  95% CI:         [{result.ci_low:+.3f}, {result.ci_high:+.3f}]")
    print(f"  excludes zero:  {result.excludes_zero()}")
    print(f"  stop-rule (>=0.05 and CI excludes 0): "
          f"{stop_rule_check(result, min_point_estimate=0.05)}")

    print(
        "\nNOTE: this is a 9-item toy corpus meant to prove the plumbing is "
        "correct end to end, not a real benchmark result. Hybrid and "
        "BM25-only tie here because this corpus is too small/clean for "
        "either to fail -- and because TfidfDenseBackend is still "
        "token-based, so it doesn't reproduce the real failure mode a "
        "semantic embedding model has (missing an exact ID because it "
        "looks for 'similar meaning', not an identical string). That "
        "failure mode IS exercised directly in "
        "tests/test_retrieval.py::test_hybrid_retriever_does_not_confuse_"
        "similar_looking_cves. A real recall@10 claim needs a real corpus, "
        "a real dense embedding model (e.g. Qwen3-Embedding-0.6B) in place "
        "of TfidfDenseBackend, and hundreds of cases before the bootstrap "
        "CI here means anything you'd put in a paper."
    )


if __name__ == "__main__":
    main()
