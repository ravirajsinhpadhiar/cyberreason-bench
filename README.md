# cyberreason-bench

Evaluation-harness scaffolding for the **Cyber Reason LM** research programme.
This is engineering infrastructure, not a paper by itself — it exists so the
"Core / v1" engineering items in the research Index.xlsx have real, tested
code under them instead of just a plan.

Built overnight (2026-09-28/29) to make direct use of session compute
rather than more idea-scoping — two research directions (Jev security study,
YARA-RegGraph) were evaluated the same night and both came back as no-go /
not-standalone (already scooped, or a subsection of existing work — see the
project chat for the full memos). This repo is the pivot: build the
infrastructure several already-scoped index items agree is needed regardless
of which paper ships first.

## What's here

| Module | Index item | What it does |
|---|---|---|
| `src/cyberreason_bench/schema.py` | **2** | Provenance-aware typed graph schema — every edge carries `ProvenanceKind` (explicit / deterministic-derived / expert-assessed / model-proposed), source, date, reviewer, uncertainty. `GraphStore.neighbors()` can filter to only high-trust provenance kinds. Includes a stub for item 3's "as of" temporal queries (`valid_from`/`valid_until` per edge) — **not** a full temporal-graph design. |
| `src/cyberreason_bench/retrieval.py` | **4** | BM25 + dense fusion (`HybridRetriever`) with an exact-ID boost so CVE/NIST-control/version identifiers can't be out-ranked by a merely similar-sounding wrong document. `recall_at_k` / `evaluate_recall_at_k` for eval. |
| `src/cyberreason_bench/splits.py` | **42** | Family-level train/dev/test splitting (`family_split`) so no shared-source leakage (same CVE, same advisory, same policy doc) crosses a split. `verify_no_leakage` and `split_manifest` for an auditable record. |
| `src/cyberreason_bench/stats.py` | **44** | Grouped/paired bootstrap (`grouped_bootstrap_pass_rate`, `paired_bootstrap_diff`) at the family level, `>=2000` resamples by default, plus `stop_rule_check` so a "system A beats system B" claim requires an explicit, prespecified threshold rather than eyeballing a CI. |

`demo/benchmark_demo.py` wires all four together on a 10-item synthetic
advisory corpus (`demo/data/toy_advisories.json`) — run it to see the full
pipeline (retrieve → split → bootstrap-compare) execute end to end.

## What this is *not* (read before citing anything from it)

- **Not a real benchmark result.** The demo corpus is 10 hand-written toy
  advisories. Every number `benchmark_demo.py` prints is a plumbing check,
  not a finding.
- **The "dense" retrieval backend is TF-IDF, not a real embedding model.**
  `TfidfDenseBackend` in `retrieval.py` is an offline, deterministic stand-in
  so this runs without a network call or GPU. It does **not** reproduce the
  actual failure mode item 4 is about (a semantic embedding model missing an
  exact identifier because it's optimizing for "similar meaning"). That
  failure mode is exercised directly in a unit test
  (`test_hybrid_retriever_does_not_confuse_similar_looking_cves`), but a real
  recall@10 claim needs a real embedding model (Qwen3-Embedding-0.6B is the
  plan's named candidate) swapped in via the `DenseBackend` protocol.
- **`schema.py`'s temporal support is a stub, not item 3.** It's a single
  `valid_from`/`valid_until` pair per edge. Item 3 (event time vs
  publication time vs ingestion time, "as of" queries with full versioning)
  is real, separate, unfinished work.
- **`assign_family_id` only handles exact shared source IDs.** Real
  duplicate/paraphrase detection across differently-worded documents about
  the same incident isn't implemented — extend it before trusting a split
  on a corpus with paraphrased duplicates.
- **In-memory only.** `GraphStore` is a dict-backed store for development
  and testing, not a production graph database (Neo4j/RDFLib/NetworkX).

Each module's own docstring repeats its specific scaffolding notes — read
those before building on top of a given piece.

## Running it

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v            # 25 tests, all passing
PYTHONPATH=src python3 demo/benchmark_demo.py
```

## Mapping back to the research programme

This covers 4 of the ~63 items in the Index.xlsx research backlog — the
ones flagged as "engineering task, well specified" or "Core / v1" that
don't depend on partner data access, expert-curated control selections, or
a corpus that doesn't exist yet. It deliberately does not attempt the
flagship items (CyberReasonBench itself, item 41; the RQ1-3 experiments,
items 1/9/22) — those need real data and real experiment design, not
overnight scaffolding.
