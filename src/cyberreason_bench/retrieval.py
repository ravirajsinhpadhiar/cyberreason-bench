"""Exact-ID lexical + dense hybrid retrieval. Maps to Index.xlsx item 4.

Goal (from the index): combine exact-match lexical lookup (BM25) with dense
semantic retrieval so identifiers (CVE, version strings, control IDs) are
never missed by embedding-only search -- embeddings are good at "similar
meaning", bad at "this exact 11-character alphanumeric string".

SCAFFOLDING NOTE: the dense backend here is `TfidfDenseBackend`, a
TF-IDF + cosine-similarity stand-in. It is deliberately NOT a real
embedding model, so this runs offline/deterministically without a network
call or GPU. Item 4's plan names Qwen3-Embedding-0.6B as the candidate --
swap in a real embedding model by implementing `DenseBackend.embed()`
against it (see the class docstring below for the interface) before this
is used for a real recall@10 claim. The exact-ID boost logic and the
fusion/evaluation code around it are real and reusable regardless of which
dense backend you plug in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- exact-ID extraction ---------------------------------------------------

# CVE-YYYY-NNNN(+), NIST control ids like AC-2 or AC-2(1), and loose semver
# version strings (e.g. 3.0.1, 1.2). Deliberately conservative (few false
# positives) over exhaustive -- tighten/extend per real corpus as needed.
_ID_PATTERNS = [
    re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2}-\d{1,2}(?:\(\d+\))?(?![A-Za-z0-9])"),  # NIST control id
    re.compile(r"\b\d+\.\d+(?:\.\d+){0,2}\b"),                    # version string
]


def extract_exact_ids(text: str) -> set[str]:
    """Pull out CVE ids, NIST control ids, and version strings verbatim."""
    ids: set[str] = set()
    for pattern in _ID_PATTERNS:
        ids.update(m.group(0).upper() for m in pattern.finditer(text))
    return ids


# --- dense backend interface ------------------------------------------------

class DenseBackend(Protocol):
    """Implement this against a real embedding model to replace the
    TF-IDF stand-in. `fit` sees the corpus once; `embed` must be callable
    on new query strings after fit."""

    def fit(self, corpus_texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfDenseBackend:
    """Offline stand-in for a real dense embedding model. See module
    docstring -- swap this out before citing recall numbers as real."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus_texts: list[str]) -> None:
        self._vectorizer.fit(corpus_texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before embed()")
        return self._vectorizer.transform(texts).toarray()


# --- documents and hybrid index --------------------------------------------

@dataclass(frozen=True)
class Document:
    id: str
    text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\.\-]+", text.lower())


class HybridRetriever:
    """BM25 + dense fusion with an exact-ID short-circuit boost.

    Fusion: normalized BM25 score and normalized cosine similarity are
    combined as `alpha * bm25_norm + (1 - alpha) * dense_norm`. Any
    document containing an exact ID that also appears verbatim in the
    query gets an additive `id_boost` on top of the fused score, so an
    identifier match can never be out-ranked by a merely similar-sounding
    but wrong document.
    """

    def __init__(self, documents: list[Document], dense_backend: DenseBackend | None = None,
                 alpha: float = 0.5, id_boost: float = 1.0) -> None:
        if not documents:
            raise ValueError("HybridRetriever needs at least one document")
        self.documents = documents
        self.alpha = alpha
        self.id_boost = id_boost

        self._tokenized_corpus = [_tokenize(d.text) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        self._dense = dense_backend or TfidfDenseBackend()
        self._dense.fit([d.text for d in documents])
        self._doc_ids = [d.id for d in documents]
        self._doc_exact_ids = [extract_exact_ids(d.text) for d in documents]

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)))
        dense_vecs = self._dense.embed([query])
        # TfidfDenseBackend returns doc-space vectors; recompute doc matrix
        # once per call is wasteful for large corpora -- fine for this
        # scaffold's scale, revisit if the real corpus grows large.
        doc_vecs = self._dense.embed([d.text for d in self.documents])
        dense_scores = cosine_similarity(dense_vecs, doc_vecs)[0]

        fused = self.alpha * self._normalize(bm25_scores) + (1 - self.alpha) * self._normalize(dense_scores)

        query_ids = extract_exact_ids(query)
        if query_ids:
            for i, doc_ids in enumerate(self._doc_exact_ids):
                if query_ids & doc_ids:
                    fused[i] += self.id_boost

        order = np.argsort(-fused)[:k]
        return [(self._doc_ids[i], float(fused[i])) for i in order]


# --- evaluation --------------------------------------------------------------

def recall_at_k(results: list[tuple[str, float]], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of `relevant_ids` present in the top-k of `results`."""
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")
    top_ids = {doc_id for doc_id, _ in results[:k]}
    return len(top_ids & relevant_ids) / len(relevant_ids)


def evaluate_recall_at_k(
    retriever: HybridRetriever,
    queries: list[tuple[str, set[str]]],  # (query_text, relevant_doc_ids)
    k: int = 10,
) -> dict[str, float]:
    """Mean recall@k over a query set. Returns per-query and mean scores."""
    per_query = []
    for query_text, relevant in queries:
        results = retriever.search(query_text, k=k)
        per_query.append(recall_at_k(results, relevant, k=k))
    return {
        "mean_recall_at_k": float(np.mean(per_query)) if per_query else 0.0,
        "n_queries": len(per_query),
        "k": k,
    }
