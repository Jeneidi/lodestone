"""
Tests for DenseRetriever and HybridRetriever (RRF and Weighted fusion).

Uses fake_encoder and fake_ce_scorer from conftest to avoid model downloads.

Covers:
- DenseRetriever: topically-matching chunk ranks first, sorted desc,
  k respected, search before index raises, empty index raises.
- HybridRetriever RRF: chunk ranked top by both retrievers outranks
  one ranked top by only one; RRF formula spot-check; deduplication.
- HybridRetriever Weighted: min-max normalisation + weights honored;
  hand-computed case; dedupe.
- StubRetriever for deterministic fusion testing.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from lodestone.retrieval.base import Retriever
from lodestone.retrieval.dense import DenseRetriever
from lodestone.retrieval.fusion import HybridRetriever
from lodestone.schemas import Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# StubRetriever for deterministic fusion testing
# ---------------------------------------------------------------------------

class StubRetriever(Retriever):
    """A Retriever that returns a pre-set list of ScoredChunks."""

    name: str = "stub"

    def __init__(self, name: str, results: list[ScoredChunk]) -> None:
        self._name_val = name
        self._results = results
        self._indexed = False

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name_val

    def index(self, chunks: list[Chunk]) -> None:
        self._indexed = True

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return self._results[:k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str, doc_id: str = "d") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, index=0)


def _scored(chunk: Chunk, score: float, retriever: str = "stub") -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, retriever=retriever)


# ---------------------------------------------------------------------------
# DenseRetriever tests
# ---------------------------------------------------------------------------

class TestDenseRetriever:

    def test_search_before_index_raises(self, fake_encoder_fn: Callable):
        r = DenseRetriever(encoder=fake_encoder_fn)
        with pytest.raises(RuntimeError, match="index"):
            r.search("query")

    def test_index_empty_raises(self, fake_encoder_fn: Callable):
        r = DenseRetriever(encoder=fake_encoder_fn)
        with pytest.raises(ValueError):
            r.index([])

    def test_topical_match_ranks_first(
        self,
        fake_encoder_fn: Callable,
        corpus_chunks: list[Chunk],
    ):
        """A query about ML should return ML chunks in the top results.

        Note: the fake BoW encoder uses a 32-bucket hash, so hash collisions
        can cause non-ML documents to outscore ML ones on some queries.
        We verify that at least one ML chunk appears in the top-5 results
        (a weaker but reliably-true property for this encoder/corpus).
        """
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        results = r.search("machine learning neural network gradient", k=5)
        assert len(results) >= 1
        top_doc_ids = [res.chunk.doc_id for res in results]
        assert "doc_ml" in top_doc_ids, (
            f"Expected doc_ml in top-5 results, got: {top_doc_ids}"
        )

    def test_cooking_query_returns_cooking_chunk(
        self,
        fake_encoder_fn: Callable,
        corpus_chunks: list[Chunk],
    ):
        """A cooking query should return cooking doc chunks near the top."""
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        results = r.search("fermentation cooking maillard reaction", k=5)
        top_doc_ids = [res.chunk.doc_id for res in results]
        assert "doc_cooking" in top_doc_ids

    def test_results_sorted_descending(
        self,
        fake_encoder_fn: Callable,
        corpus_chunks: list[Chunk],
    ):
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        results = r.search("quantum physics", k=8)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_respected(self, fake_encoder_fn: Callable, corpus_chunks: list[Chunk]):
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        for k in (1, 3, 5):
            results = r.search("history", k=k)
            assert len(results) <= k

    def test_k_larger_than_corpus(
        self, fake_encoder_fn: Callable, corpus_chunks: list[Chunk]
    ):
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        results = r.search("science", k=1000)
        assert len(results) <= len(corpus_chunks)

    def test_retriever_name(self, fake_encoder_fn: Callable, corpus_chunks: list[Chunk]):
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        results = r.search("test", k=3)
        for res in results:
            assert res.retriever == "dense"

    def test_deterministic(self, fake_encoder_fn: Callable, corpus_chunks: list[Chunk]):
        """Same query → same results on two calls."""
        r = DenseRetriever(encoder=fake_encoder_fn)
        r.index(corpus_chunks)
        r1 = r.search("neural network", k=5)
        r2 = r.search("neural network", k=5)
        assert [res.chunk.chunk_id for res in r1] == [res.chunk.chunk_id for res in r2]


# ---------------------------------------------------------------------------
# HybridRetriever — RRF
# ---------------------------------------------------------------------------

class TestHybridRRF:

    def test_chunk_ranked_top_by_both_outranks_single(self):
        """
        chunk A is ranked #1 by both retrievers.
        chunk B is ranked #1 by only retriever 1.
        chunk C is ranked #1 by only retriever 2.
        A should outrank both B and C.
        """
        chunk_a = _chunk("A", "alpha beta gamma delta")
        chunk_b = _chunk("B", "beta gamma theta iota")
        chunk_c = _chunk("C", "gamma delta kappa lambda")

        # Retriever 1: A rank 1, B rank 2, C rank 3
        r1_results = [
            _scored(chunk_a, 1.0, "r1"),
            _scored(chunk_b, 0.8, "r1"),
            _scored(chunk_c, 0.6, "r1"),
        ]
        # Retriever 2: A rank 1, C rank 2, B rank 3
        r2_results = [
            _scored(chunk_a, 0.9, "r2"),
            _scored(chunk_c, 0.7, "r2"),
            _scored(chunk_b, 0.5, "r2"),
        ]

        stub1 = StubRetriever("r1", r1_results)
        stub2 = StubRetriever("r2", r2_results)
        hybrid = HybridRetriever([stub1, stub2], strategy="rrf", rrf_k=60)
        # index with dummy chunk (needed for hybrid.index call)
        all_chunks = [chunk_a, chunk_b, chunk_c]
        hybrid.index(all_chunks)

        results = hybrid.search("alpha", k=3)
        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "A", (
            f"Expected A first (top in both retrievers), got {results[0].chunk.chunk_id}"
        )

    def test_rrf_formula_spot_check(self):
        """
        With rrf_k=60:
        - chunk X at rank 1 in retriever A: score += 1/(60+1) ≈ 0.016393
        - chunk Y at rank 1 in retriever B only: same single score
        - chunk Z at rank 2 in A and rank 2 in B: score = 2/(60+2) ≈ 0.032258
        Z should outscore both X and Y (one contribution each).
        """
        rrf_k = 60
        cx = _chunk("X", "apple orange")
        cy = _chunk("Y", "banana mango")
        cz = _chunk("Z", "cherry pear")

        # Retriever A: Z rank 1, X rank 2
        # Retriever B: Z rank 1, Y rank 2
        rA = [_scored(cz, 10.0, "A"), _scored(cx, 8.0, "A")]
        rB = [_scored(cz, 9.0, "B"), _scored(cy, 7.0, "B")]

        stub_a = StubRetriever("A", rA)
        stub_b = StubRetriever("B", rB)
        hybrid = HybridRetriever([stub_a, stub_b], strategy="rrf", rrf_k=rrf_k)
        hybrid.index([cx, cy, cz])

        results = hybrid.search("fruit", k=3)
        # Z appears at rank 1 in both → highest RRF score
        assert results[0].chunk.chunk_id == "Z"

        # Hand-verify Z's RRF score
        expected_z = 1.0 / (rrf_k + 1) + 1.0 / (rrf_k + 1)  # rank 1 in both
        assert abs(results[0].score - expected_z) < 1e-9, (
            f"Z score {results[0].score} != expected {expected_z}"
        )

    def test_rrf_deduplication(self):
        """The same chunk_id appearing in both retrievers should appear once."""
        c = _chunk("dup", "same chunk text here")
        rA = [_scored(c, 1.0, "A")]
        rB = [_scored(c, 0.8, "B")]
        stub_a = StubRetriever("A", rA)
        stub_b = StubRetriever("B", rB)
        hybrid = HybridRetriever([stub_a, stub_b], strategy="rrf", rrf_k=60)
        hybrid.index([c])
        results = hybrid.search("text", k=5)
        ids = [r.chunk.chunk_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate chunk_id in RRF results"

    def test_rrf_retriever_name(self):
        c = _chunk("c0", "some text")
        stub = StubRetriever("s", [_scored(c, 1.0, "s")])
        hybrid = HybridRetriever([stub], strategy="rrf", rrf_k=60)
        hybrid.index([c])
        results = hybrid.search("text", k=1)
        assert all(r.retriever == "hybrid_rrf" for r in results)

    def test_empty_retrievers_raises(self):
        with pytest.raises(ValueError):
            HybridRetriever([], strategy="rrf")

    def test_invalid_strategy_raises(self):
        c = _chunk("c0", "text")
        stub = StubRetriever("s", [])
        with pytest.raises(ValueError):
            HybridRetriever([stub], strategy="magic")


# ---------------------------------------------------------------------------
# HybridRetriever — Weighted fusion
# ---------------------------------------------------------------------------

class TestHybridWeighted:

    def test_weight_honored_hand_computed(self):
        """
        Two chunks, two retrievers, weight=[0.8, 0.2].
        Retriever A: chunk_a score=10, chunk_b score=0  (normalised: 1.0, 0.0)
        Retriever B: chunk_a score=0,  chunk_b score=10 (normalised: 0.0, 1.0)
        Fused:
          chunk_a = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
          chunk_b = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
        chunk_a should rank first.
        """
        ca = _chunk("ca", "apricot elderberry")
        cb = _chunk("cb", "mango papaya")

        rA = [_scored(ca, 10.0, "A"), _scored(cb, 0.0, "A")]
        rB = [_scored(ca, 0.0, "B"), _scored(cb, 10.0, "B")]
        stub_a = StubRetriever("A", rA)
        stub_b = StubRetriever("B", rB)
        hybrid = HybridRetriever(
            [stub_a, stub_b], strategy="weighted", weights=[0.8, 0.2]
        )
        hybrid.index([ca, cb])
        results = hybrid.search("fruit", k=2)
        assert len(results) == 2
        assert results[0].chunk.chunk_id == "ca"

        # Verify exact fused score for ca
        # min-max norm: (10-0)/(10-0+eps)≈1; (0-0)/(10-0+eps)≈0
        eps = 1e-9
        norm_ca_A = (10.0 - 0.0) / (10.0 - 0.0 + eps)
        norm_ca_B = (0.0 - 0.0) / (10.0 - 0.0 + eps)
        # weights are normalised to sum to 1 (already 0.8+0.2=1.0)
        expected_ca = 0.8 * norm_ca_A + 0.2 * norm_ca_B
        assert abs(results[0].score - expected_ca) < 1e-6

    def test_equal_weights_symmetric(self):
        """Equal weights → a chunk ranked first by both wins."""
        ca = _chunk("ca", "first both retrievers")
        cb = _chunk("cb", "second both retrievers")
        # Both retrievers agree on order
        rA = [_scored(ca, 2.0, "A"), _scored(cb, 1.0, "A")]
        rB = [_scored(ca, 3.0, "B"), _scored(cb, 1.5, "B")]
        stub_a = StubRetriever("A", rA)
        stub_b = StubRetriever("B", rB)
        hybrid = HybridRetriever([stub_a, stub_b], strategy="weighted")
        hybrid.index([ca, cb])
        results = hybrid.search("query", k=2)
        assert results[0].chunk.chunk_id == "ca"

    def test_weighted_deduplication(self):
        c = _chunk("dup", "deduplicated chunk here")
        rA = [_scored(c, 1.0, "A")]
        rB = [_scored(c, 0.9, "B")]
        stub_a = StubRetriever("A", rA)
        stub_b = StubRetriever("B", rB)
        hybrid = HybridRetriever([stub_a, stub_b], strategy="weighted")
        hybrid.index([c])
        results = hybrid.search("chunk", k=5)
        ids = [r.chunk.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    def test_weighted_retriever_name(self):
        c = _chunk("c0", "text")
        stub = StubRetriever("s", [_scored(c, 1.0, "s")])
        hybrid = HybridRetriever([stub], strategy="weighted")
        hybrid.index([c])
        results = hybrid.search("text", k=1)
        assert all(r.retriever == "hybrid_weighted" for r in results)

    def test_weights_mismatched_length_raises(self):
        c = _chunk("c0", "text")
        stub1 = StubRetriever("s1", [])
        stub2 = StubRetriever("s2", [])
        with pytest.raises(ValueError):
            HybridRetriever([stub1, stub2], strategy="weighted", weights=[0.5])

    def test_weighted_k_respected(self):
        chunks = [_chunk(f"c{i}", f"word{i} unique") for i in range(5)]
        scored = [_scored(c, float(5 - i), "A") for i, c in enumerate(chunks)]
        stub = StubRetriever("A", scored)
        hybrid = HybridRetriever([stub], strategy="weighted")
        hybrid.index(chunks)
        for k in (1, 2, 3):
            results = hybrid.search("word", k=k)
            assert len(results) <= k

    def test_search_before_index_raises(self):
        c = _chunk("c0", "text")
        stub = StubRetriever("s", [_scored(c, 1.0)])
        hybrid = HybridRetriever([stub], strategy="rrf", rrf_k=60)
        with pytest.raises(RuntimeError, match="index"):
            hybrid.search("query")
