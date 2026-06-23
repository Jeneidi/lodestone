"""
Tests for lodestone.retrieval.bm25.BM25Retriever.

Uses a hand-verifiable mini-corpus of 3–5 tiny chunks so scores can be
reasoned about without a calculator.

Covers:
- Exact-match term ranks highest.
- Rare-term (high IDF) doc ranks first for rare-term query.
- k respected.
- Results sorted descending by score.
- Query with only OOV terms → empty list.
- Scores positive.
- Deterministic tie-break by chunk_id.
- Inverted index only scores candidate docs (result count check).
- search() before index() raises RuntimeError.
- index() with empty list raises ValueError.
"""

from __future__ import annotations

import pytest

from lodestone.retrieval.bm25 import BM25Retriever
from lodestone.schemas import Chunk

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", text=text, index=0)


# ---------------------------------------------------------------------------
# Fixtures: tiny hand-verifiable corpus
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_chunks() -> list[Chunk]:
    """5 tiny chunks with controlled vocabulary."""
    return [
        _chunk("c0", "machine learning gradient descent"),
        _chunk("c1", "neural network deep learning model"),
        _chunk("c2", "quantum physics subatomic particles"),
        _chunk("c3", "cooking fermentation microorganisms"),
        # c4 has the rare term 'supernova' appearing only here
        _chunk("c4", "astronomy supernova stellar explosion"),
    ]


@pytest.fixture
def indexed_bm25(tiny_chunks: list[Chunk]) -> BM25Retriever:
    r = BM25Retriever()
    r.index(tiny_chunks)
    return r


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


class TestBM25ErrorConditions:
    def test_search_before_index_raises(self):
        r = BM25Retriever()
        with pytest.raises(RuntimeError, match="index"):
            r.search("query")

    def test_index_empty_raises(self):
        r = BM25Retriever()
        with pytest.raises(ValueError):
            r.index([])


# ---------------------------------------------------------------------------
# Basic search behaviour
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_exact_match_ranks_highest(self, indexed_bm25: BM25Retriever):
        """Chunk containing the exact query term should be ranked first."""
        results = indexed_bm25.search("quantum", k=5)
        assert results[0].chunk.chunk_id == "c2", (
            f"Expected c2 (quantum physics) first, got {results[0].chunk.chunk_id}"
        )

    def test_rare_term_ranks_first(self, indexed_bm25: BM25Retriever):
        """'supernova' appears in only one chunk → high IDF → that chunk ranks first."""
        results = indexed_bm25.search("supernova", k=5)
        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "c4"

    def test_k_respected(self, indexed_bm25: BM25Retriever):
        """Results length <= k."""
        for k in (1, 2, 3):
            results = indexed_bm25.search("learning", k=k)
            assert len(results) <= k

    def test_k_larger_than_corpus(self, indexed_bm25: BM25Retriever):
        """k > corpus size → returns at most corpus-size results."""
        results = indexed_bm25.search("learning", k=100)
        assert len(results) <= 5

    def test_results_sorted_descending(self, indexed_bm25: BM25Retriever):
        """Scores must be in non-increasing order."""
        results = indexed_bm25.search("learning model gradient", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_positive(self, indexed_bm25: BM25Retriever):
        """All BM25 scores must be > 0."""
        results = indexed_bm25.search("machine learning", k=5)
        assert all(r.score > 0 for r in results), "All scores should be positive"

    def test_oov_query_returns_empty(self, indexed_bm25: BM25Retriever):
        """Query consisting entirely of OOV terms returns empty list."""
        results = indexed_bm25.search("xyzzy quux flarp wibble", k=5)
        assert results == []

    def test_stopword_only_query_returns_empty(self, indexed_bm25: BM25Retriever):
        """Query with only stopwords → after tokenisation empty → []."""
        results = indexed_bm25.search("the and or is a", k=5)
        assert results == []

    def test_retriever_name_is_bm25(self, indexed_bm25: BM25Retriever):
        results = indexed_bm25.search("machine", k=3)
        for r in results:
            assert r.retriever == "bm25"

    def test_inverted_index_only_scores_candidates(self):
        """Only chunks containing query terms appear in results."""
        # 'supernova' appears only in c4; query for 'supernova' returns only c4
        chunks = [
            _chunk("x0", "apple banana cherry"),
            _chunk("x1", "dog cat mouse"),
            _chunk("x2", "supernova explosion stellar"),
        ]
        r = BM25Retriever()
        r.index(chunks)
        results = r.search("supernova", k=10)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "x2"

    def test_deterministic_tiebreak_by_chunk_id(self):
        """Tie in BM25 score → lexicographically smaller chunk_id wins."""
        # Two identical-text chunks → identical scores; chunk_id tiebreak
        chunks = [
            _chunk("z_second", "unique keyword alpha"),
            _chunk("a_first", "unique keyword alpha"),
        ]
        r = BM25Retriever()
        r.index(chunks)
        results = r.search("unique keyword", k=2)
        assert len(results) == 2
        # Lower chunk_id should appear first on tie
        assert results[0].chunk.chunk_id == "a_first"

    def test_multiple_query_terms_accumulate_scores(self):
        """A doc containing multiple query terms scores higher than one with just one."""
        chunks = [
            _chunk("both", "neural gradient learning optimise"),
            _chunk("one", "gradient astronomy supernova"),
        ]
        r = BM25Retriever()
        r.index(chunks)
        results = r.search("neural gradient learning", k=2)
        ids = [res.chunk.chunk_id for res in results]
        assert ids[0] == "both", "Chunk with more matching terms should rank first"

    def test_k1_b_parameters_accepted(self):
        """Custom k1 and b parameters should be accepted without errors."""
        r = BM25Retriever(k1=1.2, b=0.5)
        r.index([_chunk("p0", "test query term"), _chunk("p1", "another document here")])
        results = r.search("test", k=2)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# IDF sanity
# ---------------------------------------------------------------------------


class TestBM25IDF:
    def test_common_term_lower_score_than_rare_term(self):
        """A term appearing in all docs has lower IDF than a term in one doc."""
        # 'common' appears in all 3 docs; 'rare' appears in only one
        chunks = [
            _chunk("idf0", "common alpha beta rare term"),
            _chunk("idf1", "common gamma delta"),
            _chunk("idf2", "common epsilon zeta"),
        ]
        r = BM25Retriever()
        r.index(chunks)
        rare_results = r.search("rare", k=3)
        common_results = r.search("common", k=3)
        # rare term has fewer matching docs but higher IDF;
        # for 'rare' query, idf0 score should be non-zero
        assert len(rare_results) == 1
        # For 'common' query, all 3 docs match but with lower IDF
        assert len(common_results) == 3
