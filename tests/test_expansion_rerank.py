"""
Tests for lodestone.retrieval.expansion and lodestone.retrieval.rerank.

Uses a StubRetriever defined in-file (returns fixed ScoredChunk lists) so no
real BM25/dense models are needed.  Fully offline and deterministic.

Coverage
--------
Rm3QueryExpander:
- expansion adds high-frequency non-stopword terms from feedback docs
- original query string is preserved as a prefix
- fb_terms limit is respected
- terms already in the query are excluded from expansion
- empty feedback list → original query returned unchanged

ExpandingRetriever:
- delegates search to inner retriever (via expander)
- name is "rm3+{inner.name}"
- satisfies the Retriever interface (index / search contract)
- re-tags retriever field to its own name

CrossEncoderReranker:
- reorders candidates by injected scorer
- top_k respected
- all scores in (0, 1) after sigmoid
- retriever name updated to "rerank({original})"
- empty candidates → empty list
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from lodestone.retrieval.base import Retriever
from lodestone.retrieval.expansion import ExpandingRetriever, Rm3QueryExpander
from lodestone.retrieval.rerank import CrossEncoderReranker
from lodestone.schemas import Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# StubRetriever — in-file, fixed responses
# ---------------------------------------------------------------------------


class StubRetriever(Retriever):
    """Retriever that returns a pre-configured list of ScoredChunks.

    The ``fixed_results`` list is returned for *every* query (truncated to k).
    Supports recording the last query passed to search() for assertions.
    """

    name: str = "stub"

    def __init__(
        self,
        results: list[ScoredChunk],
        name: str = "stub",
    ) -> None:
        self._results = results
        self.name = name  # type: ignore[assignment]
        self.last_query: str | None = None
        self._indexed = False

    def index(self, chunks: list[Chunk]) -> None:  # noqa: ARG002
        self._indexed = True

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        self.last_query = query
        return self._results[:k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", text=text, index=0)


def _scored(chunk_id: str, text: str, score: float = 1.0, retriever: str = "stub") -> ScoredChunk:
    return ScoredChunk(
        chunk=_chunk(chunk_id, text),
        score=score,
        retriever=retriever,
    )


# ---------------------------------------------------------------------------
# Rm3QueryExpander tests
# ---------------------------------------------------------------------------


class TestRm3QueryExpander:
    """Tests for the RM3 pseudo-relevance feedback expander."""

    # Feedback corpus: rich ML vocabulary, term "gradient" appears most often
    _ML_CHUNKS: list[ScoredChunk] = [
        _scored("c0", "gradient descent optimises loss function gradient gradient"),
        _scored("c1", "neural network weights backpropagation gradient"),
        _scored("c2", "learning rate schedule gradient descent convergence"),
    ]

    def _expander(
        self,
        results: list[ScoredChunk] | None = None,
        fb_docs: int = 3,
        fb_terms: int = 5,
    ) -> Rm3QueryExpander:
        chunks = results if results is not None else self._ML_CHUNKS
        stub = StubRetriever(chunks)
        return Rm3QueryExpander(stub, fb_docs=fb_docs, fb_terms=fb_terms)

    # ------------------------------------------------------------------
    # Expansion adds terms from feedback documents
    # ------------------------------------------------------------------

    def test_expansion_adds_terms(self):
        """Expanded query has more tokens than the original query."""
        expander = self._expander()
        expanded = expander.expand("machine learning")
        assert len(expanded.split()) > len("machine learning".split()), (
            f"Expected more tokens after expansion, got: {expanded!r}"
        )

    def test_original_query_preserved_as_prefix(self):
        """The original query string must be a prefix of the expanded query."""
        query = "machine learning"
        expander = self._expander()
        expanded = expander.expand(query)
        assert expanded.startswith(query), (
            f"Expected expanded query to start with original query.\n"
            f"  original: {query!r}\n"
            f"  expanded: {expanded!r}"
        )

    def test_high_frequency_term_included(self):
        """'gradient' is the most common non-stopword → must appear in expansion."""
        expander = self._expander()
        expanded = expander.expand("machine learning")
        expansion_part = expanded[len("machine learning") :].lower()
        assert "gradient" in expansion_part, (
            f"Expected 'gradient' in expansion terms; got: {expanded!r}"
        )

    def test_fb_terms_limit_respected(self):
        """Number of appended expansion terms must not exceed fb_terms."""
        fb_terms = 3
        expander = self._expander(fb_terms=fb_terms)
        expanded = expander.expand("xyz")  # 'xyz' not in feedback vocab
        original_token_count = 1
        total_tokens = len(expanded.split())
        appended = total_tokens - original_token_count
        assert appended <= fb_terms, (
            f"Expected at most {fb_terms} expansion terms, got {appended}: {expanded!r}"
        )

    def test_query_terms_not_duplicated_in_expansion(self):
        """Terms already in the query must not appear in the expansion."""
        # Query already contains 'gradient' and 'descent'
        query = "gradient descent"
        expander = self._expander()
        expanded = expander.expand(query)
        # Extract expanded-only portion
        expansion_only = expanded[len(query) :].strip().lower().split()
        assert "gradient" not in expansion_only, (
            f"'gradient' was already in query but appears in expansion: {expansion_only}"
        )
        assert "descent" not in expansion_only, (
            f"'descent' was already in query but appears in expansion: {expansion_only}"
        )

    def test_empty_feedback_returns_original_query(self):
        """When feedback list is empty the original query is returned unchanged."""
        stub = StubRetriever([])
        expander = Rm3QueryExpander(stub, fb_docs=5, fb_terms=5)
        query = "some interesting question"
        result = expander.expand(query)
        assert result == query, f"Expected original query unchanged, got: {result!r}"

    def test_stopwords_not_in_expansion(self):
        """Stopwords from the feedback corpus must not appear as expansion terms."""
        from lodestone.retrieval.tokenize import STOPWORDS

        stopword_heavy_chunks = [
            _scored("sw0", "the and is of with by from"),
            _scored("sw1", "this that these those it a an the"),
        ]
        stub = StubRetriever(stopword_heavy_chunks)
        expander = Rm3QueryExpander(stub, fb_docs=5, fb_terms=10)
        expanded = expander.expand("test query")
        expansion_only = expanded[len("test query") :].strip().lower().split()
        for term in expansion_only:
            assert term not in STOPWORDS, f"Stopword '{term}' should not appear in expansion terms."


# ---------------------------------------------------------------------------
# ExpandingRetriever tests
# ---------------------------------------------------------------------------


class TestExpandingRetriever:
    """Tests for the RM3-wrapping ExpandingRetriever."""

    _RESULTS: list[ScoredChunk] = [
        _scored("r0", "gradient descent optimises gradient", score=0.9),
        _scored("r1", "neural network weights gradient", score=0.7),
    ]

    def _make(self, inner_name: str = "stub") -> tuple[StubRetriever, ExpandingRetriever]:
        stub = StubRetriever(self._RESULTS, name=inner_name)
        expander = Rm3QueryExpander(stub, fb_docs=2, fb_terms=3)
        expanding = ExpandingRetriever(stub, expander)
        return stub, expanding

    # ------------------------------------------------------------------

    def test_name_is_rm3_plus_inner_name(self):
        """name must be 'rm3+{inner.name}'."""
        stub, expanding = self._make(inner_name="bm25")
        assert expanding.name == "rm3+bm25"

    def test_name_default_inner(self):
        stub, expanding = self._make(inner_name="stub")
        assert expanding.name == "rm3+stub"

    def test_satisfies_retriever_interface(self):
        """ExpandingRetriever is a Retriever subclass."""
        _, expanding = self._make()
        assert isinstance(expanding, Retriever)

    def test_index_delegates_to_inner(self):
        """Calling index() on ExpandingRetriever sets inner._indexed."""
        stub, expanding = self._make()
        assert not stub._indexed
        expanding.index([_chunk("c0", "dummy text")])
        assert stub._indexed

    def test_search_returns_scored_chunks(self):
        """search() returns a non-empty list of ScoredChunk objects."""
        _, expanding = self._make()
        results = expanding.search("machine learning", k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        for item in results:
            assert isinstance(item, ScoredChunk)

    def test_search_retriever_name_is_own_name(self):
        """Each returned ScoredChunk must carry the ExpandingRetriever's name."""
        stub, expanding = self._make(inner_name="bm25")
        results = expanding.search("machine learning", k=5)
        for sc in results:
            assert sc.retriever == "rm3+bm25", (
                f"Expected retriever='rm3+bm25', got {sc.retriever!r}"
            )

    def test_k_respected(self):
        """search() must return at most k results."""
        _, expanding = self._make()
        for k in (1, 2):
            results = expanding.search("machine learning", k=k)
            assert len(results) <= k, f"Expected ≤ {k} results, got {len(results)}"

    def test_delegates_search_through_expander(self):
        """The stub's last_query should differ from the original query
        (expansion added terms) or be the original if no expansion found."""
        stub, expanding = self._make()
        expanding.search("machine learning", k=5)
        # The expanded query was passed to the stub; it starts with original query
        assert stub.last_query is not None
        assert stub.last_query.startswith("machine learning")


# ---------------------------------------------------------------------------
# CrossEncoderReranker tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker using injected scorer callables."""

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _constant_scorer(value: float) -> Callable[[list[tuple[str, str]]], np.ndarray]:
        """Returns a scorer that always outputs the same logit value."""

        def scorer(pairs: list[tuple[str, str]]) -> np.ndarray:
            return np.full(len(pairs), value, dtype=np.float32)

        return scorer

    @staticmethod
    def _index_scorer(pairs: list[tuple[str, str]]) -> np.ndarray:
        """Scorer that assigns logit = negative index (first pair → highest logit)."""
        return np.array([-float(i) for i in range(len(pairs))], dtype=np.float32)

    @staticmethod
    def _reverse_scorer(pairs: list[tuple[str, str]]) -> np.ndarray:
        """Scorer that assigns logit = index (last pair → highest logit)."""
        return np.array([float(i) for i in range(len(pairs))], dtype=np.float32)

    # A small fixed candidate list for reuse
    _CANDIDATES: list[ScoredChunk] = [
        _scored("cA", "apple banana cherry", score=0.5, retriever="bm25"),
        _scored("cB", "delta echo foxtrot", score=0.4, retriever="bm25"),
        _scored("cC", "golf hotel india", score=0.3, retriever="bm25"),
        _scored("cD", "juliet kilo lima", score=0.2, retriever="bm25"),
        _scored("cE", "mike november oscar", score=0.1, retriever="bm25"),
    ]

    # ------------------------------------------------------------------
    # Empty candidates
    # ------------------------------------------------------------------

    def test_empty_candidates_returns_empty_list(self):
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(0.0))
        result = reranker.rerank("any query", [], top_k=5)
        assert result == []

    # ------------------------------------------------------------------
    # top_k
    # ------------------------------------------------------------------

    def test_top_k_respected(self):
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(1.0))
        for k in (1, 2, 3):
            result = reranker.rerank("test", self._CANDIDATES, top_k=k)
            assert len(result) == k, f"Expected {k} results, got {len(result)}"

    def test_top_k_larger_than_candidates_returns_all(self):
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(0.0))
        result = reranker.rerank("test", self._CANDIDATES, top_k=100)
        assert len(result) == len(self._CANDIDATES)

    # ------------------------------------------------------------------
    # Scores in (0, 1)
    # ------------------------------------------------------------------

    def test_scores_in_zero_one(self):
        """Sigmoid of any finite logit must be in (0, 1)."""
        for logit_value in (-10.0, -1.0, 0.0, 1.0, 10.0):
            reranker = CrossEncoderReranker(scorer=self._constant_scorer(logit_value))
            results = reranker.rerank("test", self._CANDIDATES[:3], top_k=3)
            for sc in results:
                assert 0.0 < sc.score < 1.0, (
                    f"Score {sc.score} is out of (0, 1) for logit={logit_value}"
                )

    def test_sigmoid_applied_correctly(self):
        """Verify sigmoid formula: score = 1 / (1 + exp(-logit))."""
        logit = 2.0
        expected = 1.0 / (1.0 + math.exp(-logit))
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(logit))
        results = reranker.rerank("test", self._CANDIDATES[:1], top_k=1)
        assert abs(results[0].score - expected) < 1e-5

    # ------------------------------------------------------------------
    # Ordering / reranking
    # ------------------------------------------------------------------

    def test_reranks_by_scorer_descending(self):
        """Results must be sorted by reranker score (highest first)."""
        # _reverse_scorer: last candidate gets highest logit → should rank first
        reranker = CrossEncoderReranker(scorer=self._reverse_scorer)
        results = reranker.rerank("test", self._CANDIDATES, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), f"Results not sorted descending: {scores}"

    def test_reranker_reverses_original_order(self):
        """With reverse_scorer the last original candidate becomes first ranked."""
        reranker = CrossEncoderReranker(scorer=self._reverse_scorer)
        results = reranker.rerank("test", self._CANDIDATES, top_k=5)
        # Last candidate (cE) gets highest logit (index 4) → must be first
        assert results[0].chunk.chunk_id == "cE", (
            f"Expected 'cE' to rank first, got {results[0].chunk.chunk_id!r}"
        )

    # ------------------------------------------------------------------
    # Retriever name
    # ------------------------------------------------------------------

    def test_retriever_name_reflects_reranking(self):
        """Each result's retriever must be 'rerank(bm25)'."""
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(0.0))
        results = reranker.rerank("test", self._CANDIDATES[:3], top_k=3)
        for sc in results:
            assert sc.retriever == "rerank(bm25)", (
                f"Expected retriever='rerank(bm25)', got {sc.retriever!r}"
            )

    def test_retriever_name_wraps_original_name(self):
        """Retriever name wraps whatever original retriever name was set."""
        candidates_dense = [
            _scored("d0", "text one", retriever="dense"),
            _scored("d1", "text two", retriever="dense"),
        ]
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(1.0))
        results = reranker.rerank("query", candidates_dense, top_k=2)
        for sc in results:
            assert sc.retriever == "rerank(dense)", (
                f"Expected 'rerank(dense)', got {sc.retriever!r}"
            )

    def test_unknown_retriever_name_handled(self):
        """Candidates with empty retriever name produce 'rerank(unknown)'."""
        candidates = [
            ScoredChunk(chunk=_chunk("u0", "text"), score=0.5, retriever=""),
        ]
        reranker = CrossEncoderReranker(scorer=self._constant_scorer(0.0))
        results = reranker.rerank("q", candidates, top_k=1)
        assert results[0].retriever == "rerank(unknown)"
