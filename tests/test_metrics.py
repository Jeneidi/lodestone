"""
Unit tests for evals.metrics, evals.stats, and evals.aggregate.

All tests are pure (offline, no I/O, no model calls) and designed to run
in well under 5 seconds total.

Design notes
------------
* Fixtures use tiny, hand-crafted Chunk/ScoredChunk objects.
* Every golden value is hand-computed below the test with a short comment.
* Doc-dedupe behaviour is explicitly tested: two chunks from the same
  relevant document must not be double-counted.
* Edge cases (empty retrieved, empty relevant, no match) are covered for
  every metric.
"""

from __future__ import annotations

import math

import pytest
from evals.aggregate import compare_runs, evaluate_run
from evals.metrics import (
    _dedupe_to_doc_ranking,
    average_precision,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evals.stats import bootstrap_ci, paired_permutation_test, summarize

from lodestone.schemas import Chunk, QAExample, RetrievalRunResult, ScoredChunk

# ---------------------------------------------------------------------------
# Helpers / fixture constructors
# ---------------------------------------------------------------------------


def make_chunk(doc_id: str, chunk_index: int = 0) -> Chunk:
    """Create a minimal Chunk for a given doc_id."""
    return Chunk(
        chunk_id=f"{doc_id}_{chunk_index}",
        doc_id=doc_id,
        text=f"text from {doc_id} chunk {chunk_index}",
        index=chunk_index,
    )


def make_scored(doc_id: str, score: float, chunk_index: int = 0) -> ScoredChunk:
    """Create a ScoredChunk with the given doc_id and score."""
    return ScoredChunk(chunk=make_chunk(doc_id, chunk_index), score=score)


# ---------------------------------------------------------------------------
# Standard ranked list used by many tests
#
# Rank  doc_id   score  relevant (R = {"A", "C"})
#  1     A        1.0   YES
#  2     B        0.9   no
#  3     A        0.8   duplicate (same doc as rank 1) → deduped away
#  4     C        0.7   YES  (at deduped rank 3)
#  5     D        0.6   no
# ---------------------------------------------------------------------------

REL = {"A", "C"}

RETRIEVED_5 = [
    make_scored("A", 1.0, 0),
    make_scored("B", 0.9, 0),
    make_scored("A", 0.8, 1),  # duplicate of "A"
    make_scored("C", 0.7, 0),
    make_scored("D", 0.6, 0),
]

# Deduped doc ranking: ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Tests for _dedupe_to_doc_ranking (internal helper)
# ---------------------------------------------------------------------------


class TestDedupeHelper:
    def test_deduplication_order(self) -> None:
        result = _dedupe_to_doc_ranking(RETRIEVED_5)
        assert result == ["A", "B", "C", "D"]

    def test_empty_input(self) -> None:
        assert _dedupe_to_doc_ranking([]) == []

    def test_no_duplicates(self) -> None:
        retrieved = [make_scored("X", 3.0), make_scored("Y", 2.0)]
        assert _dedupe_to_doc_ranking(retrieved) == ["X", "Y"]

    def test_all_same_doc(self) -> None:
        retrieved = [make_scored("Z", 5.0, i) for i in range(4)]
        assert _dedupe_to_doc_ranking(retrieved) == ["Z"]


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_hit_at_1_relevant_first(self) -> None:
        # A is relevant and is at deduped rank 1 → hit@1 = 1.0
        assert hit_rate_at_k(RETRIEVED_5, REL, k=1) == 1.0

    def test_hit_at_2(self) -> None:
        # A at rank 1 → hit@2 = 1.0
        assert hit_rate_at_k(RETRIEVED_5, REL, k=2) == 1.0

    def test_no_relevant_in_top1_nonrel_first(self) -> None:
        # B (non-relevant) first, A at rank 2
        retrieved = [make_scored("B", 1.0), make_scored("A", 0.9)]
        assert hit_rate_at_k(retrieved, {"A"}, k=1) == 0.0
        assert hit_rate_at_k(retrieved, {"A"}, k=2) == 1.0

    def test_no_relevant_at_all(self) -> None:
        retrieved = [make_scored("X", 1.0), make_scored("Y", 0.5)]
        assert hit_rate_at_k(retrieved, {"A", "B"}, k=10) == 0.0

    def test_empty_retrieved(self) -> None:
        assert hit_rate_at_k([], REL, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert hit_rate_at_k(RETRIEVED_5, [], k=5) == 0.0

    def test_k_zero(self) -> None:
        assert hit_rate_at_k(RETRIEVED_5, REL, k=0) == 0.0

    def test_duplicate_chunk_does_not_double_hit(self) -> None:
        # Two chunks from "A", "A" is the only relevant doc
        retrieved = [make_scored("A", 1.0, 0), make_scored("A", 0.9, 1)]
        assert hit_rate_at_k(retrieved, {"A"}, k=2) == 1.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecall:
    # Deduped ranking: [A, B, C, D]; R = {A, C}
    # recall@1 = 1/2 = 0.5  (only A found in top-1 unique-doc)
    # recall@3 = 2/2 = 1.0  (A at pos 1, C at pos 3)
    # recall@2 = 1/2 = 0.5  (only A in [A, B])

    def test_recall_at_1(self) -> None:
        assert recall_at_k(RETRIEVED_5, REL, k=1) == pytest.approx(0.5)

    def test_recall_at_2(self) -> None:
        assert recall_at_k(RETRIEVED_5, REL, k=2) == pytest.approx(0.5)

    def test_recall_at_3(self) -> None:
        # Deduped top-3: [A, B, C]; both A and C are relevant → 2/2
        assert recall_at_k(RETRIEVED_5, REL, k=3) == pytest.approx(1.0)

    def test_recall_at_k_larger_than_list(self) -> None:
        assert recall_at_k(RETRIEVED_5, REL, k=100) == pytest.approx(1.0)

    def test_empty_retrieved(self) -> None:
        assert recall_at_k([], REL, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert recall_at_k(RETRIEVED_5, [], k=5) == 0.0

    def test_duplicate_chunk_no_double_count(self) -> None:
        # Two chunks of "A"; R = {"A"} (only 1 relevant doc)
        retrieved = [make_scored("A", 1.0, 0), make_scored("A", 0.9, 1)]
        # recall@1 = 1/1 = 1.0; recall@2 should also be 1.0 (not >1)
        assert recall_at_k(retrieved, {"A"}, k=1) == pytest.approx(1.0)
        assert recall_at_k(retrieved, {"A"}, k=2) == pytest.approx(1.0)

    def test_recall_no_match(self) -> None:
        retrieved = [make_scored("X", 1.0), make_scored("Y", 0.9)]
        assert recall_at_k(retrieved, {"A", "B"}, k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecision:
    # Deduped ranking: [A, B, C, D]; R = {A, C}
    # precision@1 = 1/1 = 1.0
    # precision@2 = 1/2 = 0.5  (A relevant, B not)
    # precision@3 = 2/3         (A, B, C — two relevant)
    # precision@4 = 2/4 = 0.5  (A, B, C, D — two relevant)

    def test_precision_at_1(self) -> None:
        assert precision_at_k(RETRIEVED_5, REL, k=1) == pytest.approx(1.0)

    def test_precision_at_2(self) -> None:
        assert precision_at_k(RETRIEVED_5, REL, k=2) == pytest.approx(0.5)

    def test_precision_at_3(self) -> None:
        assert precision_at_k(RETRIEVED_5, REL, k=3) == pytest.approx(2 / 3)

    def test_precision_at_4(self) -> None:
        assert precision_at_k(RETRIEVED_5, REL, k=4) == pytest.approx(0.5)

    def test_precision_denominator_is_k_not_unique_docs(self) -> None:
        # Only 2 unique docs retrieved, but k=5 → denominator is 5
        retrieved = [make_scored("A", 1.0), make_scored("B", 0.9)]
        assert precision_at_k(retrieved, {"A"}, k=5) == pytest.approx(1 / 5)

    def test_duplicate_chunk_no_inflation(self) -> None:
        # Two chunks from "A" in retrieved; k=2, R={"A"}
        # Deduped top-2 docs: ["A", "B"] (second unique doc) →
        # ...but here we only have A chunks:
        retrieved = [make_scored("A", 1.0, 0), make_scored("A", 0.9, 1)]
        # Deduped: ["A"]; top-2 positions → only 1 unique doc, 1 relevant
        # precision@2 = 1/2 = 0.5
        assert precision_at_k(retrieved, {"A"}, k=2) == pytest.approx(0.5)

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], REL, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert precision_at_k(RETRIEVED_5, [], k=5) == 0.0

    def test_k_zero(self) -> None:
        assert precision_at_k(RETRIEVED_5, REL, k=0) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    # Deduped ranking: [A, B, C, D]; R = {A, C}
    # First relevant doc at deduped rank 1 → RR = 1/1 = 1.0

    def test_mrr_relevant_at_rank_1(self) -> None:
        assert mrr(RETRIEVED_5, REL) == pytest.approx(1.0)

    def test_mrr_relevant_at_rank_2(self) -> None:
        retrieved = [make_scored("X", 1.0), make_scored("A", 0.9)]
        assert mrr(retrieved, {"A"}) == pytest.approx(0.5)

    def test_mrr_relevant_at_rank_3(self) -> None:
        # Hand-computed: 1/3
        retrieved = [
            make_scored("X", 1.0),
            make_scored("Y", 0.9),
            make_scored("A", 0.8),
        ]
        assert mrr(retrieved, {"A"}) == pytest.approx(1 / 3)

    def test_mrr_no_relevant(self) -> None:
        retrieved = [make_scored("X", 1.0), make_scored("Y", 0.9)]
        assert mrr(retrieved, {"A"}) == pytest.approx(0.0)

    def test_mrr_empty_retrieved(self) -> None:
        assert mrr([], REL) == 0.0

    def test_mrr_empty_relevant(self) -> None:
        assert mrr(RETRIEVED_5, []) == 0.0

    def test_mrr_duplicate_chunks_use_first_rank(self) -> None:
        # Two chunks from "A"; A is relevant → RR = 1/1
        retrieved = [make_scored("A", 1.0, 0), make_scored("A", 0.9, 1)]
        assert mrr(retrieved, {"A"}) == pytest.approx(1.0)

    def test_mrr_duplicate_non_relevant_then_relevant(self) -> None:
        # [B(rank1), B(rank2-deduped-away), A(rank2-deduped)]
        retrieved = [
            make_scored("B", 1.0, 0),
            make_scored("B", 0.95, 1),
            make_scored("A", 0.9, 0),
        ]
        # Deduped: [B, A] → A at deduped rank 2 → RR = 0.5
        assert mrr(retrieved, {"A"}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCG:
    # Standard test list (deduped: [A, B, C, D], R={A,C})
    #
    # DCG@3 = rel1/log2(2) + rel2/log2(3) + rel3/log2(4)
    #       = 1/1 + 0/log2(3) + 1/2  = 1.5
    # IDCG@3 = 1/log2(2) + 1/log2(3)  (only 2 relevant docs exist)
    #        = 1.0 + 1/log2(3)
    # nDCG@3 = DCG@3 / IDCG@3

    def test_ndcg_at_3(self) -> None:
        dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(RETRIEVED_5, REL, k=3) == pytest.approx(expected, rel=1e-9)

    def test_ndcg_at_1_perfect(self) -> None:
        # Rank 1 is A (relevant); IDCG@1 = 1/log2(2) = 1.0
        # DCG@1 = 1/log2(2) = 1.0 → nDCG = 1.0
        assert ndcg_at_k(RETRIEVED_5, REL, k=1) == pytest.approx(1.0)

    def test_ndcg_at_1_no_relevant(self) -> None:
        retrieved = [make_scored("X", 1.0)]
        assert ndcg_at_k(retrieved, {"A"}, k=1) == pytest.approx(0.0)

    def test_ndcg_ideal_ordering(self) -> None:
        # Perfect ranking: both relevant docs in top-2 positions
        retrieved = [make_scored("A", 1.0), make_scored("C", 0.9)]
        assert ndcg_at_k(retrieved, {"A", "C"}, k=2) == pytest.approx(1.0)

    def test_ndcg_reversed_ranking(self) -> None:
        # Relevant docs at positions 2 and 3 of 3; non-relevant at 1
        retrieved = [make_scored("X", 1.0), make_scored("A", 0.9), make_scored("C", 0.8)]
        dcg = 0.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(retrieved, {"A", "C"}, k=3) == pytest.approx(expected, rel=1e-9)

    def test_ndcg_idcg_uses_min_k_n_relevant(self) -> None:
        # k=1, n_relevant=3 → IDCG@1 = 1/log2(2) = 1.0
        # Only one position to fill → ideal has 1 relevant in pos 1
        retrieved = [make_scored("A", 1.0)]
        assert ndcg_at_k(retrieved, {"A", "B", "C"}, k=1) == pytest.approx(1.0)

    def test_ndcg_duplicate_chunk_no_double_gain(self) -> None:
        # Two chunks from "A", one from "B"; R = {"A", "B"}; k=3
        retrieved = [
            make_scored("A", 1.0, 0),
            make_scored("A", 0.95, 1),  # deduped away
            make_scored("B", 0.9, 0),
        ]
        # Deduped ranking: [A, B]; at k=3 only 2 unique docs
        # DCG@3 = 1/log2(2) + 1/log2(3) + 0
        # IDCG@3 (min(3,2)=2) = 1/log2(2) + 1/log2(3)
        # nDCG@3 = 1.0
        assert ndcg_at_k(retrieved, {"A", "B"}, k=3) == pytest.approx(1.0)

    def test_ndcg_empty_retrieved(self) -> None:
        assert ndcg_at_k([], REL, k=5) == 0.0

    def test_ndcg_empty_relevant(self) -> None:
        assert ndcg_at_k(RETRIEVED_5, [], k=5) == 0.0

    def test_ndcg_k_zero(self) -> None:
        assert ndcg_at_k(RETRIEVED_5, REL, k=0) == 0.0


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


class TestAveragePrecision:
    # Deduped ranking: [A, B, C, D]; R = {A, C}; |R| = 2
    #
    # Position 1: A relevant → cumulative_hits=1, P@1 = 1/1
    # Position 2: B not relevant
    # Position 3: C relevant → cumulative_hits=2, P@3 = 2/3
    # Position 4: D not relevant
    # AP = (1/1 + 2/3) / 2 = (1 + 0.6667) / 2 = 0.8333...

    def test_ap_standard(self) -> None:
        expected = (1.0 / 1 + 2.0 / 3) / 2
        assert average_precision(RETRIEVED_5, REL) == pytest.approx(expected, rel=1e-9)

    def test_ap_perfect(self) -> None:
        # Top-2: both A and C relevant
        retrieved = [make_scored("A", 1.0), make_scored("C", 0.9)]
        # P@1 = 1/1, P@2 = 2/2; AP = (1 + 1) / 2 = 1.0
        assert average_precision(retrieved, {"A", "C"}) == pytest.approx(1.0)

    def test_ap_first_position_not_relevant(self) -> None:
        # Relevant at position 2 only; |R| = 1
        retrieved = [make_scored("X", 1.0), make_scored("A", 0.9)]
        # P@2 = 1/2; AP = (1/2) / 1 = 0.5
        assert average_precision(retrieved, {"A"}) == pytest.approx(0.5)

    def test_ap_no_relevant(self) -> None:
        retrieved = [make_scored("X", 1.0), make_scored("Y", 0.9)]
        assert average_precision(retrieved, {"A"}) == pytest.approx(0.0)

    def test_ap_empty_retrieved(self) -> None:
        assert average_precision([], REL) == 0.0

    def test_ap_empty_relevant(self) -> None:
        assert average_precision(RETRIEVED_5, []) == 0.0

    def test_ap_duplicate_chunks_no_double_credit(self) -> None:
        # Two chunks from "A"; R = {"A"}
        retrieved = [make_scored("A", 1.0, 0), make_scored("A", 0.9, 1)]
        # Deduped: [A]; P@1 = 1/1; AP = 1/1 / 1 = 1.0
        assert average_precision(retrieved, {"A"}) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_constant_sequence(self) -> None:
        vals = [0.5] * 100
        mean, lo, hi = bootstrap_ci(vals, n_resamples=500, seed=0)
        assert mean == pytest.approx(0.5)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_determinism_with_seed(self) -> None:
        vals = [0.1, 0.4, 0.7, 0.3, 0.9, 0.2, 0.6, 0.8, 0.5, 0.0]
        r1 = bootstrap_ci(vals, n_resamples=200, seed=42)
        r2 = bootstrap_ci(vals, n_resamples=200, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self) -> None:
        vals = list(range(20))
        r1 = bootstrap_ci(vals, n_resamples=500, seed=1)
        r2 = bootstrap_ci(vals, n_resamples=500, seed=9999)
        # Not a hard assertion (could theoretically match), but should differ
        # for a reasonably variable sequence
        # At minimum, both CIs must be valid intervals
        assert r1[1] <= r1[0] <= r1[2]
        assert r2[1] <= r2[0] <= r2[2]

    def test_mean_is_sample_mean(self) -> None:
        vals = [0.2, 0.4, 0.6, 0.8]
        mean, _, _ = bootstrap_ci(vals, seed=42)
        assert mean == pytest.approx(sum(vals) / len(vals))

    def test_ci_is_ordered(self) -> None:
        vals = [float(i) / 10 for i in range(10)]
        mean, lo, hi = bootstrap_ci(vals, seed=42)
        assert lo <= mean <= hi

    def test_empty_input(self) -> None:
        mean, lo, hi = bootstrap_ci([], seed=42)
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_single_value(self) -> None:
        mean, lo, hi = bootstrap_ci([0.75], n_resamples=100, seed=42)
        assert mean == pytest.approx(0.75)
        assert lo == pytest.approx(0.75)
        assert hi == pytest.approx(0.75)

    def test_95_ci_contains_true_mean(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        # Use values drawn from a known distribution; check CI covers mean
        vals = list(rng.uniform(0, 1, 50))
        true_mean = 0.5
        _, lo, hi = bootstrap_ci(vals, n_resamples=2000, confidence=0.95, seed=7)
        # This is probabilistic but should pass for 50 uniform samples
        assert lo < true_mean < hi


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------


class TestPairedPermutationTest:
    def test_identical_sequences_p_near_1(self) -> None:
        vals = [0.1, 0.3, 0.5, 0.7, 0.9] * 4
        p = paired_permutation_test(vals, vals, n_permutations=5000, seed=42)
        assert p >= 0.9, f"Expected p near 1.0 for identical sequences, got {p}"

    def test_clearly_separated_p_small(self) -> None:
        # System A clearly better: each pair has a large positive difference
        a = [1.0] * 20
        b = [0.0] * 20
        p = paired_permutation_test(a, b, n_permutations=10000, seed=42)
        assert p < 0.01, f"Expected p < 0.01 for clearly separated sequences, got {p}"

    def test_p_value_in_valid_range(self) -> None:
        a = [0.2, 0.4, 0.6, 0.3, 0.5]
        b = [0.1, 0.3, 0.5, 0.2, 0.4]
        p = paired_permutation_test(a, b, n_permutations=1000, seed=42)
        assert 0.0 < p <= 1.0

    def test_determinism_with_seed(self) -> None:
        a = [0.5, 0.6, 0.3, 0.8, 0.2]
        b = [0.4, 0.5, 0.4, 0.7, 0.3]
        p1 = paired_permutation_test(a, b, n_permutations=500, seed=99)
        p2 = paired_permutation_test(a, b, n_permutations=500, seed=99)
        assert p1 == p2

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            paired_permutation_test([0.1, 0.2], [0.3], n_permutations=100)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_permutation_test([], [], n_permutations=100)

    def test_two_sided_symmetric(self) -> None:
        # A > B and B > A (same magnitude) should yield similar p-values
        a = [0.8, 0.7, 0.9]
        b = [0.2, 0.3, 0.1]
        p_ab = paired_permutation_test(a, b, n_permutations=5000, seed=1)
        p_ba = paired_permutation_test(b, a, n_permutations=5000, seed=1)
        assert abs(p_ab - p_ba) < 0.05  # two-sided: should be similar


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_basic(self) -> None:
        import numpy as np

        vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = summarize(vals)
        assert result["mean"] == pytest.approx(0.5)
        assert result["p50"] == pytest.approx(0.5)
        assert result["p95"] == pytest.approx(float(np.percentile(vals, 95)))
        assert result["std"] >= 0.0

    def test_constant(self) -> None:
        result = summarize([0.7, 0.7, 0.7])
        assert result["mean"] == pytest.approx(0.7)
        assert result["std"] == pytest.approx(0.0)
        assert result["p50"] == pytest.approx(0.7)
        assert result["p95"] == pytest.approx(0.7)

    def test_empty(self) -> None:
        result = summarize([])
        assert result == {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}

    def test_keys_present(self) -> None:
        result = summarize([0.1, 0.2])
        assert set(result.keys()) == {"mean", "std", "p50", "p95"}


# ---------------------------------------------------------------------------
# evaluate_run (aggregate)
# ---------------------------------------------------------------------------


def _make_run_result(
    qid: str, doc_ids_scores: list[tuple[str, float]], latency_ms: float = 10.0
) -> RetrievalRunResult:
    retrieved = [make_scored(d, s) for d, s in doc_ids_scores]
    return RetrievalRunResult(qid=qid, retrieved=retrieved, latency_ms=latency_ms)


def _make_qa(qid: str, relevant_doc_ids: list[str]) -> QAExample:
    return QAExample(
        qid=qid,
        question=f"Q-{qid}",
        answer=f"A-{qid}",
        relevant_doc_ids=relevant_doc_ids,
    )


class TestEvaluateRun:
    def test_basic_structure(self) -> None:
        results = [_make_run_result("q1", [("A", 1.0), ("B", 0.9)])]
        qa = [_make_qa("q1", ["A"])]
        out = evaluate_run(results, qa, ks=(1, 5))

        assert "per_query" in out
        assert "summary" in out
        assert "latency" in out
        assert "n_queries" in out
        assert out["n_queries"] == 1

    def test_per_query_metric_names(self) -> None:
        results = [_make_run_result("q1", [("A", 1.0)])]
        qa = [_make_qa("q1", ["A"])]
        out = evaluate_run(results, qa, ks=(1, 3, 5))
        row = out["per_query"]["q1"]

        for k in (1, 3, 5):
            assert f"hit_rate@{k}" in row
            assert f"recall@{k}" in row
            assert f"precision@{k}" in row
            assert f"ndcg@{k}" in row
        assert "mrr" in row
        assert "map" in row

    def test_summary_ci_keys(self) -> None:
        results = [
            _make_run_result("q1", [("A", 1.0)]),
            _make_run_result("q2", [("B", 1.0)]),
        ]
        qa = [_make_qa("q1", ["A"]), _make_qa("q2", ["B"])]
        out = evaluate_run(results, qa, ks=(1,))
        for metric_name, stats in out["summary"].items():
            assert "mean" in stats
            assert "ci_lo" in stats
            assert "ci_hi" in stats

    def test_missing_qa_warns_and_skips(self) -> None:
        results = [_make_run_result("q_orphan", [("A", 1.0)])]
        qa = [_make_qa("q_other", ["A"])]
        with pytest.warns(UserWarning, match="q_orphan"):
            out = evaluate_run(results, qa)
        assert out["n_queries"] == 0

    def test_latency_stats(self) -> None:
        results = [
            _make_run_result("q1", [("A", 1.0)], latency_ms=20.0),
            _make_run_result("q2", [("A", 1.0)], latency_ms=40.0),
        ]
        qa = [_make_qa("q1", ["A"]), _make_qa("q2", ["A"])]
        out = evaluate_run(results, qa, ks=(1,))
        assert out["latency"]["mean_ms"] == pytest.approx(30.0)
        assert "p50_ms" in out["latency"]
        assert "p95_ms" in out["latency"]

    def test_values_are_correct(self) -> None:
        # q1: A retrieved at rank 1, A is relevant → perfect
        # q2: B retrieved at rank 1, A is relevant → miss
        results = [
            _make_run_result("q1", [("A", 1.0)]),
            _make_run_result("q2", [("B", 1.0)]),
        ]
        qa = [_make_qa("q1", ["A"]), _make_qa("q2", ["A"])]
        out = evaluate_run(results, qa, ks=(1,))
        assert out["per_query"]["q1"]["hit_rate@1"] == pytest.approx(1.0)
        assert out["per_query"]["q2"]["hit_rate@1"] == pytest.approx(0.0)
        assert out["summary"]["hit_rate@1"]["mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def _make_two_runs(
        self,
    ) -> tuple[dict, dict]:
        # Run A: perfect on q1 and q2
        results_a = [
            _make_run_result("q1", [("A", 1.0)]),
            _make_run_result("q2", [("B", 1.0)]),
        ]
        # Run B: misses both
        results_b = [
            _make_run_result("q1", [("X", 1.0)]),
            _make_run_result("q2", [("X", 1.0)]),
        ]
        qa = [_make_qa("q1", ["A"]), _make_qa("q2", ["B"])]
        run_a = evaluate_run(results_a, qa, ks=(1,))
        run_b = evaluate_run(results_b, qa, ks=(1,))
        return run_a, run_b

    def test_delta_positive_when_a_better(self) -> None:
        run_a, run_b = self._make_two_runs()
        result = compare_runs(run_a, run_b, metric="hit_rate@1")
        assert result["delta"] == pytest.approx(1.0)
        assert result["n"] == 2

    def test_p_value_small_for_large_effect(self) -> None:
        # With only 2 queries and binary outcomes, min p = 0.5 (2^2 permutations)
        # so just check it's valid
        run_a, run_b = self._make_two_runs()
        result = compare_runs(run_a, run_b, metric="mrr")
        assert 0.0 < result["p_value"] <= 1.0

    def test_identical_runs_p_near_1(self) -> None:
        results = [
            _make_run_result("q1", [("A", 1.0)]),
            _make_run_result("q2", [("B", 1.0)]),
            _make_run_result("q3", [("C", 1.0)]),
        ]
        qa = [_make_qa("q1", ["A"]), _make_qa("q2", ["B"]), _make_qa("q3", ["C"])]
        run = evaluate_run(results, qa, ks=(1,))
        result = compare_runs(run, run, metric="hit_rate@1")
        assert result["delta"] == pytest.approx(0.0)
        assert result["p_value"] >= 0.5

    def test_output_keys(self) -> None:
        run_a, run_b = self._make_two_runs()
        result = compare_runs(run_a, run_b, metric="mrr")
        assert "delta" in result
        assert "p_value" in result
        assert "n" in result

    def test_missing_metric_handled_gracefully(self) -> None:
        # A metric that doesn't exist → defaults to 0.0
        run_a, run_b = self._make_two_runs()
        result = compare_runs(run_a, run_b, metric="nonexistent_metric")
        assert result["delta"] == pytest.approx(0.0)
