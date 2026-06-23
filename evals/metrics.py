"""Retrieval evaluation metrics for Lodestone.

Relevance convention
--------------------
A retrieved chunk is considered **relevant** if and only if its
``chunk.doc_id`` is present in the query's ``relevant_doc_ids`` set.
Relevance is assessed at the **document** level, not the chunk level.

Document-deduplication design decision
---------------------------------------
A single source document may be split into many chunks, several of which can
appear in the same ranked result list.  Counting every relevant chunk
independently would inflate precision and recall figures and produce
misleading comparisons between systems that return fine-grained vs.
coarse-grained chunks.

Therefore, for all precision/recall/nDCG-style metrics, the ranked list is
first collapsed to a *document-ranked list* by keeping only the **first
occurrence** of each ``doc_id`` in the score-ordered ``retrieved`` list.
"First occurrence" corresponds to the highest-scoring chunk from that
document, which is the most informative signal for whether the system
retrieved the document.  Hit-rate and MRR apply the same deduplicated view.

Edge-case contract
------------------
* ``retrieved`` is empty **or** ``relevant_doc_ids`` is empty → return 0.0
  for every metric.
* ``k`` is 0 → return 0.0 (no positions to evaluate).
* Division by zero is always guarded; the function never raises on valid
  (but degenerate) inputs.

All functions are **pure** (no I/O, no side-effects, no global state).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from lodestone.schemas import ScoredChunk

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _dedupe_to_doc_ranking(retrieved: Sequence[ScoredChunk]) -> list[str]:
    """Return an ordered list of unique doc_ids, first-occurrence wins.

    The input is assumed to be already rank-ordered (highest score first).
    Each doc_id appears at most once in the output, positioned at the rank
    of its first (highest-scoring) chunk.

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.

    Returns
    -------
    list[str]
        Ordered unique doc_ids.

    """
    seen: set[str] = set()
    doc_ranking: list[str] = []
    for sc in retrieved:
        did = sc.chunk.doc_id
        if did not in seen:
            seen.add(did)
            doc_ranking.append(did)
    return doc_ranking


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def hit_rate_at_k(
    retrieved: Sequence[ScoredChunk],
    relevant_doc_ids: list[str] | set[str],
    k: int,
) -> float:
    """Compute Hit Rate @ k (also called Recall-Any @ k or Success @ k).

    Formula
    -------
    .. math::

        \\text{HR@k} = \\mathbf{1}\\left[
            \\exists\\, i \\le k : \\text{doc\\_ranking}[i] \\in R
        \\right]

    where ``doc_ranking`` is the deduplicated document ranking and ``R`` is
    the set of relevant doc_ids.

    Returns 1.0 if at least one relevant document appears in the top-k
    unique-document positions, 0.0 otherwise.

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.
    relevant_doc_ids:
        Ground-truth set of relevant document identifiers.
    k:
        Cut-off depth.  Only the first ``k`` unique-document positions
        are considered.

    Returns
    -------
    float
        1.0 or 0.0.

    Edge cases
    ----------
    * Empty ``retrieved`` or empty ``relevant_doc_ids`` → 0.0.
    * ``k`` <= 0 → 0.0.

    """
    if not retrieved or not relevant_doc_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_doc_ids)
    doc_ranking = _dedupe_to_doc_ranking(retrieved)
    for doc_id in doc_ranking[:k]:
        if doc_id in rel_set:
            return 1.0
    return 0.0


def recall_at_k(
    retrieved: Sequence[ScoredChunk],
    relevant_doc_ids: list[str] | set[str],
    k: int,
) -> float:
    """Compute Recall @ k.

    Formula
    -------
    .. math::

        \\text{R@k} = \\frac{
            |\\{d \\in \\text{doc\\_ranking}[:k] : d \\in R\\}|
        }{|R|}

    where ``doc_ranking`` is the deduplicated document ranking and ``R`` is
    the set of relevant doc_ids.

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.
    relevant_doc_ids:
        Ground-truth set of relevant document identifiers.
    k:
        Cut-off depth.

    Returns
    -------
    float
        Fraction of relevant documents found in the top-k unique-document
        positions, in [0.0, 1.0].

    Edge cases
    ----------
    * Empty ``retrieved`` or empty ``relevant_doc_ids`` → 0.0.
    * ``k`` <= 0 → 0.0.

    """
    if not retrieved or not relevant_doc_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_doc_ids)
    doc_ranking = _dedupe_to_doc_ranking(retrieved)
    hits = sum(1 for d in doc_ranking[:k] if d in rel_set)
    return hits / len(rel_set)


def precision_at_k(
    retrieved: Sequence[ScoredChunk],
    relevant_doc_ids: list[str] | set[str],
    k: int,
) -> float:
    """Compute Precision @ k (document-deduped).

    Definition
    ----------
    1. Collapse ``retrieved`` to a deduplicated document ranking (first
       occurrence of each doc_id, score-ordered).
    2. Truncate to the first ``k`` document positions.
    3. Precision = (number of relevant documents in those k positions) / k.

    Note: the denominator is always ``k`` (not the number of unique docs
    returned), following the standard IR convention.  If fewer than ``k``
    unique documents are retrieved, those positions count as non-relevant
    for the purposes of this metric.

    Formula
    -------
    .. math::

        \\text{P@k} = \\frac{
            |\\{d \\in \\text{doc\\_ranking}[:k] : d \\in R\\}|
        }{k}

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.
    relevant_doc_ids:
        Ground-truth set of relevant document identifiers.
    k:
        Cut-off depth.

    Returns
    -------
    float
        Fraction of k positions that are relevant, in [0.0, 1.0].

    Edge cases
    ----------
    * Empty ``retrieved`` or empty ``relevant_doc_ids`` → 0.0.
    * ``k`` <= 0 → 0.0.

    """
    if not retrieved or not relevant_doc_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_doc_ids)
    doc_ranking = _dedupe_to_doc_ranking(retrieved)
    hits = sum(1 for d in doc_ranking[:k] if d in rel_set)
    return hits / k


def mrr(
    retrieved: Sequence[ScoredChunk],
    relevant_doc_ids: list[str] | set[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    In multi-query evaluation, average this value across queries to obtain
    the standard MRR metric.

    Formula
    -------
    .. math::

        \\text{RR} = \\frac{1}{\\text{rank of first relevant document}}

    where ranks are 1-indexed and computed over the deduplicated document
    ranking.  If no relevant document is found, RR = 0.

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.
    relevant_doc_ids:
        Ground-truth set of relevant document identifiers.

    Returns
    -------
    float
        Reciprocal rank in [0.0, 1.0].

    Edge cases
    ----------
    * Empty ``retrieved`` or empty ``relevant_doc_ids`` → 0.0.
    * No relevant document found → 0.0.

    """
    if not retrieved or not relevant_doc_ids:
        return 0.0
    rel_set = set(relevant_doc_ids)
    doc_ranking = _dedupe_to_doc_ranking(retrieved)
    for rank, doc_id in enumerate(doc_ranking, start=1):
        if doc_id in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[ScoredChunk],
    relevant_doc_ids: list[str] | set[str],
    k: int,
) -> float:
    """Compute Normalised Discounted Cumulative Gain @ k (binary relevance).

    Formula
    -------
    .. math::

        \\text{DCG@k} = \\sum_{i=1}^{k}
            \\frac{\\text{rel}_i}{\\log_2(i + 1)}

    where :math:`\\text{rel}_i \\in \\{0, 1\\}` is the binary relevance of
    the document at position *i* in the deduplicated ranking.

    The ideal DCG (IDCG) is computed assuming the top
    :math:`\\min(k, |R|)` positions are all relevant:

    .. math::

        \\text{IDCG@k} = \\sum_{i=1}^{\\min(k,|R|)}
            \\frac{1}{\\log_2(i + 1)}

    .. math::

        \\text{nDCG@k} = \\frac{\\text{DCG@k}}{\\text{IDCG@k}}

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.
    relevant_doc_ids:
        Ground-truth set of relevant document identifiers.
    k:
        Cut-off depth.

    Returns
    -------
    float
        nDCG@k in [0.0, 1.0].

    Edge cases
    ----------
    * Empty ``retrieved`` or empty ``relevant_doc_ids`` → 0.0.
    * ``k`` <= 0 → 0.0.
    * IDCG == 0 → 0.0 (guarded division by zero, should not occur given
      the guard on ``relevant_doc_ids``).

    """
    if not retrieved or not relevant_doc_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_doc_ids)
    doc_ranking = _dedupe_to_doc_ranking(retrieved)

    # Actual DCG
    dcg = 0.0
    for i, doc_id in enumerate(doc_ranking[:k], start=1):
        if doc_id in rel_set:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: best possible — top min(k, |R|) positions all relevant
    n_ideal = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision(
    retrieved: Sequence[ScoredChunk],
    relevant_doc_ids: list[str] | set[str],
) -> float:
    """Compute Average Precision (AP) for a single query.

    In multi-query evaluation, average this value across queries to obtain
    the standard MAP (Mean Average Precision) metric.

    Formula
    -------
    .. math::

        \\text{AP} = \\frac{1}{|R|}
        \\sum_{i=1}^{N} P@i \\cdot \\text{rel}_i

    where:

    * :math:`|R|` is the total number of relevant documents.
    * :math:`N` is the length of the deduplicated document ranking.
    * :math:`P@i` is precision at position *i* in the deduped ranking.
    * :math:`\\text{rel}_i \\in \\{0, 1\\}` is the binary relevance of
      the document at position *i*.

    The sum accumulates a precision term only at positions where a new
    relevant document is found (``rel_i == 1``).

    Parameters
    ----------
    retrieved:
        Score-ordered list of ScoredChunk objects.
    relevant_doc_ids:
        Ground-truth set of relevant document identifiers.

    Returns
    -------
    float
        Average Precision in [0.0, 1.0].

    Edge cases
    ----------
    * Empty ``retrieved`` or empty ``relevant_doc_ids`` → 0.0.
    * No relevant document found → 0.0.

    """
    if not retrieved or not relevant_doc_ids:
        return 0.0
    rel_set = set(relevant_doc_ids)
    doc_ranking = _dedupe_to_doc_ranking(retrieved)

    n_relevant = len(rel_set)
    cumulative_hits = 0
    ap_sum = 0.0
    for i, doc_id in enumerate(doc_ranking, start=1):
        if doc_id in rel_set:
            cumulative_hits += 1
            ap_sum += cumulative_hits / i  # precision at position i

    if n_relevant == 0:
        return 0.0
    return ap_sum / n_relevant
