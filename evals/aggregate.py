"""Aggregate retrieval evaluation: joins RetrievalRunResult × QAExample,
computes per-query metrics, bootstrap-CI summaries, and latency statistics.

Usage
-----
.. code-block:: python

    from evals.aggregate import evaluate_run, compare_runs

    summary = evaluate_run(run_results, qa_examples, ks=(1, 3, 5, 10))
    comparison = compare_runs(summary_a, summary_b, metric="recall@5")
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from evals.metrics import (
    average_precision,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evals.stats import bootstrap_ci, summarize
from lodestone.schemas import QAExample, RetrievalRunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_run(
    results: list[RetrievalRunResult],
    qa: list[QAExample],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate a retrieval run against ground-truth QA examples.

    Joins ``results`` and ``qa`` by ``qid``.  Queries present in ``results``
    but missing from ``qa`` emit a warning and are skipped.  Queries in
    ``qa`` but not in ``results`` are silently ignored (the run may be
    partial).

    Metric names
    ------------
    For each k in ``ks``:

    * ``"hit_rate@{k}"``  — Hit Rate @ k
    * ``"recall@{k}"``    — Recall @ k
    * ``"precision@{k}"`` — Precision @ k
    * ``"ndcg@{k}"``      — nDCG @ k

    Plus (independent of k):

    * ``"mrr"``  — Mean Reciprocal Rank (single-query reciprocal rank)
    * ``"map"``  — Mean Average Precision (single-query AP)

    Parameters
    ----------
    results:
        List of RetrievalRunResult objects, one per query.
    qa:
        List of QAExample ground-truth objects.
    ks:
        Tuple of cut-off depths for rank-cut metrics.

    Returns
    -------
    dict with keys:

    * ``"per_query"``  — ``{qid: {metric_name: float}}``
    * ``"summary"``    — ``{metric_name: {"mean": float, "ci_lo": float, "ci_hi": float}}``
    * ``"latency"``    — ``{"p50_ms": float, "p95_ms": float, "mean_ms": float}``
    * ``"n_queries"``  — int, number of evaluated queries

    """
    # Build qid → QAExample lookup
    qa_map: dict[str, QAExample] = {ex.qid: ex for ex in qa}

    per_query: dict[str, dict[str, float]] = {}
    latencies: list[float] = []

    for run_result in results:
        qid = run_result.qid
        if qid not in qa_map:
            warnings.warn(
                f"evaluate_run: qid '{qid}' not found in qa examples — skipping.",
                stacklevel=2,
            )
            continue

        example = qa_map[qid]
        rel_ids = example.relevant_doc_ids
        retrieved = run_result.retrieved
        latencies.append(run_result.latency_ms)

        row: dict[str, float] = {}

        for k in ks:
            row[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, rel_ids, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, rel_ids, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, rel_ids, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved, rel_ids, k)

        row["mrr"] = mrr(retrieved, rel_ids)
        row["map"] = average_precision(retrieved, rel_ids)

        per_query[qid] = row

    # Build summary with bootstrap CIs for every metric
    summary: dict[str, dict[str, float]] = {}
    if per_query:
        all_metric_names = list(next(iter(per_query.values())).keys())
        for metric_name in all_metric_names:
            vals = [per_query[qid][metric_name] for qid in per_query]
            mean_val, ci_lo, ci_hi = bootstrap_ci(vals)
            summary[metric_name] = {
                "mean": mean_val,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }

    # Latency statistics
    lat_stats = summarize(latencies)
    latency_summary = {
        "p50_ms": lat_stats["p50"],
        "p95_ms": lat_stats["p95"],
        "mean_ms": lat_stats["mean"],
    }

    return {
        "per_query": per_query,
        "summary": summary,
        "latency": latency_summary,
        "n_queries": len(per_query),
    }


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------


def compare_runs(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Compare two evaluated runs on a single metric via paired permutation test.

    Extracts per-query metric scores from the ``"per_query"`` sub-dicts of
    both run evaluation dicts (as returned by :func:`evaluate_run`).  Only
    queries present in **both** runs are used (inner join on qids).

    Parameters
    ----------
    run_a:
        Output of ``evaluate_run(...)`` for system A.
    run_b:
        Output of ``evaluate_run(...)`` for system B.
    metric:
        Metric name to compare, e.g. ``"recall@5"``, ``"ndcg@10"``,
        ``"mrr"``, ``"map"``.

    Returns
    -------
    dict with keys:

    * ``"delta"``   — mean(a) - mean(b) over shared queries.
    * ``"p_value"`` — two-sided paired permutation p-value.
    * ``"n"``       — number of shared queries used for the test.

    If fewer than 2 shared queries are found, ``"p_value"`` is set to 1.0
    and a warning is emitted.

    """
    # Import here to avoid circular issues if stats is not yet imported
    from evals.stats import paired_permutation_test

    per_q_a: dict[str, dict[str, float]] = run_a.get("per_query", {})
    per_q_b: dict[str, dict[str, float]] = run_b.get("per_query", {})

    shared_qids = sorted(set(per_q_a.keys()) & set(per_q_b.keys()))
    n = len(shared_qids)

    if n < 2:
        warnings.warn(
            f"compare_runs: only {n} shared queries for metric '{metric}'; "
            "p-value set to 1.0.",
            stacklevel=2,
        )
        if n == 0:
            return {"delta": 0.0, "p_value": 1.0, "n": 0}
        a_val = per_q_a[shared_qids[0]].get(metric, 0.0)
        b_val = per_q_b[shared_qids[0]].get(metric, 0.0)
        return {"delta": a_val - b_val, "p_value": 1.0, "n": n}

    scores_a = [per_q_a[qid].get(metric, 0.0) for qid in shared_qids]
    scores_b = [per_q_b[qid].get(metric, 0.0) for qid in shared_qids]

    delta = float(
        sum(a - b for a, b in zip(scores_a, scores_b)) / n
    )
    p_value = paired_permutation_test(scores_a, scores_b)

    return {
        "delta": delta,
        "p_value": p_value,
        "n": n,
    }
