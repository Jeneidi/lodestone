"""evals.harness — Main evaluation entry point.

Run via::

    python -m evals.harness [options]
    make eval

CLI flags
---------
--k               Number of documents to retrieve per query (default: 10).
--pipelines       Comma-separated pipeline names to run
                  (default: hybrid_rrf_fixed,hybrid_rrf_fixed_rerank,
                             bm25_fixed,dense_fixed).
--max-questions   Cap the number of QA examples (default: all).
--out             Output directory for report artefacts (default: evals/reports).

Outputs
-------
evals/reports/results.json   Full nested results (per-pipeline summaries,
                             comparisons, metadata block).
evals/reports/RESULTS.md     Clean markdown tables suitable for pasting into
                             the project README.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.aggregate import compare_runs, evaluate_run
from evals.runner import default_pipelines, run_retrieval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float, lo: float, hi: float, decimals: int = 3) -> str:
    """Format a metric value with its bootstrap CI.

    Returns a string like ``"0.812 [0.790, 0.830]"``.
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(value)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _fmt_latency(ms: float) -> str:
    """Format a latency in milliseconds."""
    return f"{ms:.1f} ms"


def _sig_marker(p: float) -> str:
    """Return a significance marker string for a p-value."""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Rich table printing
# ---------------------------------------------------------------------------


def _print_results_table(
    names: list[str],
    eval_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    reference_name: str,
) -> None:
    """Print a formatted table of metrics to stdout.

    Columns: pipeline | ndcg@5 (CI) | recall@5 (CI) | mrr (CI) | map (CI)
           | p50 latency | p95 latency

    Args:
        names:          Ordered list of pipeline names.
        eval_results:   Mapping pipeline_name -> evaluate_run output dict.
        comparisons:    Mapping pipeline_name -> compare_runs output vs reference.
        reference_name: Name of the baseline pipeline (first in list).

    """
    header_metrics = ["ndcg@5", "recall@5", "mrr", "map"]
    col_width = 26
    name_width = 30

    sep = "-" * (name_width + len(header_metrics) * (col_width + 3) + 2 * 14 + 3)

    print()
    print("  Retrieval Evaluation Results")
    print(sep)
    header = (
        f"{'Pipeline':<{name_width}}"
        + "".join(f"  {m:^{col_width}}" for m in header_metrics)
        + f"  {'p50 lat':>12}  {'p95 lat':>12}"
    )
    print(header)
    print(sep)

    for name in names:
        ev = eval_results[name]
        summary = ev.get("summary", {})
        latency = ev.get("latency", {})

        row = f"{name:<{name_width}}"
        for metric in header_metrics:
            m = summary.get(metric, {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
            cell = _fmt(m["mean"], m["ci_lo"], m["ci_hi"])
            row += f"  {cell:^{col_width}}"

        p50 = _fmt_latency(latency.get("p50_ms", 0.0))
        p95 = _fmt_latency(latency.get("p95_ms", 0.0))
        row += f"  {p50:>12}  {p95:>12}"
        print(row)

    print(sep)
    print()

    # Pairwise significance vs reference
    if comparisons:
        print(f"  Pairwise significance vs '{reference_name}' on ndcg@5:")
        print(f"  {'Pipeline':<{name_width}}  {'delta':>10}  {'p-value':>10}  {'sig':>4}")
        print(f"  {'-' * (name_width + 32)}")
        for name in names:
            if name == reference_name:
                continue
            cmp = comparisons.get(name, {})
            delta = cmp.get("delta", 0.0)
            pv = cmp.get("p_value", 1.0)
            sig = _sig_marker(pv)
            print(f"  {name:<{name_width}}  {delta:>+10.4f}  {pv:>10.4f}  {sig:>4}")
        print()
        print("  (* p<0.05  ** p<0.01)")
        print()


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------


def _write_markdown_report(
    names: list[str],
    eval_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    reference_name: str,
    out_path: Path,
) -> None:
    """Write ``RESULTS.md`` with markdown tables.

    Args:
        names:          Ordered pipeline names.
        eval_results:   Pipeline -> evaluate_run output.
        comparisons:    Pipeline -> compare_runs output vs reference.
        reference_name: Name of the baseline pipeline.
        out_path:       Destination file path.

    """
    lines: list[str] = []
    lines.append("# Lodestone Retrieval Evaluation Results\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

    # Metrics table
    header_metrics = ["ndcg@5", "recall@5", "mrr", "map"]
    header_row = "| Pipeline | " + " | ".join(header_metrics) + " | p50 lat | p95 lat |"
    sep_row = (
        "|" + "|".join(["-" * 30] + ["-" * 28] * len(header_metrics) + ["-" * 10, "-" * 10]) + "|"
    )

    lines.append("\n## Per-Pipeline Metrics (mean [95% CI])\n")
    lines.append(header_row)
    lines.append(sep_row)

    for name in names:
        ev = eval_results[name]
        summary = ev.get("summary", {})
        latency = ev.get("latency", {})
        cells = [f"`{name}`"]
        for metric in header_metrics:
            m = summary.get(metric, {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
            cells.append(_fmt(m["mean"], m["ci_lo"], m["ci_hi"]))
        cells.append(_fmt_latency(latency.get("p50_ms", 0.0)))
        cells.append(_fmt_latency(latency.get("p95_ms", 0.0)))
        lines.append("| " + " | ".join(cells) + " |")

    # Significance table
    lines.append(f"\n## Pairwise Significance vs `{reference_name}` on ndcg@5\n")
    lines.append("| Pipeline | delta | p-value | sig |")
    lines.append("|" + "-" * 30 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 5 + "|")

    for name in names:
        if name == reference_name:
            continue
        cmp = comparisons.get(name, {})
        delta = cmp.get("delta", 0.0)
        pv = cmp.get("p_value", 1.0)
        sig = _sig_marker(pv)
        lines.append(f"| `{name}` | {delta:+.4f} | {pv:.4f} | {sig} |")

    lines.append("\n_\\* p<0.05  \\*\\* p<0.01_\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown report to %s", out_path)


# ---------------------------------------------------------------------------
# JSON report writer
# ---------------------------------------------------------------------------


def _write_json_report(
    names: list[str],
    eval_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    out_path: Path,
) -> None:
    """Write ``results.json`` with full nested results.

    Args:
        names:          Pipeline names.
        eval_results:   Pipeline -> evaluate_run output.
        comparisons:    Pipeline -> compare_runs result vs reference.
        metadata:       Metadata block (timestamp, model names, corpus stats).
        out_path:       Destination file path.

    """
    payload: dict[str, Any] = {
        "metadata": metadata,
        "pipelines": {},
        "comparisons": comparisons,
    }

    for name in names:
        ev = eval_results[name]
        # Exclude per_query to keep file manageable — store summary + latency only
        payload["pipelines"][name] = {
            "summary": ev.get("summary", {}),
            "latency": ev.get("latency", {}),
            "n_queries": ev.get("n_queries", 0),
        }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote JSON report to %s", out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the evaluation harness and write reports.

    Returns:
        Exit code (0 on success, 1 on error).

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="python -m evals.harness",
        description="Lodestone retrieval evaluation harness.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="hybrid_rrf_fixed,hybrid_rrf_fixed_rerank,bm25_fixed,dense_fixed",
        help=(
            "Comma-separated list of pipeline names to run. "
            "Default: hybrid_rrf_fixed,hybrid_rrf_fixed_rerank,bm25_fixed,dense_fixed"
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        dest="max_questions",
        help="Cap the number of QA examples evaluated (default: all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evals/reports",
        help="Output directory for report artefacts (default: evals/reports).",
    )
    args = parser.parse_args()

    requested = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    try:
        from lodestone.data import load_corpus, load_qa  # noqa: PLC0415

        corpus = load_corpus()
        qa = load_qa()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nPlease run `make data` to build the dataset before running the harness.",
            file=sys.stderr,
        )
        return 1

    if args.max_questions is not None and args.max_questions > 0:
        qa = qa[: args.max_questions]
        logger.info("Capped QA set to %d questions.", len(qa))

    logger.info("Corpus: %d documents | QA: %d examples", len(corpus), len(qa))

    # ------------------------------------------------------------------
    # Resolve requested pipelines
    # ------------------------------------------------------------------
    all_specs = {spec.name: spec for spec in default_pipelines()}
    unknown = [p for p in requested if p not in all_specs]
    if unknown:
        print(
            f"\nERROR: Unknown pipeline(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(all_specs.keys())}",
            file=sys.stderr,
        )
        return 1

    selected_specs = [all_specs[name] for name in requested]

    # ------------------------------------------------------------------
    # Run pipelines
    # ------------------------------------------------------------------
    from lodestone.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    eval_results: dict[str, dict[str, Any]] = {}

    for spec in selected_specs:
        logger.info("=== Running pipeline: %s ===", spec.name)
        try:
            retriever, reranker = spec.build(corpus)
            run = run_retrieval(
                retriever=retriever,
                qa=qa,
                k=args.k,
                reranker=reranker,
                rerank_top_k=settings.rerank_top_k,
            )
            ev = evaluate_run(run, qa, ks=(1, 3, 5, 10))
            eval_results[spec.name] = ev
            n_q = ev.get("n_queries", 0)
            ndcg5 = ev["summary"].get("ndcg@5", {}).get("mean", 0.0)
            logger.info("Pipeline %s: n_queries=%d  ndcg@5=%.4f", spec.name, n_q, ndcg5)
        except Exception:
            logger.exception("Pipeline %s failed — skipping.", spec.name)
            continue

    if not eval_results:
        print("\nERROR: All pipelines failed. Check logs above.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Pairwise comparisons vs the first successfully evaluated pipeline
    # ------------------------------------------------------------------
    completed_names = list(eval_results.keys())
    reference_name = completed_names[0]
    reference_ev = eval_results[reference_name]

    comparisons: dict[str, dict[str, Any]] = {}
    for name in completed_names:
        if name == reference_name:
            continue
        cmp = compare_runs(reference_ev, eval_results[name], metric="ndcg@5")
        comparisons[name] = cmp

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    _print_results_table(completed_names, eval_results, comparisons, reference_name)

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_docs": len(corpus),
        "n_questions": len(qa),
        "k": args.k,
        "pipelines_requested": requested,
        "pipelines_completed": completed_names,
        "reference_pipeline": reference_name,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
    }

    json_path = out_dir / "results.json"
    md_path = out_dir / "RESULTS.md"

    _write_json_report(completed_names, eval_results, comparisons, metadata, json_path)
    _write_markdown_report(completed_names, eval_results, comparisons, reference_name, md_path)

    print("Reports written to:")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
