"""evals.ablation — Ablation study runner.

Run via::

    python -m evals.ablation [options]
    make ablate

Grid
----
chunker in {fixed(200/40), sentence_window(3/2)}
× retriever in {bm25, dense, hybrid_rrf}
× rerank in {off, on}
= 12 configurations

CLI flags
---------
--max-questions   Cap QA examples per pipeline (default: 200).
--out             Output directory (default: evals/reports).

Outputs
-------
evals/reports/ablation.csv     Raw results table (one row per config).
evals/reports/ABLATION.md      Pivot-style markdown table.
evals/reports/ablation_ndcg.png     Grouped bar chart of ndcg@5 with CI error bars.
evals/reports/latency_quality.png   Scatter of p50 latency vs ndcg@5.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Deterministic seed for bootstrap CI in evaluate_run → stats.bootstrap_ci
_SEED = 42


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """One cell in the ablation grid.

    Attributes:
        label:         Short human-readable name for tables and plots.
        chunker_name:  ``"fixed"`` or ``"sentwin"``.
        retriever_name: ``"bm25"``, ``"dense"``, or ``"hybrid_rrf"``.
        use_rerank:    Whether a cross-encoder reranker is applied.

    """

    label: str
    chunker_name: str
    retriever_name: str
    use_rerank: bool


def _build_ablation_grid() -> list[AblationConfig]:
    """Return all 12 ablation configurations.

    Returns:
        List of :class:`AblationConfig` covering the full 2×3×2 grid.

    """
    configs: list[AblationConfig] = []
    for chunker_name, chunker_label in [("fixed", "fixed"), ("sentwin", "sentwin")]:
        for retriever_name in ["bm25", "dense", "hybrid_rrf"]:
            for use_rerank in [False, True]:
                rerank_label = "+rerank" if use_rerank else ""
                label = f"{retriever_name}_{chunker_label}{rerank_label}"
                configs.append(
                    AblationConfig(
                        label=label,
                        chunker_name=chunker_name,
                        retriever_name=retriever_name,
                        use_rerank=use_rerank,
                    )
                )
    return configs


def _config_to_pipeline_spec(cfg: AblationConfig) -> Any:
    """Convert an :class:`AblationConfig` to a :class:`~evals.runner.PipelineSpec`.

    Args:
        cfg: Ablation grid cell.

    Returns:
        A configured :class:`~evals.runner.PipelineSpec`.

    """
    from evals.runner import PipelineSpec
    from lodestone.chunking.strategies import FixedSizeChunker, SentenceWindowChunker
    from lodestone.config import get_settings
    from lodestone.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

    settings = get_settings()
    emb_model = settings.embedding_model_name

    if cfg.chunker_name == "fixed":
        chunker = FixedSizeChunker(chunk_size=200, overlap=40)
    else:
        chunker = SentenceWindowChunker(window_size=3, stride=2)

    def make_bm25():
        return BM25Retriever(k1=1.5, b=0.75)

    def make_dense():
        return DenseRetriever(model_name=emb_model)

    def make_hybrid():
        return HybridRetriever(
            retrievers=[BM25Retriever(k1=1.5, b=0.75), DenseRetriever(model_name=emb_model)],
            strategy="rrf",
            rrf_k=settings.rrf_k,
        )

    factories = {
        "bm25": make_bm25,
        "dense": make_dense,
        "hybrid_rrf": make_hybrid,
    }

    return PipelineSpec(
        name=cfg.label,
        chunker=chunker,
        retriever_factory=factories[cfg.retriever_name],
        use_rerank=cfg.use_rerank,
        use_rm3=False,
    )


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


def _build_dataframe(rows: list[dict[str, Any]]) -> Any:
    """Build a pandas DataFrame from a list of result dicts.

    Args:
        rows: Each dict has keys: label, chunker, retriever, rerank,
              ndcg5_mean, ndcg5_lo, ndcg5_hi,
              recall5_mean, recall5_lo, recall5_hi,
              mrr_mean, mrr_lo, mrr_hi,
              p50_ms.

    Returns:
        A pandas DataFrame with those columns.

    """
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_ablation_ndcg(df: Any, out_path: Path) -> None:
    """Save a grouped bar chart of ndcg@5 sorted descending.

    Error bars show the [ci_lo, ci_hi] bootstrap interval.

    Args:
        df:       DataFrame with columns: label, ndcg5_mean, ndcg5_lo, ndcg5_hi.
        out_path: Destination .png path.

    """
    df_sorted = df.sort_values("ndcg5_mean", ascending=False).reset_index(drop=True)

    labels = df_sorted["label"].tolist()
    means = df_sorted["ndcg5_mean"].to_numpy()
    errs_lo = (means - df_sorted["ndcg5_lo"].to_numpy()).clip(min=0.0)
    errs_hi = (df_sorted["ndcg5_hi"].to_numpy() - means).clip(min=0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=[errs_lo, errs_hi],
        capsize=4,
        color="#4C72B0",
        edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("nDCG@5", fontsize=11)
    ax.set_ylim(0, min(1.0, means.max() * 1.2 + 0.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bars with mean value
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ablation ndcg plot to %s", out_path)


def _plot_latency_quality(df: Any, out_path: Path) -> None:
    """Save a scatter plot of p50 latency (x, log scale) vs ndcg@5 (y).

    Each point is labelled with the config name.

    Args:
        df:       DataFrame with columns: label, p50_ms, ndcg5_mean.
        out_path: Destination .png path.

    """
    labels = df["label"].tolist()
    x = df["p50_ms"].to_numpy()
    y = df["ndcg5_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=60, color="#DD8452", edgecolors="white", linewidths=0.8, zorder=3)

    for label, xi, yi in zip(labels, x, y):
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )

    ax.set_xscale("log")
    ax.set_xlabel("p50 Latency (ms, log scale)", fontsize=11)
    ax.set_ylabel("nDCG@5", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved latency/quality scatter to %s", out_path)


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _write_ablation_markdown(df: Any, out_path: Path) -> None:
    """Write a pivot-style markdown table for the ablation results.

    Args:
        df:       Ablation DataFrame.
        out_path: Destination .md file path.

    """
    from datetime import datetime, timezone  # noqa: PLC0415

    lines: list[str] = []
    lines.append("# Lodestone Ablation Study\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append("\n## Results Grid (sorted by nDCG@5 desc)\n")

    header = (
        "| Config | Chunker | Retriever | Rerank "
        "| nDCG@5 (CI) | Recall@5 (CI) | MRR (CI) | p50 lat |"
    )
    sep = (
        "|"
        + "|".join(["-" * 30, "-" * 10, "-" * 12, "-" * 8, "-" * 26, "-" * 26, "-" * 24, "-" * 10])
        + "|"
    )
    lines.append(header)
    lines.append(sep)

    df_sorted = df.sort_values("ndcg5_mean", ascending=False).reset_index(drop=True)

    def _ci_cell(mean: float, lo: float, hi: float) -> str:
        return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

    for _, row in df_sorted.iterrows():
        rerank_str = "yes" if row["rerank"] else "no"
        cells = [
            f"`{row['label']}`",
            row["chunker"],
            row["retriever"],
            rerank_str,
            _ci_cell(row["ndcg5_mean"], row["ndcg5_lo"], row["ndcg5_hi"]),
            _ci_cell(row["recall5_mean"], row["recall5_lo"], row["recall5_hi"]),
            _ci_cell(row["mrr_mean"], row["mrr_lo"], row["mrr_hi"]),
            f"{row['p50_ms']:.1f} ms",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote ablation markdown to %s", out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the ablation grid and write all output artefacts.

    Returns:
        Exit code (0 on success, 1 on error).

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="python -m evals.ablation",
        description="Lodestone ablation sweep: chunker × retriever × rerank grid.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=200,
        dest="max_questions",
        help="Cap QA examples per pipeline (default: 200).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evals/reports",
        help="Output directory (default: evals/reports).",
    )
    args = parser.parse_args()

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
            "\nPlease run `make data` to build the dataset before running the ablation.",
            file=sys.stderr,
        )
        return 1

    if args.max_questions and args.max_questions > 0:
        qa = qa[: args.max_questions]
        logger.info("Capped QA set to %d questions.", len(qa))

    logger.info("Corpus: %d documents | QA: %d examples", len(corpus), len(qa))

    from evals.aggregate import evaluate_run  # noqa: PLC0415
    from evals.runner import run_retrieval  # noqa: PLC0415
    from lodestone.config import get_settings  # noqa: PLC0415

    settings = get_settings()

    # ------------------------------------------------------------------
    # Run the 12-cell grid
    # ------------------------------------------------------------------
    grid = _build_ablation_grid()
    rows: list[dict[str, Any]] = []

    for cfg in grid:
        logger.info("=== Ablation config: %s ===", cfg.label)
        spec = _config_to_pipeline_spec(cfg)
        try:
            retriever, reranker = spec.build(corpus)
            run = run_retrieval(
                retriever=retriever,
                qa=qa,
                k=10,
                reranker=reranker,
                rerank_top_k=settings.rerank_top_k,
            )
            ev = evaluate_run(run, qa, ks=(1, 3, 5, 10))
        except Exception:
            logger.exception("Config %s failed — skipping.", cfg.label)
            continue

        summary = ev.get("summary", {})
        latency = ev.get("latency", {})

        def _get(metric: str, _summary: dict = summary) -> tuple[float, float, float]:
            m = _summary.get(metric, {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
            return m["mean"], m["ci_lo"], m["ci_hi"]

        ndcg5_mean, ndcg5_lo, ndcg5_hi = _get("ndcg@5")
        recall5_mean, recall5_lo, recall5_hi = _get("recall@5")
        mrr_mean, mrr_lo, mrr_hi = _get("mrr")
        p50 = latency.get("p50_ms", 0.0)

        rows.append(
            {
                "label": cfg.label,
                "chunker": cfg.chunker_name,
                "retriever": cfg.retriever_name,
                "rerank": cfg.use_rerank,
                "ndcg5_mean": ndcg5_mean,
                "ndcg5_lo": ndcg5_lo,
                "ndcg5_hi": ndcg5_hi,
                "recall5_mean": recall5_mean,
                "recall5_lo": recall5_lo,
                "recall5_hi": recall5_hi,
                "mrr_mean": mrr_mean,
                "mrr_lo": mrr_lo,
                "mrr_hi": mrr_hi,
                "p50_ms": p50,
            }
        )
        logger.info(
            "Config %s: ndcg@5=%.4f  recall@5=%.4f  mrr=%.4f  p50=%.1f ms",
            cfg.label,
            ndcg5_mean,
            recall5_mean,
            mrr_mean,
            p50,
        )

    if not rows:
        print("\nERROR: All ablation configs failed. Check logs above.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Build DataFrame and write outputs
    # ------------------------------------------------------------------
    df = _build_dataframe(rows)

    csv_path = out_dir / "ablation.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote ablation CSV to %s", csv_path)

    md_path = out_dir / "ABLATION.md"
    _write_ablation_markdown(df, md_path)

    ndcg_plot_path = out_dir / "ablation_ndcg.png"
    _plot_ablation_ndcg(df, ndcg_plot_path)

    scatter_path = out_dir / "latency_quality.png"
    _plot_latency_quality(df, scatter_path)

    print("\nAblation outputs written to:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {ndcg_plot_path}")
    print(f"  {scatter_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
