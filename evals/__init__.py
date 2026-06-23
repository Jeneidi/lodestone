"""Lodestone evaluation laboratory.

This package is scaffolded in wave 1.  The full evaluation harness and
ablation runner are implemented in wave 4.

Planned modules:

``evals.metrics``
    Retrieval metrics: Recall@k, MRR, nDCG@k, Precision@k.
    Answer quality metrics: faithfulness (NLI), exact match, F1.

``evals.harness``
    End-to-end evaluation loop over a QA dataset.
    Entry point: ``python -m evals.harness`` (or ``make eval``).

``evals.ablation``
    Grid-search runner: sweeps chunking × retriever × reranking combos.
    Emits per-configuration metric tables, latency percentiles,
    comparison plots (matplotlib), and a markdown report.
    Entry point: ``python -m evals.ablation`` (or ``make ablate``).

``evals.stats``
    Bootstrap confidence intervals and paired permutation significance
    tests for comparing retrieval configurations.
"""
