"""Statistical utilities for comparing retrieval system configurations.

Design rationale
----------------
**Unit of resampling for bootstrap CI**
    In IR evaluation, each query is an independent observation.  A system's
    overall metric score (e.g. MAP, MRR) is the mean of per-query values.
    Bootstrap resampling must therefore operate at the **query level**: we
    draw *n* queries with replacement (not individual rank positions).  This
    preserves the correlation structure within each query's ranked list and
    gives a CI on the *mean across queries*, which is the quantity of
    interest for comparing systems.  Resampling at the position or document
    level would violate independence assumptions and produce overconfident
    intervals.

**Paired permutation test**
    When two systems are evaluated on the **same** set of queries, their
    per-query scores are positively correlated (hard queries tend to be hard
    for both systems; easy queries easy for both).  An unpaired test ignores
    this structure and has lower statistical power.  The paired permutation
    test computes per-query *differences* d_i = a_i - b_i and tests whether
    the mean of d_i differs from 0.  Under H₀ (the two systems are equally
    good), the sign of each d_i is exchangeable, so we randomly flip signs
    and measure how often the permuted mean is at least as extreme as the
    observed mean.  This is exact (non-parametric), makes no distributional
    assumptions, and fully exploits the pairing.

All functions are **pure** (no I/O, no side-effects) and require only
numpy (scipy is not needed but allowed).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a percentile bootstrap confidence interval over per-query values.

    Algorithm
    ---------
    1. Treat ``values`` as i.i.d. per-query metric scores.
    2. Draw ``n_resamples`` bootstrap samples of size ``n`` with replacement.
    3. Compute the mean of each bootstrap sample.
    4. Return the ``(1-confidence)/2`` and ``(1+confidence)/2`` percentiles
       of the bootstrap mean distribution as the CI bounds.

    Formula
    -------
    Let :math:`\\hat{\\mu}` be the sample mean and
    :math:`\\hat{\\mu}^*_b` the mean of bootstrap sample *b*.

    .. math::

        \\text{lo} = \\text{percentile}\\left(
            \\{\\hat{\\mu}^*_b\\}_{b=1}^{B},\\;
            \\frac{1 - c}{2} \\cdot 100
        \\right)

    .. math::

        \\text{hi} = \\text{percentile}\\left(
            \\{\\hat{\\mu}^*_b\\}_{b=1}^{B},\\;
            \\frac{1 + c}{2} \\cdot 100
        \\right)

    This is the **percentile bootstrap** (not BCa), which is unbiased for
    symmetric distributions and adequate for IR metrics in practice.

    Why per-query bootstrap?
    ------------------------
    See module docstring.  The query is the natural independent unit in IR
    evaluation; resampling queries preserves within-query correlation while
    correctly estimating uncertainty in the cross-query mean.

    Parameters
    ----------
    values:
        Per-query metric scores (e.g. one AP value per query).
    n_resamples:
        Number of bootstrap samples to draw.  2000 is sufficient for 95 %
        CIs; increase to 10 000 for publication-grade results.
    confidence:
        Desired confidence level, e.g. 0.95 for a 95 % CI.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, lo, hi)`` where ``mean`` is the sample mean of ``values``
        and ``[lo, hi]`` is the percentile bootstrap CI.

    Edge cases
    ----------
    * Single value → lo == hi == mean (CI collapses).
    * All identical values → lo == hi == mean.
    * Empty ``values`` → returns (0.0, 0.0, 0.0).

    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(arr))

    rng = np.random.default_rng(seed)
    n = arr.size
    # Draw all bootstrap indices at once: shape (n_resamples, n)
    indices = rng.integers(0, n, size=(n_resamples, n))
    # Bootstrap sample means: shape (n_resamples,)
    boot_means = arr[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boot_means, alpha * 100.0))
    hi = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return mean_val, lo, hi


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test for mean(a) == mean(b).

    Null hypothesis
    ---------------
    H₀: The two systems have equal expected per-query performance, i.e.
    E[a_i - b_i] = 0 for all queries i.

    Why paired?
    -----------
    See module docstring.  When both systems are evaluated on the same query
    set, the per-query differences d_i = a_i - b_i are the correct unit of
    comparison.  Under H₀, the sign of each d_i is exchangeable (either
    system could have been better), so we randomly flip signs.

    Algorithm
    ---------
    1. Compute differences ``d = a - b`` (length ``n``).
    2. Compute the observed test statistic ``T_obs = |mean(d)|``.
    3. Generate a matrix of random sign-flips of shape ``(n_permutations, n)``
       with entries ±1 (vectorised, no Python loop over permutations).
    4. Compute permuted means and count how many satisfy
       ``|mean(d_perm)| >= T_obs``.
    5. p-value = count / n_permutations (two-sided).

    Formula
    -------
    .. math::

        p = \\frac{
            \\#\\{b : |\\bar{d}^{(b)}| \\ge |\\bar{d}|\\}
        }{B}

    where :math:`\\bar{d}^{(b)}` is the mean of the sign-flipped differences
    in permutation *b*.

    Parameters
    ----------
    a:
        Per-query scores for system A (must have same length as ``b``).
    b:
        Per-query scores for system B.
    n_permutations:
        Number of random sign-flip permutations.  10 000 gives stable
        p-values to roughly ±0.005.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    float
        Two-sided p-value in (0.0, 1.0].

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or are empty.

    Edge cases
    ----------
    * Identical sequences (a == b) → d is all zeros → p ≈ 1.0.
    * Clearly separated differences → small p (< 0.01 for large effects).

    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("paired_permutation_test requires non-empty sequences.")
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"a and b must have the same length; got {a_arr.size} vs {b_arr.size}.")

    d = a_arr - b_arr
    n = d.size
    t_obs = abs(d.mean())

    rng = np.random.default_rng(seed)
    # Matrix of ±1 sign flips: shape (n_permutations, n)
    # rng.integers(0, 2, ...) gives 0/1; 2*x - 1 maps to ±1
    signs = 2 * rng.integers(0, 2, size=(n_permutations, n), dtype=np.int8) - 1
    # Permuted means: shape (n_permutations,)
    perm_means = (signs * d).mean(axis=1)

    p_value = float(np.mean(np.abs(perm_means) >= t_obs))
    # Ensure p-value is never exactly 0 (use 1/n_permutations as floor)
    return max(p_value, 1.0 / n_permutations)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Compute summary statistics for a sequence of per-query metric values.

    Parameters
    ----------
    values:
        Sequence of floats (e.g. per-query AP or nDCG values).

    Returns
    -------
    dict[str, float]
        Dictionary with keys:

        * ``"mean"``  — arithmetic mean.
        * ``"std"``   — population standard deviation (ddof=0).
        * ``"p50"``   — 50th percentile (median).
        * ``"p95"``   — 95th percentile.

        All values are 0.0 for an empty input.

    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
