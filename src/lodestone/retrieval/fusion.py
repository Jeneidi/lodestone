"""lodestone.retrieval.fusion — Hybrid retrieval via score fusion.

Two fusion strategies are implemented:

RRF (Reciprocal Rank Fusion)
----------------------------
For each retriever ``r`` and result at rank ``rank_r`` (1-based)::

    rrf_score(d) = Σ_r  1 / (rrf_k + rank_r(d))

where ``rrf_k`` (default 60) is a smoothing constant that reduces the
influence of very high ranks.  Chunks not returned by a retriever are
implicitly assigned an infinite rank (zero contribution).

Reference: Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning
Methods.  SIGIR 2009.

Weighted Score Fusion
---------------------
Each retriever's scores are min-max normalised to [0, 1] independently::

    norm_score_r(d) = (score_r(d) - min_r) / (max_r - min_r + ε)

Then a weighted average is computed::

    fused_score(d) = Σ_r  w_r * norm_score_r(d)

where ``w_r`` are caller-supplied weights (default: equal weights summing
to 1).  This strategy rewards consistent high scores across retrievers.

Implementation notes
--------------------
- Each child retriever is queried for ``k * 3`` candidates before fusion
  to increase recall at the fusion stage.
- Results are deduplicated by ``chunk_id``; the *first* encountered chunk
  object is kept (all retrievers should return the same Chunk content).
- Tie-breaking: stable sort on ``(-fused_score, chunk_id)`` for determinism.
"""

from __future__ import annotations

import logging

from lodestone.retrieval.base import Retriever
from lodestone.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

_EPSILON = 1e-9   # prevents divide-by-zero in min-max normalisation


class HybridRetriever(Retriever):
    """Hybrid retriever that fuses results from multiple child retrievers.

    Supports two fusion strategies:

    - ``"rrf"``      — Reciprocal Rank Fusion (rank-based; strategy-agnostic).
    - ``"weighted"`` — Weighted sum of min-max normalised scores.

    The ``name`` attribute reflects the strategy:
    ``"hybrid_rrf"`` or ``"hybrid_weighted"``.

    Args:
        retrievers: List of child :class:`~lodestone.retrieval.base.Retriever`
                    instances to fuse.  Must contain at least one.
        strategy:   Fusion strategy; ``"rrf"`` (default) or ``"weighted"``.
        rrf_k:      RRF smoothing constant (only used when strategy="rrf").
                    Defaults to the value from
                    :func:`~lodestone.config.get_settings`.
        weights:    Per-retriever weights for weighted fusion.  If ``None``,
                    equal weights are used.  Must have the same length as
                    *retrievers* when provided.  Values are normalised to
                    sum to 1 internally.

    Raises:
        ValueError: If *retrievers* is empty.
        ValueError: If *strategy* is not ``"rrf"`` or ``"weighted"``.
        ValueError: If *weights* length does not match *retrievers* length.

    Example (RRF)::

        hybrid = HybridRetriever([bm25, dense], strategy="rrf")
        hybrid.index(chunks)
        results = hybrid.search("neural networks", k=10)

    Example (weighted)::

        hybrid = HybridRetriever(
            [bm25, dense],
            strategy="weighted",
            weights=[0.3, 0.7],
        )

    """

    def __init__(
        self,
        retrievers: list[Retriever],
        strategy: str = "rrf",
        rrf_k: int | None = None,
        weights: list[float] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one child retriever.")
        if strategy not in ("rrf", "weighted"):
            raise ValueError(
                f"Unknown fusion strategy '{strategy}'. Choose 'rrf' or 'weighted'."
            )
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(
                f"weights length ({len(weights)}) must match retrievers length "
                f"({len(retrievers)})."
            )

        self._retrievers = list(retrievers)
        self._strategy = strategy

        # Resolve rrf_k from settings if not supplied
        if rrf_k is not None:
            self._rrf_k = rrf_k
        else:
            try:
                from lodestone.config import get_settings  # noqa: PLC0415
                self._rrf_k = get_settings().rrf_k
            except Exception:
                self._rrf_k = 60

        # Normalise weights to sum to 1
        if weights is None:
            n = len(retrievers)
            self._weights = [1.0 / n] * n
        else:
            total = sum(weights)
            if total <= 0:
                raise ValueError("Weights must sum to a positive number.")
            self._weights = [w / total for w in weights]

        self._indexed: bool = False

    @property
    def name(self) -> str:  # type: ignore[override]
        """Retriever name reflecting the fusion strategy."""
        return "hybrid_rrf" if self._strategy == "rrf" else "hybrid_weighted"

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Delegate :meth:`index` to all child retrievers.

        Args:
            chunks: All chunks to index.  Passed unchanged to each child.

        Raises:
            ValueError: If *chunks* is empty (propagated from children).

        """
        logger.info(
            "HybridRetriever (%s): indexing %d chunks across %d retrievers",
            self.name,
            len(chunks),
            len(self._retrievers),
        )
        for r in self._retrievers:
            r.index(chunks)
        self._indexed = True

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Retrieve and fuse results from all child retrievers.

        Fetches ``k * 3`` candidates from each child, deduplicates by
        ``chunk_id``, applies the fusion strategy, and returns the top *k*.

        Args:
            query: Natural-language query string.
            k:     Maximum number of results to return.

        Returns:
            Up to *k* :class:`~lodestone.schemas.ScoredChunk` objects (with
            ``retriever=self.name``) sorted by fused score descending.

        Raises:
            RuntimeError: If called before :meth:`index`.

        """
        if not self._indexed:
            raise RuntimeError("HybridRetriever.search() called before index().")

        fetch_k = k * 3
        per_retriever: list[list[ScoredChunk]] = [
            r.search(query, k=fetch_k) for r in self._retrievers
        ]

        if self._strategy == "rrf":
            fused = self._fuse_rrf(per_retriever)
        else:
            fused = self._fuse_weighted(per_retriever)

        return fused[:k]

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def _fuse_rrf(self, per_retriever: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        """Apply Reciprocal Rank Fusion.

        For each chunk seen in any retriever's results:

            rrf_score(d) = Σ_r  1 / (rrf_k + rank_r(d))

        Ranks are 1-based.  Chunks missing from a retriever contribute 0.

        Args:
            per_retriever: Results lists from each child retriever.

        Returns:
            Deduplicated list sorted by RRF score descending then chunk_id.

        """
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}

        for results in per_retriever:
            for rank, sc in enumerate(results, start=1):
                cid = sc.chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                if cid not in chunk_by_id:
                    chunk_by_id[cid] = sc.chunk

        return [
            ScoredChunk(chunk=chunk_by_id[cid], score=score, retriever=self.name)
            for cid, score in sorted(
                rrf_scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    def _fuse_weighted(self, per_retriever: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        """Apply weighted sum of min-max normalised scores.

        For each retriever *r*::

            norm_score_r(d) = (score_r(d) - min_r) / (max_r - min_r + ε)

            fused_score(d) = Σ_r  w_r * norm_score_r(d)

        Args:
            per_retriever: Results lists from each child retriever.

        Returns:
            Deduplicated list sorted by fused score descending then chunk_id.

        """
        chunk_by_id: dict[str, Chunk] = {}
        # For each retriever, build a map: chunk_id -> normalised score
        normalised_maps: list[dict[str, float]] = []

        for results in per_retriever:
            if not results:
                normalised_maps.append({})
                continue
            raw_scores = [sc.score for sc in results]
            min_s = min(raw_scores)
            max_s = max(raw_scores)
            denom = max_s - min_s + _EPSILON
            norm_map: dict[str, float] = {}
            for sc in results:
                cid = sc.chunk.chunk_id
                norm_map[cid] = (sc.score - min_s) / denom
                if cid not in chunk_by_id:
                    chunk_by_id[cid] = sc.chunk
            normalised_maps.append(norm_map)

        # Accumulate fused scores
        fused_scores: dict[str, float] = {}
        for cid in chunk_by_id:
            total = 0.0
            for w, norm_map in zip(self._weights, normalised_maps):
                total += w * norm_map.get(cid, 0.0)
            fused_scores[cid] = total

        return [
            ScoredChunk(chunk=chunk_by_id[cid], score=score, retriever=self.name)
            for cid, score in sorted(
                fused_scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]


__all__ = ["HybridRetriever"]
