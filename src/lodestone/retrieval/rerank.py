"""lodestone.retrieval.rerank — Cross-encoder reranker using sentence-transformers.

Architecture
------------
The cross-encoder jointly encodes each ``(query, chunk_text)`` pair and
outputs a relevance logit.  Unlike bi-encoders, the query and document
interact through full attention, giving much higher-quality relevance
estimates at the cost of O(n) model calls (one per candidate).

Scoring
-------
Raw cross-encoder logits are passed through a sigmoid function to produce
scores in [0, 1]::

    score = sigmoid(logit) = 1 / (1 + exp(-logit))

This makes scores interpretable as approximate relevance probabilities and
comparable across queries.

Lazy imports
------------
``sentence_transformers.CrossEncoder`` is imported *inside* the method that
needs it so that ``import lodestone.retrieval.rerank`` succeeds even when
sentence-transformers is not installed.

Injectable scorer
-----------------
The constructor accepts an optional ``scorer`` callable
``Callable[[list[tuple[str, str]]], np.ndarray]`` that replaces the
CrossEncoder model.  This is intended for unit testing (avoids network/GPU
dependency) and for swapping in custom ranking backends.  When ``scorer``
is provided, *model_name* is ignored.

Example (production)::

    reranker = CrossEncoderReranker()
    top_results = reranker.rerank(query, candidates, top_k=5)

Example (testing)::

    def fake_scorer(pairs):
        return np.zeros(len(pairs), dtype=np.float32)

    reranker = CrossEncoderReranker(scorer=fake_scorer)
    top_results = reranker.rerank("test", candidates, top_k=3)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from lodestone.schemas import ScoredChunk

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid: ``1 / (1 + exp(-x))``.

    Numerically stable implementation that avoids overflow for large
    negative values.

    Args:
        x: Input array of any shape.

    Returns:
        Array of the same shape with values in (0, 1).

    """
    # Clip to avoid exp overflow in float32
    x_clipped = np.clip(x, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class CrossEncoderReranker:
    """Cross-encoder reranker that re-scores a candidate set.

    Encodes each ``(query, chunk_text)`` pair through a cross-encoder
    model, applies sigmoid to the logits, and returns the top-*k* chunks
    sorted by the new scores.

    Args:
        model_name:  HuggingFace model identifier for the cross-encoder.
                     Defaults to the value from
                     :func:`~lodestone.config.get_settings`
                     (``"cross-encoder/ms-marco-MiniLM-L-6-v2"``).
                     Ignored when *scorer* is provided.
        batch_size:  Number of pairs to score in one forward pass.
                     Defaults to 32.
        scorer:      Optional callable ``(list[tuple[str, str]]) -> np.ndarray``
                     that replaces the CrossEncoder.  Each input element is
                     a ``(query, passage)`` pair; output is a 1-D float
                     array of raw logits (one per pair).  For testing and
                     custom ranking backends.

    Example::

        reranker = CrossEncoderReranker()
        candidates = retriever.search("deep learning", k=20)
        top5 = reranker.rerank("deep learning", candidates, top_k=5)

    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        scorer: Callable[[list[tuple[str, str]]], np.ndarray] | None = None,
    ) -> None:
        self._model_name = model_name
        self.batch_size = batch_size
        self._scorer_override = scorer
        # Cached CrossEncoder instance (loaded once on first score call)
        self._ce_model: object | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model_name(self) -> str:
        """Return resolved reranker model name (falls back to settings)."""
        if self._model_name is not None:
            return self._model_name
        from lodestone.config import get_settings  # noqa: PLC0415
        return get_settings().reranker_model_name

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score ``(query, passage)`` pairs and return raw logits.

        Uses the injected ``scorer`` if available, otherwise lazily loads
        a ``sentence_transformers.CrossEncoder`` model.

        Args:
            pairs: List of ``(query_text, passage_text)`` tuples.

        Returns:
            Float32 numpy array of shape ``(len(pairs),)`` containing raw
            relevance logits.

        """
        if self._scorer_override is not None:
            result = self._scorer_override(pairs)
            return np.asarray(result, dtype=np.float32)

        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        if self._ce_model is None:
            model_name = self._get_model_name()
            logger.info("CrossEncoderReranker: loading model '%s'", model_name)
            self._ce_model = CrossEncoder(model_name)
        logits = self._ce_model.predict(  # type: ignore[union-attr]
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(logits, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Re-score *candidates* with a cross-encoder and return the top *top_k*.

        Scoring pipeline:

        1. Build ``(query, chunk_text)`` pairs for all *candidates*.
        2. Score all pairs via the cross-encoder (in *batch_size* batches).
        3. Apply sigmoid to map logits → [0, 1].
        4. Return the top *top_k* candidates sorted by new score descending.
           The ``retriever`` field of each :class:`~lodestone.schemas.ScoredChunk`
           is updated to ``f"rerank({original_retriever_name})"``.

        Tie-breaking is by ``chunk_id`` (ascending) for determinism.

        Args:
            query:      Natural-language query string.
            candidates: Candidate chunks to re-score (typically the output
                        of a first-stage retriever).
            top_k:      Number of top results to return.  Defaults to 5.

        Returns:
            Up to *top_k* :class:`~lodestone.schemas.ScoredChunk` objects,
            sorted by reranker score descending.  Returns ``[]`` if
            *candidates* is empty.

        """
        if not candidates:
            logger.debug("CrossEncoderReranker.rerank: no candidates, returning []")
            return []

        pairs: list[tuple[str, str]] = [
            (query, sc.chunk.text) for sc in candidates
        ]

        logits = self._score_pairs(pairs)       # shape (n,)
        scores = _sigmoid(logits)               # shape (n,), in (0, 1)

        # Build ScoredChunk list with updated scores and retriever name
        reranked: list[ScoredChunk] = []
        for sc, new_score in zip(candidates, scores.tolist()):
            orig_name = sc.retriever or "unknown"
            reranked.append(
                ScoredChunk(
                    chunk=sc.chunk,
                    score=float(new_score),
                    retriever=f"rerank({orig_name})",
                )
            )

        # Sort: descending score, then ascending chunk_id for determinism
        reranked.sort(key=lambda x: (-x.score, x.chunk.chunk_id))

        logger.debug(
            "CrossEncoderReranker: re-scored %d candidates, returning top %d",
            len(candidates),
            min(top_k, len(reranked)),
        )
        return reranked[:top_k]


__all__ = ["CrossEncoderReranker"]
