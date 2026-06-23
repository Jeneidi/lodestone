"""lodestone.engine — high-level facade for the Lodestone retrieval + generation stack.

This module provides :class:`LodestoneEngine`, a single entry-point that wires
together corpus loading, chunking, hybrid retrieval, optional cross-encoder
reranking, and answer generation.  It is the object consumed by the API server,
CLI, and dashboard — none of those layers should interact with retrieval or
generation internals directly.

Usage::

    from lodestone.engine import get_engine

    engine = get_engine()            # singleton; loads lazily on first call
    results = engine.search("What is backpropagation?", k=5)
    answer  = engine.ask("What is backpropagation?", k=5, score_faithfulness=True)
"""

from __future__ import annotations

import logging
import time

from lodestone.schemas import Answer, ScoredChunk

logger = logging.getLogger(__name__)


class LodestoneEngine:
    """High-level facade over the full Lodestone pipeline.

    Responsible for:

    - Loading the corpus from disk via :func:`~lodestone.data.load_corpus`.
    - Chunking documents with :class:`~lodestone.chunking.strategies.FixedSizeChunker`.
    - Building a hybrid BM25 + Dense retrieval pipeline via
      :func:`~lodestone.retrieval.build_default_pipeline`.
    - Optionally constructing a :class:`~lodestone.retrieval.CrossEncoderReranker`
      for second-stage re-scoring.
    - Delegating answer generation to the appropriate
      :func:`~lodestone.generation.get_answerer` backend.
    - Optionally computing NLI faithfulness via
      :class:`~lodestone.generation.NliFaithfulnessScorer`.

    The engine is *lazy*: heavy models are only initialised on the first
    :meth:`load` call (or implicitly on the first :meth:`search` / :meth:`ask`
    call when using :func:`get_engine`).

    Args:
        settings: Optional :class:`~lodestone.config.Settings` instance.
                  When ``None``, the singleton from
                  :func:`~lodestone.config.get_settings` is used.

    Example::

        from lodestone.engine import LodestoneEngine

        engine = LodestoneEngine()
        engine.load(use_rerank=True)
        hits = engine.search("transformer architecture", k=10)
        ans  = engine.ask("What is a transformer?", k=5, score_faithfulness=False)

    """

    def __init__(self, settings: object | None = None) -> None:
        if settings is None:
            from lodestone.config import get_settings  # noqa: PLC0415
            settings = get_settings()
        self._settings = settings
        self._retriever: object | None = None          # HybridRetriever once loaded
        self._reranker: object | None = None           # CrossEncoderReranker or None
        self._faithfulness_scorer: object | None = None  # NliFaithfulnessScorer or None
        self._loaded: bool = False
        self._use_rerank: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, use_rerank: bool = True) -> None:
        """Load the corpus, build indexes, and optionally prepare the reranker.

        This method is *idempotent*: calling it multiple times with the same
        arguments is safe and only performs work on the first call.

        Args:
            use_rerank: When ``True`` (default), a
                        :class:`~lodestone.retrieval.CrossEncoderReranker` is
                        constructed and used in :meth:`search` / :meth:`ask`.
                        Set to ``False`` to use the hybrid retriever score directly.

        Raises:
            FileNotFoundError: Re-raised from :func:`~lodestone.data.load_corpus`
                when the corpus file has not been built yet.  The error message
                instructs the user to run ``make data``.
            ValueError: If the corpus produces no chunks (empty corpus).

        """
        if self._loaded and self._use_rerank == use_rerank:
            logger.debug("LodestoneEngine.load: already loaded, skipping.")
            return

        from lodestone.chunking.strategies import FixedSizeChunker  # noqa: PLC0415
        from lodestone.data import load_corpus  # noqa: PLC0415
        from lodestone.retrieval import (  # noqa: PLC0415
            CrossEncoderReranker,
            build_default_pipeline,
        )

        logger.info("LodestoneEngine: loading corpus...")
        documents = load_corpus()  # FileNotFoundError propagates if missing
        logger.info("LodestoneEngine: loaded %d documents.", len(documents))

        chunker = FixedSizeChunker()
        chunks = []
        for doc in documents:
            chunks.extend(chunker.chunk(doc))
        logger.info("LodestoneEngine: produced %d chunks.", len(chunks))

        if not chunks:
            raise ValueError(
                "LodestoneEngine: corpus produced zero chunks.  "
                "Ensure the corpus file is non-empty."
            )

        logger.info("LodestoneEngine: building hybrid retrieval pipeline...")
        self._retriever = build_default_pipeline(chunks, settings=self._settings)
        logger.info("LodestoneEngine: pipeline ready.")

        self._use_rerank = use_rerank
        if use_rerank:
            logger.info("LodestoneEngine: initialising cross-encoder reranker.")
            self._reranker = CrossEncoderReranker()
        else:
            self._reranker = None

        self._loaded = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the engine with defaults if :meth:`load` has not been called."""
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Retrieve the top-*k* chunks for *query*.

        Retrieves ``k * 3`` candidates from the hybrid retriever and then
        re-scores them with the cross-encoder (if reranking is enabled),
        returning the top *k*.  When reranking is disabled the top *k* raw
        results are returned directly.

        Args:
            query: Natural-language query string.
            k:     Number of results to return.  Defaults to 10.

        Returns:
            List of up to *k* :class:`~lodestone.schemas.ScoredChunk` objects,
            sorted by score descending.

        Raises:
            FileNotFoundError: If the engine has not been loaded and corpus
                is missing.

        """
        self._ensure_loaded()

        from lodestone.retrieval import HybridRetriever  # noqa: PLC0415
        retriever: HybridRetriever = self._retriever  # type: ignore[assignment]

        if self._use_rerank and self._reranker is not None:
            candidates = retriever.search(query, k=k * 3)
            from lodestone.retrieval import CrossEncoderReranker  # noqa: PLC0415
            reranker: CrossEncoderReranker = self._reranker  # type: ignore[assignment]
            results = reranker.rerank(query, candidates, top_k=k)
        else:
            results = retriever.search(query, k=k)

        return results

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        k: int = 5,
        score_faithfulness: bool = False,
    ) -> Answer:
        """Answer *query* using retrieved evidence.

        Retrieves the top-*k* chunks (via :meth:`search`), passes them to the
        configured answerer, and optionally scores answer faithfulness.  The
        :attr:`~lodestone.schemas.Answer.latency_ms` field reflects the full
        end-to-end wall-clock time (retrieval + generation + optional NLI).

        Args:
            query:              Natural-language question string.
            k:                  Number of supporting chunks to retrieve.
                                Defaults to 5.
            score_faithfulness: When ``True``, lazily initialise an
                                :class:`~lodestone.generation.NliFaithfulnessScorer`
                                and fill
                                :attr:`~lodestone.schemas.Answer.faithfulness`.
                                Defaults to ``False``.

        Returns:
            An :class:`~lodestone.schemas.Answer` with ``text``,
            ``supporting_chunks``, ``generator``, ``faithfulness`` (or ``None``),
            and ``latency_ms`` set.

        Raises:
            FileNotFoundError: If corpus is missing and the engine has not
                been loaded.

        """
        t_start = time.perf_counter()

        chunks = self.search(query, k=k)

        from lodestone.generation import NliFaithfulnessScorer, get_answerer  # noqa: PLC0415
        answerer = get_answerer(settings=self._settings)
        answer = answerer.answer(query, chunks)

        if score_faithfulness:
            if self._faithfulness_scorer is None:
                logger.info("LodestoneEngine: initialising NLI faithfulness scorer.")
                self._faithfulness_scorer = NliFaithfulnessScorer()
            scorer: NliFaithfulnessScorer = self._faithfulness_scorer  # type: ignore[assignment]
            answer = answer.model_copy(
                update={"faithfulness": scorer.score(answer.text, chunks)}
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        answer = answer.model_copy(update={"latency_ms": latency_ms})

        return answer


# ---------------------------------------------------------------------------
# Module-level cached singleton
# ---------------------------------------------------------------------------

_engine_instance: LodestoneEngine | None = None


def get_engine() -> LodestoneEngine:
    """Return the module-level singleton :class:`LodestoneEngine`.

    The engine is constructed on the first call and cached for all subsequent
    calls.  On first use :meth:`LodestoneEngine.load` is called automatically
    with default settings (``use_rerank=True``).

    Returns:
        The cached :class:`LodestoneEngine` instance, fully loaded.

    Raises:
        FileNotFoundError: If the corpus has not been built yet.  Run
            ``make data`` to generate it.

    Example::

        from lodestone.engine import get_engine

        engine = get_engine()
        answer = engine.ask("What is attention?")

    """
    global _engine_instance  # noqa: PLW0603
    if _engine_instance is None:
        _engine_instance = LodestoneEngine()
        _engine_instance.load()
    return _engine_instance


__all__ = ["LodestoneEngine", "get_engine"]
