"""evals.runner — Pipeline construction, indexing, and retrieval execution engine.

Public surface
--------------
- :func:`build_chunks`      — chunk a corpus with any Chunker.
- :func:`run_retrieval`     — time-each-query retrieval loop with optional reranking.
- :class:`PipelineSpec`     — dataclass describing a named pipeline configuration.
- :func:`default_pipelines` — factory for the six standard eval pipelines.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from tqdm import tqdm

from lodestone.chunking.strategies import FixedSizeChunker, SentenceWindowChunker
from lodestone.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    ExpandingRetriever,
    HybridRetriever,
    Retriever,
    Rm3QueryExpander,
)
from lodestone.schemas import Chunk, Document, QAExample, RetrievalRunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------


def build_chunks(corpus: list[Document], chunker: object) -> list[Chunk]:
    """Chunk every document in *corpus* using *chunker*.

    The *chunker* must expose a ``chunk(doc: Document) -> list[Chunk]``
    method (i.e. satisfy the Chunker protocol defined by
    :class:`~lodestone.chunking.strategies.FixedSizeChunker` /
    :class:`~lodestone.chunking.strategies.SentenceWindowChunker`).

    Args:
        corpus:  List of :class:`~lodestone.schemas.Document` objects to chunk.
        chunker: A Chunker instance with a ``chunk`` method.

    Returns:
        Flat ordered list of :class:`~lodestone.schemas.Chunk` objects,
        preserving document order and intra-document chunk order.

    Raises:
        ValueError: If *corpus* is empty.

    """
    if not corpus:
        raise ValueError("build_chunks: corpus must not be empty.")

    chunks: list[Chunk] = []
    for doc in corpus:
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)

    logger.info(
        "build_chunks: %d documents -> %d chunks (chunker=%s)",
        len(corpus),
        len(chunks),
        getattr(chunker, "name", type(chunker).__name__),
    )
    return chunks


# ---------------------------------------------------------------------------
# Retrieval execution loop
# ---------------------------------------------------------------------------


def run_retrieval(
    retriever: Retriever,
    qa: list[QAExample],
    k: int = 10,
    reranker: CrossEncoderReranker | None = None,
    rerank_top_k: int = 5,
) -> list[RetrievalRunResult]:
    """Run retrieval over every question in *qa*, recording per-query latency.

    For each query:

    1. If *reranker* is ``None``: call ``retriever.search(question, k=k)``
       and record wall-clock time.
    2. If *reranker* is provided: fetch ``k * 3`` first-stage candidates,
       rerank with :meth:`~CrossEncoderReranker.rerank` to *k* results.
       Both the retrieval **and** rerank time are included in ``latency_ms``.

    A ``tqdm`` progress bar is shown on stderr.

    Args:
        retriever:    An already-indexed :class:`~lodestone.retrieval.base.Retriever`.
        qa:           QA examples to evaluate; one retrieval call per example.
        k:            Number of final results to return per query.
        reranker:     Optional :class:`~lodestone.retrieval.rerank.CrossEncoderReranker`.
                      When provided the first stage fetches ``k * 3`` candidates.
        rerank_top_k: Number of results to keep after reranking.  Must be <= k.
                      Only used when *reranker* is not ``None``.

    Returns:
        One :class:`~lodestone.schemas.RetrievalRunResult` per QA example,
        in the same order as *qa*.

    """
    results: list[RetrievalRunResult] = []
    fetch_k = k * 3 if reranker is not None else k

    for example in tqdm(qa, desc="Retrieving", unit="q", dynamic_ncols=True):
        t0 = time.perf_counter()

        candidates = retriever.search(example.question, k=fetch_k)

        if reranker is not None:
            candidates = reranker.rerank(example.question, candidates, top_k=rerank_top_k)
            # Pad or truncate to exactly k if needed
            candidates = candidates[:k]

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1_000.0

        results.append(
            RetrievalRunResult(
                qid=example.qid,
                retrieved=candidates,
                latency_ms=latency_ms,
            )
        )

    logger.info(
        "run_retrieval: evaluated %d queries (retriever=%s, reranker=%s, k=%d)",
        len(results),
        getattr(retriever, "name", type(retriever).__name__),
        "yes" if reranker is not None else "no",
        k,
    )
    return results


# ---------------------------------------------------------------------------
# PipelineSpec dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineSpec:
    """Declarative specification for a retrieval pipeline.

    Attributes:
        name:               Human-readable identifier used in reports.
        chunker:            An instantiated Chunker (must have a ``chunk`` method).
        retriever_factory:  Zero-argument callable that returns a fresh,
                            un-indexed :class:`~lodestone.retrieval.base.Retriever`.
        use_rerank:         If ``True``, a :class:`~CrossEncoderReranker`
                            will be constructed and applied after retrieval.
        use_rm3:            If ``True``, the retriever returned by
                            *retriever_factory* is wrapped in an
                            :class:`~ExpandingRetriever` before indexing.

    """

    name: str
    chunker: object
    retriever_factory: Callable[[], Retriever]
    use_rerank: bool = False
    use_rm3: bool = False

    def build(self, corpus: list[Document]) -> tuple[Retriever, CrossEncoderReranker | None]:
        """Materialise the pipeline: chunk, construct, and index.

        Steps:

        1. Call :func:`build_chunks` to produce chunks from *corpus*.
        2. Call *retriever_factory* to obtain a fresh retriever.
        3. If ``use_rm3`` is ``True``, construct an
           :class:`~Rm3QueryExpander` using the raw retriever (after indexing
           the raw retriever first) and wrap it in an
           :class:`~ExpandingRetriever`.
        4. If ``use_rerank`` is ``True``, construct a
           :class:`~CrossEncoderReranker` with default settings.
        5. Return ``(retriever, reranker | None)``.

        Args:
            corpus: Source documents to chunk and index.

        Returns:
            A 2-tuple ``(retriever, reranker)`` where *reranker* is ``None``
            when ``use_rerank=False``.

        """
        logger.info("PipelineSpec.build: building pipeline '%s'", self.name)

        chunks = build_chunks(corpus, self.chunker)

        retriever = self.retriever_factory()

        if self.use_rm3:
            # Index the inner retriever first so the expander can search it
            retriever.index(chunks)
            expander = Rm3QueryExpander(retriever, fb_docs=5, fb_terms=8)
            retriever = ExpandingRetriever(inner=retriever, expander=expander)
            # ExpandingRetriever.index delegates back to the inner retriever,
            # which is already indexed — calling index again is harmless because
            # it just re-builds, but we skip it to avoid re-indexing overhead.
        else:
            retriever.index(chunks)

        reranker: CrossEncoderReranker | None = None
        if self.use_rerank:
            reranker = CrossEncoderReranker()

        logger.info(
            "PipelineSpec.build: pipeline '%s' ready (%d chunks, rm3=%s, rerank=%s)",
            self.name,
            len(chunks),
            self.use_rm3,
            self.use_rerank,
        )
        return retriever, reranker


# ---------------------------------------------------------------------------
# Default pipeline registry
# ---------------------------------------------------------------------------


def default_pipelines() -> list[PipelineSpec]:
    """Return the standard set of six evaluation pipeline specifications.

    Pipelines
    ---------
    1. ``bm25_fixed``               — BM25, FixedSizeChunker(200/40), no rerank.
    2. ``dense_fixed``              — DenseRetriever, FixedSizeChunker(200/40), no rerank.
    3. ``hybrid_rrf_fixed``         — HybridRetriever(BM25+Dense, RRF), FixedSize(200/40).
    4. ``hybrid_rrf_fixed_rerank``  — same as above + CrossEncoderReranker.
    5. ``hybrid_rrf_sentwin``       — HybridRetriever(BM25+Dense, RRF),
                                     SentenceWindowChunker(3/2).
    6. ``bm25_rm3_fixed``           — BM25 wrapped in ExpandingRetriever (RM3),
                                     FixedSizeChunker(200/40).

    Note on model reuse
    -------------------
    sentence-transformers caches the model internally after the first load, so
    constructing multiple :class:`DenseRetriever` instances with the same
    ``model_name`` does NOT load the model multiple times at runtime.

    Returns:
        Ordered list of :class:`PipelineSpec` objects.

    """
    from lodestone.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    emb_model = settings.embedding_model_name

    # Chunkers (reused across specs where possible)
    fixed_chunker = FixedSizeChunker(chunk_size=200, overlap=40)
    sentwin_chunker = SentenceWindowChunker(window_size=3, stride=2)

    def make_bm25() -> BM25Retriever:
        return BM25Retriever(k1=1.5, b=0.75)

    def make_dense() -> DenseRetriever:
        return DenseRetriever(model_name=emb_model)

    def make_hybrid_rrf() -> HybridRetriever:
        return HybridRetriever(
            retrievers=[BM25Retriever(k1=1.5, b=0.75), DenseRetriever(model_name=emb_model)],
            strategy="rrf",
            rrf_k=settings.rrf_k,
        )

    return [
        PipelineSpec(
            name="bm25_fixed",
            chunker=fixed_chunker,
            retriever_factory=make_bm25,
            use_rerank=False,
            use_rm3=False,
        ),
        PipelineSpec(
            name="dense_fixed",
            chunker=fixed_chunker,
            retriever_factory=make_dense,
            use_rerank=False,
            use_rm3=False,
        ),
        PipelineSpec(
            name="hybrid_rrf_fixed",
            chunker=fixed_chunker,
            retriever_factory=make_hybrid_rrf,
            use_rerank=False,
            use_rm3=False,
        ),
        PipelineSpec(
            name="hybrid_rrf_fixed_rerank",
            chunker=fixed_chunker,
            retriever_factory=make_hybrid_rrf,
            use_rerank=True,
            use_rm3=False,
        ),
        PipelineSpec(
            name="hybrid_rrf_sentwin",
            chunker=sentwin_chunker,
            retriever_factory=make_hybrid_rrf,
            use_rerank=False,
            use_rm3=False,
        ),
        PipelineSpec(
            name="bm25_rm3_fixed",
            chunker=fixed_chunker,
            retriever_factory=make_bm25,
            use_rerank=False,
            use_rm3=True,
        ),
    ]


__all__ = ["build_chunks", "run_retrieval", "PipelineSpec", "default_pipelines"]
