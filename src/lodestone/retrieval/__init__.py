"""lodestone.retrieval — retrieval backends package.

Public surface
--------------
Abstract base:

- :class:`~lodestone.retrieval.base.Retriever` — ABC every retrieval backend
  must implement (``index``, ``search``).

Concrete retrievers:

- :class:`~lodestone.retrieval.bm25.BM25Retriever` — sparse Okapi BM25 with a
  hand-built inverted index.
- :class:`~lodestone.retrieval.dense.DenseRetriever` — dense bi-encoder
  retrieval backed by sentence-transformers + numpy cosine similarity.
- :class:`~lodestone.retrieval.fusion.HybridRetriever` — fuses multiple
  retrievers via RRF or weighted score fusion.
- :class:`~lodestone.retrieval.expansion.ExpandingRetriever` — RM3-style
  pseudo-relevance feedback wrapper around any retriever.

Utilities:

- :class:`~lodestone.retrieval.expansion.Rm3QueryExpander` — standalone query
  expander (can be used independently of :class:`ExpandingRetriever`).
- :class:`~lodestone.retrieval.rerank.CrossEncoderReranker` — cross-encoder
  reranker for second-stage re-scoring.

Convenience factory:

- :func:`build_default_pipeline` — returns a ready :class:`HybridRetriever`
  (BM25 + Dense, RRF) pre-loaded with the provided chunks.

Tokenisation (internal; exported for testing):

- :mod:`lodestone.retrieval.tokenize` — normalise / tokenise / stopwords /
  Porter-lite stemmer.
"""

from __future__ import annotations

from lodestone.retrieval.base import Retriever
from lodestone.retrieval.bm25 import BM25Retriever
from lodestone.retrieval.dense import DenseRetriever
from lodestone.retrieval.expansion import ExpandingRetriever, Rm3QueryExpander
from lodestone.retrieval.fusion import HybridRetriever
from lodestone.retrieval.rerank import CrossEncoderReranker
from lodestone.schemas import Chunk


def build_default_pipeline(
    chunks: list[Chunk],
    settings=None,
) -> HybridRetriever:
    """Build and index a default BM25 + Dense hybrid retrieval pipeline.

    Creates a :class:`BM25Retriever` and a :class:`DenseRetriever`, wraps
    them in a :class:`HybridRetriever` using Reciprocal Rank Fusion (RRF),
    calls ``index(chunks)`` on the hybrid (which delegates to both children),
    and returns the ready-to-use retriever.

    This is the recommended starting point for new Lodestone deployments.
    For custom configurations (different models, fusion weights, expansion)
    construct the components directly.

    Args:
        chunks:   All :class:`~lodestone.schemas.Chunk` objects to index.
                  Must be non-empty.
        settings: Optional :class:`~lodestone.config.Settings` instance.
                  If ``None``, :func:`~lodestone.config.get_settings` is used.
                  Provide an explicit instance to override model names or
                  hyper-parameters without touching environment variables.

    Returns:
        A :class:`HybridRetriever` (strategy ``"rrf"``) that is already
        indexed and ready to call :meth:`~HybridRetriever.search` on.

    Raises:
        ValueError: If *chunks* is empty.

    Example::

        from lodestone.retrieval import build_default_pipeline

        pipeline = build_default_pipeline(chunks)
        results = pipeline.search("what is backpropagation?", k=10)

    """
    if not chunks:
        raise ValueError("build_default_pipeline: chunks list must not be empty.")

    if settings is None:
        from lodestone.config import get_settings  # noqa: PLC0415

        settings = get_settings()

    bm25 = BM25Retriever(k1=1.5, b=0.75)
    dense = DenseRetriever(model_name=settings.embedding_model_name)

    hybrid = HybridRetriever(
        retrievers=[bm25, dense],
        strategy="rrf",
        rrf_k=settings.rrf_k,
    )
    hybrid.index(chunks)
    return hybrid


__all__ = [
    "Retriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "Rm3QueryExpander",
    "ExpandingRetriever",
    "CrossEncoderReranker",
    "build_default_pipeline",
]
