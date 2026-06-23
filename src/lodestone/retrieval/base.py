"""Abstract base class for all Lodestone retrievers.

Contract
--------
Every concrete retriever MUST:

1. Set a class-level ``name`` attribute that uniquely identifies it
   (e.g. ``"bm25"``, ``"dense"``, ``"hybrid"``).  This name is stored in
   :attr:`ScoredChunk.retriever` and appears in evaluation reports.

2. Implement :meth:`index` to accept a list of :class:`~lodestone.schemas.Chunk`
   objects and build whatever internal data structure the retriever needs
   (an inverted index, a FAISS/numpy vector store, …).
   ``index`` may be called exactly once before any calls to ``search``.
   Calling ``search`` before ``index`` raises :exc:`RuntimeError`.

3. Implement :meth:`search` to return a *descending-score-sorted* list of
   :class:`~lodestone.schemas.ScoredChunk` objects of length ≤ ``k``.

Score semantics
---------------
- Scores are **higher-is-better** across all retrievers.
- Scores have no fixed range; they are retriever-specific.
  (BM25 scores are non-negative floats; cosine-similarity dense scores are
  in [-1, 1]; fused scores are in [0, 1] after normalisation.)
- Fusion/reranking components are responsible for normalising scores before
  combining them; individual retrievers need not normalise.

Thread safety
-------------
Retrievers are not required to be thread-safe.  The evaluation harness runs
queries sequentially unless otherwise noted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lodestone.schemas import Chunk, ScoredChunk


class Retriever(ABC):
    """Abstract retriever — the contract every retrieval backend must satisfy.

    Subclass this and implement :meth:`index` and :meth:`search`.

    Example usage::

        class MyRetriever(Retriever):
            name = "my_retriever"

            def index(self, chunks: list[Chunk]) -> None:
                self._store = {c.chunk_id: c for c in chunks}

            def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
                # ... scoring logic ...
                return sorted(results, key=lambda x: x.score, reverse=True)[:k]
    """

    #: Human-readable identifier; MUST be overridden in every concrete subclass.
    name: str = "retriever"

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build the retriever's internal index from ``chunks``.

        This method is called once, before any calls to :meth:`search`.
        Implementations are free to store the index in memory or on disk.

        Args:
            chunks: All chunks that should be searchable.  The list may be
                    large (tens of thousands of chunks for a full corpus).

        Returns:
            ``None``.  Side effects only — the index is stored internally.

        Raises:
            ValueError: If ``chunks`` is empty (optional but recommended).

        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Retrieve the ``k`` most relevant chunks for ``query``.

        Args:
            query: Natural-language query string.
            k:     Maximum number of results to return.  The returned list
                   may be shorter than ``k`` if the index contains fewer
                   than ``k`` chunks.

        Returns:
            A list of :class:`~lodestone.schemas.ScoredChunk` objects sorted
            by ``score`` in **descending** order (most relevant first).
            Each result has ``retriever`` set to ``self.name``.

        Raises:
            RuntimeError: If called before :meth:`index`.

        """
        ...
