"""lodestone.retrieval.dense — Dense bi-encoder retriever using sentence-transformers.

Architecture
------------
Chunks are encoded to a float32 numpy matrix during :meth:`index`.  Each
vector is L2-normalised so that cosine similarity reduces to a dot product::

    cosine(u, v) = u · v   (when ||u|| = ||v|| = 1)

At search time the query is encoded, normalised, and dotted against all
chunk vectors via a single matrix multiplication.  Top-k results are
selected with ``numpy.argpartition`` (O(n) expected) rather than a full
sort (O(n log n)), which is faster for large corpora.

Lazy imports
------------
``sentence_transformers`` is imported *inside* :meth:`index` and
:meth:`search` so that ``import lodestone.retrieval.dense`` succeeds even in
environments where sentence-transformers is not installed.  This keeps
module-level imports light and avoids triggering model downloads on import.

Injectable encoder
------------------
The constructor accepts an optional ``encoder`` callable
``Callable[[list[str]], np.ndarray]`` that overrides the SentenceTransformer
model.  This is intended for unit testing and extensibility (e.g. to swap in
a different embedding backend without subclassing).  When ``encoder`` is
provided, *model_name* is ignored entirely.

Example (production)::

    retriever = DenseRetriever()
    retriever.index(chunks)
    results = retriever.search("what is backpropagation?", k=5)

Example (testing)::

    def fake_encoder(texts):
        return np.random.randn(len(texts), 16).astype(np.float32)

    retriever = DenseRetriever(encoder=fake_encoder)
    retriever.index(chunks)
    results = retriever.search("test query", k=3)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from lodestone.retrieval.base import Retriever
from lodestone.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of *matrix* in-place and return it.

    Rows with zero norm are left as-is (no division by zero).

    Args:
        matrix: Float32 2-D array of shape ``(n, d)``.

    Returns:
        The normalised matrix (same object, modified in-place).

    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid dividing by zero for all-zero vectors
    norms = np.where(norms == 0.0, 1.0, norms)
    matrix /= norms
    return matrix


class DenseRetriever(Retriever):
    """Bi-encoder dense retriever backed by sentence-transformers and numpy.

    Encodes all chunks into a float32 numpy matrix (L2-normalised) during
    :meth:`index`, then answers queries with a single matrix-multiply at
    search time.

    Scoring formula::

        score(q, d) = normalise(encode(q)) · normalise(encode(d))ᵀ

    This equals the cosine similarity because both vectors are unit-length
    after L2 normalisation.

    Args:
        model_name:  HuggingFace model identifier for the bi-encoder.
                     Defaults to the value from :func:`~lodestone.config.get_settings`
                     (``"sentence-transformers/all-MiniLM-L6-v2"``).
                     Ignored when *encoder* is supplied.
        batch_size:  Number of texts to encode in a single forward pass.
                     Defaults to 64.
        encoder:     Optional callable ``(list[str]) -> np.ndarray`` that
                     replaces the SentenceTransformer model.  Intended for
                     testing and custom embedding backends.

    Attributes:
        name: ``"dense"``

    Raises:
        RuntimeError: If :meth:`search` is called before :meth:`index`.
        ValueError:   If :meth:`index` is called with an empty chunk list.

    Example::

        retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
        retriever.index(chunks)
        results = retriever.search("gradient descent", k=10)

    """

    name: str = "dense"

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 64,
        encoder: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self._model_name = model_name          # resolved lazily if None
        self.batch_size = batch_size
        self._encoder_override = encoder       # for testing / custom backends

        # Cached SentenceTransformer instance (loaded once on first encode call)
        self._st_model: object | None = None

        # Internal state populated by index()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None   # shape (n, d), float32, L2-normed
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model_name(self) -> str:
        """Return the resolved model name (falls back to settings)."""
        if self._model_name is not None:
            return self._model_name
        # Lazy import to avoid pulling in pydantic_settings at module load
        from lodestone.config import get_settings  # noqa: PLC0415

        return get_settings().embedding_model_name

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* to a float32 numpy matrix.

        Uses the injected ``encoder`` callable if available, otherwise
        lazily loads SentenceTransformer and encodes with *batch_size*.

        Args:
            texts: List of strings to encode.

        Returns:
            Float32 array of shape ``(len(texts), embedding_dim)``.

        """
        if self._encoder_override is not None:
            result = self._encoder_override(texts)
            return np.array(result, dtype=np.float32)

        # Lazy import — sentence_transformers not required at module level
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        if self._st_model is None:
            model_name = self._get_model_name()
            logger.info("DenseRetriever: loading model '%s'", model_name)
            self._st_model = SentenceTransformer(model_name)
        embeddings = self._st_model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Encode all *chunks* and store as a normalised numpy matrix.

        Args:
            chunks: All chunks to make searchable.

        Raises:
            ValueError: If *chunks* is empty.

        """
        if not chunks:
            raise ValueError("DenseRetriever.index() received an empty chunk list.")

        self._chunks = list(chunks)
        texts = [c.text for c in chunks]

        logger.info("DenseRetriever: encoding %d chunks …", len(texts))
        matrix = self._encode(texts)
        self._vectors = _l2_normalize(matrix)

        self._indexed = True
        logger.info(
            "DenseRetriever: indexed %d chunks; embedding dim=%d",
            len(chunks),
            self._vectors.shape[1],
        )

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Find the *k* most similar chunks to *query* via cosine similarity.

        Encodes the query, L2-normalises it, then computes cosine similarity
        with all indexed chunk vectors via a matrix multiply.  Top-k results
        are selected with ``numpy.argpartition`` for efficiency.

        Tie-breaking is by chunk_id (lexicographic ascending) to guarantee
        determinism.

        Args:
            query: Natural-language query string.
            k:     Maximum number of results to return.

        Returns:
            Up to *k* :class:`~lodestone.schemas.ScoredChunk` objects sorted
            by cosine similarity descending.

        Raises:
            RuntimeError: If called before :meth:`index`.

        """
        if not self._indexed or self._vectors is None:
            raise RuntimeError("DenseRetriever.search() called before index().")

        # Encode and normalise query
        q_vec = self._encode([query])            # shape (1, d)
        q_vec = _l2_normalize(q_vec)             # still (1, d)

        # Cosine similarities: (n,)
        scores = (self._vectors @ q_vec.T).squeeze(axis=1)   # shape (n,)

        n = len(self._chunks)
        actual_k = min(k, n)

        if actual_k == n:
            # Return all
            top_indices = np.arange(n)
        else:
            # argpartition gives the top-k in arbitrary order (O(n))
            top_indices = np.argpartition(scores, -actual_k)[-actual_k:]

        # Sort top_indices by (-score, chunk_id) for determinism
        top_indices_sorted = sorted(
            top_indices.tolist(),
            key=lambda i: (-float(scores[i]), self._chunks[i].chunk_id),
        )

        results: list[ScoredChunk] = []
        for idx in top_indices_sorted[:k]:
            results.append(
                ScoredChunk(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    retriever=self.name,
                )
            )
        return results


__all__ = ["DenseRetriever"]
