"""lodestone.retrieval.bm25 — BM25 sparse retriever with a hand-built inverted index.

Scoring formula (Okapi BM25)
----------------------------
For a query ``q`` composed of terms ``t_1, ..., t_n`` and a document ``d``::

    score(d, q) = sum_{t in q} IDF(t) * f(t, d) * (k1 + 1)
                                        --------------------------
                                        f(t, d) + k1 * (1 - b + b * |d| / avgdl)

where:

- ``f(t, d)``  = term frequency of ``t`` in document ``d``
- ``|d|``      = length of ``d`` in tokens
- ``avgdl``    = average document length across the corpus
- ``k1``       = term-frequency saturation parameter (default 1.5)
- ``b``        = length normalisation parameter (default 0.75)
- ``IDF(t)``   = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  - ``N``      = total number of documents (chunks) in the corpus
  - ``df(t)``  = number of documents containing term ``t``
  - The ``+ 1`` outside the log keeps IDF non-negative even when df = N.

The inverted index maps each term to a list of ``(chunk_index, term_freq)``
pairs so that only candidate chunks containing at least one query term are
scored — the full corpus is never iterated.

References
----------
Robertson, S. & Zaragoza, H. (2009).  The Probabilistic Relevance Framework:
BM25 and Beyond.  Foundations and Trends in Information Retrieval, 3(4), 333–389.

"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

from lodestone.retrieval.base import Retriever
from lodestone.retrieval.tokenize import tokenize
from lodestone.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """Okapi BM25 retriever backed by a from-scratch inverted index.

    Scoring formula (see module docstring for full derivation)::

        score(d, q) = Σ_t  IDF(t) · TF_norm(t, d)

        TF_norm(t, d) = f(t,d) · (k1 + 1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

    Args:
        k1: Term-frequency saturation parameter.  Controls how quickly
            additional occurrences of a term contribute diminishing returns.
            Typical values: 1.2–2.0.  Defaults to 1.5.
        b:  Length normalisation strength.  0.0 disables length normalisation;
            1.0 fully normalises by document length.  Defaults to 0.75.

    Attributes:
        name: ``"bm25"``

    Raises:
        RuntimeError: If :meth:`search` is called before :meth:`index`.
        ValueError:   If :meth:`index` is called with an empty chunk list.

    Example::

        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(chunks)
        results = retriever.search("machine learning", k=5)

    """

    name: str = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # Internal state — populated by index()
        self._chunks: list[Chunk] = []
        self._doc_lengths: list[int] = []      # token count per chunk
        self._avgdl: float = 0.0
        # term -> list of (chunk_index, term_freq)
        self._inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Build an inverted index from *chunks*.

        Tokenises each chunk (stopwords removed, no stemming by default for
        better exact-match recall), records term frequencies and document
        lengths, then pre-computes IDF for every term.

        Args:
            chunks: All chunks to make searchable.

        Raises:
            ValueError: If *chunks* is empty.

        """
        if not chunks:
            raise ValueError("BM25Retriever.index() received an empty chunk list.")

        self._chunks = list(chunks)
        n = len(chunks)

        # Build forward index (term frequencies per chunk)
        forward: list[Counter[str]] = []
        for chunk in chunks:
            tf: Counter[str] = Counter(tokenize(chunk.text, remove_stops=True, stem=False))
            forward.append(tf)
            self._doc_lengths.append(sum(tf.values()))

        self._avgdl = sum(self._doc_lengths) / n if n > 0 else 1.0

        # Build inverted index
        self._inverted_index = defaultdict(list)
        df: Counter[str] = Counter()
        for chunk_idx, tf in enumerate(forward):
            for term, freq in tf.items():
                self._inverted_index[term].append((chunk_idx, freq))
                df[term] += 1

        # Pre-compute IDF for every term in the vocabulary
        # IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        self._idf = {}
        for term, doc_freq in df.items():
            self._idf[term] = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        self._indexed = True
        logger.info(
            "BM25Retriever indexed %d chunks; vocab size=%d; avgdl=%.1f",
            n,
            len(self._idf),
            self._avgdl,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Retrieve the *k* most relevant chunks for *query* using BM25.

        Only chunks containing at least one query term are scored (sparse
        retrieval via the inverted index).  Ties are broken by chunk_id
        (lexicographic, ascending) to guarantee a stable, deterministic order.

        Args:
            query: Natural-language query string.
            k:     Maximum number of results.

        Returns:
            Up to *k* :class:`~lodestone.schemas.ScoredChunk` objects sorted
            by BM25 score descending.

        Raises:
            RuntimeError: If called before :meth:`index`.

        """
        if not self._indexed:
            raise RuntimeError("BM25Retriever.search() called before index().")

        query_terms = tokenize(query, remove_stops=True, stem=False)
        if not query_terms:
            logger.debug("BM25Retriever: empty query after tokenisation, returning []")
            return []

        # Accumulate scores for candidate chunks only
        scores: dict[int, float] = {}
        for term in query_terms:
            if term not in self._inverted_index:
                continue
            idf = self._idf.get(term, 0.0)
            for chunk_idx, tf in self._inverted_index[term]:
                dl = self._doc_lengths[chunk_idx]
                # BM25 term score
                tf_norm = tf * (self.k1 + 1.0) / (
                    tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                )
                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + idf * tf_norm

        if not scores:
            return []

        # Sort: descending score, then ascending chunk_id for determinism
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._chunks[item[0]].chunk_id),
        )

        results: list[ScoredChunk] = []
        for chunk_idx, score in ranked[:k]:
            results.append(
                ScoredChunk(
                    chunk=self._chunks[chunk_idx],
                    score=score,
                    retriever=self.name,
                )
            )
        return results


__all__ = ["BM25Retriever"]
