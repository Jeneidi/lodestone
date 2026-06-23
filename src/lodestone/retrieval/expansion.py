"""lodestone.retrieval.expansion — RM3-style pseudo-relevance feedback query expansion.

Algorithm (RM3-inspired)
------------------------
Given a query ``q``:

1. Retrieve the top ``fb_docs`` chunks using the inner retriever.
2. Collect all tokens from those chunks (using :mod:`~lodestone.retrieval.tokenize`,
   stopwords removed).
3. Compute a term frequency distribution over the pseudo-relevant set.
4. Select the top ``fb_terms`` terms by frequency that are NOT already
   present in the tokenised query.
5. Return the expanded query::

       q_expanded = q + " " + " ".join(expansion_terms)

6. Interpolate with original query weight ``orig_weight``::

       The expanded query string simply appends the new terms; the
       retriever's own scoring function handles term weighting.
       ``orig_weight`` is reserved for future scored-vector expansion
       but currently controls whether the original query terms are
       repeated (values < 1.0 are a no-op in the string representation).

Note: This is a *lexical* approximation of RM3; it does not perform the
full probability-model interpolation.  For dense RM3 see the literature.

References
----------
Lavrenko, V. & Croft, W. B. (2001). Relevance-Based Language Models.
SIGIR 2001, pp. 120–127.

Abdul-Jaleel, N. et al. (2004). UMASS at TREC 2004: Novelty and HARD.
TREC 2004 Proceedings.

"""

from __future__ import annotations

import logging
from collections import Counter

from lodestone.retrieval.base import Retriever
from lodestone.retrieval.tokenize import tokenize
from lodestone.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class Rm3QueryExpander:
    """RM3-style pseudo-relevance feedback query expander.

    Uses a retriever to fetch pseudo-relevant chunks, then selects
    high-frequency expansion terms not already in the query.

    Args:
        retriever:   Retriever used to fetch pseudo-relevant documents.
                     Must already be indexed.
        fb_docs:     Number of top-ranked chunks to use as pseudo-relevant
                     feedback documents.  Defaults to 5.
        fb_terms:    Number of expansion terms to add to the query.
                     Defaults to 8.
        orig_weight: Conceptual weight of the original query terms (0..1).
                     At 1.0 only expansion terms are appended (original query
                     unchanged); at values < 1.0 the original terms are
                     appended again proportionally — currently informational,
                     reserved for future probability-weighted expansion.
                     Defaults to 0.6.

    Example::

        bm25 = BM25Retriever()
        bm25.index(chunks)
        expander = Rm3QueryExpander(bm25, fb_docs=5, fb_terms=8)
        expanded = expander.expand("machine learning optimization")
        # expanded might be:
        # "machine learning optimization gradient descent neural network ..."

    """

    def __init__(
        self,
        retriever: Retriever,
        fb_docs: int = 5,
        fb_terms: int = 8,
        orig_weight: float = 0.6,
    ) -> None:
        self._retriever = retriever
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.orig_weight = orig_weight

    def expand(self, query: str) -> str:
        """Expand *query* with pseudo-relevance feedback terms.

        Pipeline:

        1. Tokenise *query* to get the current query term set.
        2. Retrieve the top ``fb_docs`` chunks.
        3. Build a term-frequency distribution over the retrieved chunks
           (stopwords removed via :func:`~lodestone.retrieval.tokenize.tokenize`).
        4. Select the ``fb_terms`` highest-frequency terms not already in
           the query.
        5. Append them to the original query string.

        Args:
            query: Original natural-language query.

        Returns:
            Expanded query string (original + expansion terms separated by
            spaces).  Returns *query* unchanged if no expansion terms are
            found.

        """
        # Step 1: tokenise original query to identify existing terms
        query_terms: set[str] = set(tokenize(query, remove_stops=True, stem=False))

        # Step 2: retrieve pseudo-relevant chunks
        try:
            pseudo_relevant: list[ScoredChunk] = self._retriever.search(
                query, k=self.fb_docs
            )
        except Exception as exc:
            logger.warning("Rm3QueryExpander: retrieval failed — %s; returning original query", exc)
            return query

        if not pseudo_relevant:
            logger.debug(
                "Rm3QueryExpander: no pseudo-relevant docs found; returning original query"
            )
            return query

        # Step 3: build term-frequency distribution over pseudo-relevant set
        tf: Counter[str] = Counter()
        for sc in pseudo_relevant:
            tokens = tokenize(sc.chunk.text, remove_stops=True, stem=False)
            tf.update(tokens)

        # Step 4: select top fb_terms not already in the query
        expansion_terms: list[str] = []
        for term, _ in tf.most_common():
            if term not in query_terms:
                expansion_terms.append(term)
            if len(expansion_terms) >= self.fb_terms:
                break

        if not expansion_terms:
            logger.debug("Rm3QueryExpander: no novel expansion terms found")
            return query

        # Step 5: return expanded query
        expanded = query + " " + " ".join(expansion_terms)
        logger.debug(
            "Rm3QueryExpander: expanded query with %d terms: %r",
            len(expansion_terms),
            expansion_terms,
        )
        return expanded


class ExpandingRetriever(Retriever):
    """Retriever wrapper that applies RM3 query expansion before searching.

    Delegates :meth:`index` and :meth:`search` to the *inner* retriever,
    expanding the query with *expander* before each search.

    Args:
        inner:   The underlying retriever (must be indexed before searching).
        expander: An :class:`Rm3QueryExpander` (or any object with an
                  ``expand(query: str) -> str`` method).

    Attributes:
        name: ``f"rm3+{inner.name}"``

    Example::

        bm25 = BM25Retriever()
        bm25.index(chunks)
        expander = Rm3QueryExpander(bm25, fb_docs=5, fb_terms=8)
        rm3_retriever = ExpandingRetriever(bm25, expander)
        results = rm3_retriever.search("transformer architecture", k=10)

    """

    def __init__(self, inner: Retriever, expander: Rm3QueryExpander) -> None:
        self._inner = inner
        self._expander = expander

    @property
    def name(self) -> str:  # type: ignore[override]
        """Name combines ``"rm3"`` prefix with the inner retriever name."""
        return f"rm3+{self._inner.name}"

    def index(self, chunks: list[Chunk]) -> None:
        """Delegate indexing to the inner retriever.

        Args:
            chunks: All chunks to index.

        """
        self._inner.index(chunks)

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Expand *query* then delegate to the inner retriever.

        The expansion uses the same inner retriever for pseudo-relevance
        feedback (internally calling its :meth:`search`).  Results are
        tagged with ``retriever=self.name``.

        Args:
            query: Original query string.
            k:     Maximum number of results.

        Returns:
            Up to *k* :class:`~lodestone.schemas.ScoredChunk` objects from
            the inner retriever, with ``retriever`` set to ``self.name``.

        """
        expanded = self._expander.expand(query)
        logger.debug("ExpandingRetriever: original=%r  expanded=%r", query, expanded)

        results = self._inner.search(expanded, k=k)
        # Re-tag retriever name
        return [
            ScoredChunk(chunk=sc.chunk, score=sc.score, retriever=self.name)
            for sc in results
        ]


__all__ = ["Rm3QueryExpander", "ExpandingRetriever"]
