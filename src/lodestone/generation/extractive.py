"""lodestone.generation.extractive — free, LLM-free extractive answer generation.

Algorithm
---------
1. Take the top-scored chunks from ``chunks`` (ordered by descending score).
2. Split each chunk's text into sentences using a regex-based splitter.
3. Score every sentence by the Jaccard overlap of its lowercased word tokens
   with the query's word tokens.
4. Rank sentences by score descending; break ties by (chunk_rank, sentence_index)
   for deterministic output.
5. Return the top ``max_sentences`` sentences joined by a space as the answer
   text, stored in an :class:`~lodestone.schemas.Answer` with
   ``generator="extractive"``.

Tokenisation
------------
A tiny local tokeniser is used (``_tokenize``) — no external library required.
It lower-cases the input, strips punctuation, and splits on whitespace.

If ``lodestone.retrieval.tokenize`` is added in a later wave it can be wired
in as a drop-in replacement without changing the public API.
"""

from __future__ import annotations

import logging
import re
import time

from lodestone.schemas import Answer, ScoredChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiny local tokeniser (no external dependency)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split *text* into word tokens.

    Args:
        text: Input string.

    Returns:
        List of non-empty lowercase word tokens.

    """
    lowered = text.lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    tokens = _WHITESPACE_RE.split(no_punct.strip())
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Regex that splits on sentence-ending punctuation followed by whitespace.
# Handles common abbreviations imperfectly — good enough for extractive QA.
_SENT_SPLIT_RE = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into a list of sentences.

    Uses a simple regex-based splitter that handles most English prose.
    Returns the stripped parts; empty parts are discarded.

    Args:
        text: Input paragraph or multi-sentence string.

    Returns:
        List of sentence strings.

    """
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _jaccard_overlap(query_tokens: frozenset[str], sentence: str) -> float:
    """Compute Jaccard similarity between *query_tokens* and sentence tokens.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|.  Returns 0.0 when both sets are empty.

    Args:
        query_tokens: Pre-tokenised, frozen set of query word tokens.
        sentence:     Candidate sentence string to score.

    Returns:
        Jaccard similarity in [0.0, 1.0].

    """
    if not sentence:
        return 0.0
    sent_tokens = frozenset(_tokenize(sentence))
    if not query_tokens and not sent_tokens:
        return 0.0
    intersection = len(query_tokens & sent_tokens)
    union = len(query_tokens | sent_tokens)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ExtractiveAnswerer:
    """Extractive answer generator — no model call, no network access.

    Selects the most query-relevant sentences from the top retrieved chunks
    using lexical (Jaccard) overlap scoring.  Fully deterministic.

    Args:
        max_sentences: Number of top-scoring sentences to include in the
                       answer.  Defaults to 2.

    Example::

        from lodestone.generation.extractive import ExtractiveAnswerer
        from lodestone.schemas import Chunk, ScoredChunk

        chunks = [
            ScoredChunk(
                chunk=Chunk(chunk_id="c0", doc_id="d0", text="Paris is the capital of France."),
                score=0.9,
                retriever="bm25",
            )
        ]
        answerer = ExtractiveAnswerer(max_sentences=1)
        ans = answerer.answer("What is the capital of France?", chunks)
        print(ans.text)   # "Paris is the capital of France."

    """

    def __init__(self, max_sentences: int = 2) -> None:
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}.")
        self.max_sentences = max_sentences

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, query: str, chunks: list[ScoredChunk]) -> Answer:
        """Generate an extractive answer for *query* from *chunks*.

        Steps:
        1. Collect all sentences from the provided chunks, preserving
           (chunk_rank, sentence_index) for deterministic tie-breaking.
        2. Score each sentence via Jaccard overlap with the query tokens.
        3. Return the top ``max_sentences`` sentences joined by a space.

        Args:
            query:  The natural-language question.
            chunks: Scored chunks from the retriever (order preserved;
                    higher-scored chunks are preferred in tie-breaking).

        Returns:
            An :class:`~lodestone.schemas.Answer` with:
            - ``text``: top sentences joined by " ".
            - ``supporting_chunks``: the input *chunks* unchanged.
            - ``generator``: ``"extractive"``.
            - ``faithfulness``: ``None`` (not computed here).
            - ``latency_ms``: wall-clock time for this call in ms.

        Notes:
            If *chunks* is empty or all sentences score 0.0, returns an
            empty string answer rather than raising.

        """
        t_start = time.perf_counter()

        if not chunks:
            logger.debug("ExtractiveAnswerer: no chunks provided.")
            return Answer(
                text="",
                supporting_chunks=[],
                generator="extractive",
                faithfulness=None,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        query_tokens = frozenset(_tokenize(query))

        # (score, chunk_rank, sentence_index, sentence) for every sentence in
        # every chunk; chunk_rank/sentence_index keep ties deterministic.
        candidates: list[tuple[float, int, int, str]] = [
            (_jaccard_overlap(query_tokens, sentence), chunk_rank, sentence_index, sentence)
            for chunk_rank, scored_chunk in enumerate(chunks)
            for sentence_index, sentence in enumerate(_split_sentences(scored_chunk.chunk.text))
        ]
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        top_sentences = [c[3] for c in candidates[: self.max_sentences]]
        answer_text = " ".join(top_sentences)

        latency = (time.perf_counter() - t_start) * 1000.0
        logger.debug(
            "ExtractiveAnswerer: selected %d sentence(s) in %.2f ms.",
            len(top_sentences),
            latency,
        )

        return Answer(
            text=answer_text,
            supporting_chunks=list(chunks),
            generator="extractive",
            faithfulness=None,
            latency_ms=latency,
        )


__all__ = ["ExtractiveAnswerer"]
