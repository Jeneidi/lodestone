"""lodestone.chunking.strategies — concrete text-chunking implementations.

Two strategies are provided:

1. :class:`FixedSizeChunker` — word-based sliding windows with configurable
   overlap.  Fast, deterministic, and corpus-agnostic.

2. :class:`SentenceWindowChunker` — regex sentence splitting with sliding
   windows of sentences, providing more natural chunk boundaries.

Both classes satisfy the :class:`~lodestone.chunking.Chunker` Protocol.
Chunk IDs use the deterministic format ``f"{doc_id}::{chunker_name}::{index}"``.
"""

from __future__ import annotations

import logging
import re

from lodestone.schemas import Chunk, Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence-splitting helper (shared; avoids nltk dependency)
# ---------------------------------------------------------------------------

# Abbreviations that should NOT trigger a sentence boundary even when
# followed by a period and whitespace.
_ABBREVS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "fig",
        "no",
        "vol",
        "pp",
        "approx",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "st",
        "ave",
        "blvd",
        "u.s",
        "u.k",
        "e.g",
        "i.e",
    }
)

# Sentence boundary: one or more sentence-ending punctuation chars followed
# by optional closing quotes/brackets, then whitespace and an uppercase letter
# (or end-of-string).
_SENT_END_RE = re.compile(r"(?<=[.!?])[\"'\)\]]*\s+(?=[A-Z])|(?<=[.!?])[\"'\)\]]*\s*$")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a regex heuristic.

    The splitter handles common abbreviations (Dr., Mr., etc.) to avoid
    false sentence boundaries.  It does NOT require nltk.

    Limitations:
    - Single-letter abbreviations followed by a period are not distinguished
      from sentence endings (e.g. "A. Smith" may be split).
    - Ellipsis (``...``) is treated as a sentence boundary.

    Args:
        text: Raw text to split.

    Returns:
        A list of sentence strings, stripped of leading/trailing whitespace.
        Returns ``[]`` for empty *text*.

    """
    if not text.strip():
        return []

    # Tokenise on candidate boundaries; re-join if the preceding token looks
    # like a known abbreviation.
    tokens = _SENT_END_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for token in tokens:
        candidate = (buffer + " " + token).strip() if buffer else token.strip()
        # Check if the last "word" before the split is a known abbreviation
        last_word_match = re.search(r"\b([A-Za-z]+)\.?\s*$", buffer)
        if last_word_match and last_word_match.group(1).lower().rstrip(".") in _ABBREVS:
            # Merge: the dot was an abbreviation, not a sentence boundary
            buffer = candidate
        else:
            if buffer:
                sentences.append(buffer.strip())
            buffer = token.strip()
    if buffer:
        sentences.append(buffer.strip())

    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------


class FixedSizeChunker:
    """Word-based sliding-window chunker with configurable overlap.

    Tokenises ``doc.text`` by whitespace, then emits windows of ``chunk_size``
    words stepping forward by ``chunk_size - overlap`` words each iteration.

    Chunk IDs are ``f"{doc_id}::fixed::{index}"``.

    Args:
        chunk_size: Maximum number of words per chunk.  Defaults to 200.
        overlap:    Number of words shared between consecutive chunks.
                    Must be strictly less than ``chunk_size``.  Defaults to 40.

    Raises:
        ValueError: If ``overlap >= chunk_size``.

    Example::

        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(doc)

    """

    name: str = "fixed"

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into fixed-size word windows.

        Args:
            doc: Source document.

        Returns:
            Ordered list of :class:`~lodestone.schemas.Chunk` objects.
            Returns ``[]`` if ``doc.text`` is empty or contains only
            whitespace.

        """
        text = doc.text.strip()
        if not text:
            logger.debug("FixedSizeChunker: doc '%s' has empty text, returning []", doc.doc_id)
            return []

        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0

        while start < len(words):
            window = words[start : start + self.chunk_size]
            chunk_text = " ".join(window)
            chunk_id = f"{doc.doc_id}::fixed::{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    index=index,
                    metadata={
                        "word_start": start,
                        "word_end": start + len(window) - 1,
                        "chunker": self.name,
                    },
                )
            )
            index += 1
            start += step

        logger.debug(
            "FixedSizeChunker: doc '%s' -> %d chunks (size=%d, overlap=%d)",
            doc.doc_id,
            len(chunks),
            self.chunk_size,
            self.overlap,
        )
        return chunks


# ---------------------------------------------------------------------------
# SentenceWindowChunker
# ---------------------------------------------------------------------------


class SentenceWindowChunker:
    """Sliding-window chunker that groups sentences into chunks.

    Splits the document into sentences using a regex heuristic (no nltk
    required), then emits windows of ``window_size`` sentences stepping
    forward by ``stride`` sentences per iteration.

    Chunk IDs are ``f"{doc_id}::sentence_window::{index}"``.

    Args:
        window_size: Number of sentences per chunk.  Defaults to 3.
        stride:      Step size (in sentences) between consecutive windows.
                     Must be >= 1 and <= ``window_size``.  Defaults to 2.

    Raises:
        ValueError: If ``stride < 1`` or ``stride > window_size``.

    Example::

        chunker = SentenceWindowChunker(window_size=5, stride=3)
        chunks = chunker.chunk(doc)

    """

    name: str = "sentence_window"

    def __init__(self, window_size: int = 3, stride: int = 2) -> None:
        if stride < 1:
            raise ValueError(f"stride ({stride}) must be >= 1")
        if stride > window_size:
            raise ValueError(f"stride ({stride}) should be <= window_size ({window_size})")
        self.window_size = window_size
        self.stride = stride

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into sentence windows.

        Args:
            doc: Source document.

        Returns:
            Ordered list of :class:`~lodestone.schemas.Chunk` objects.
            Returns ``[]`` if ``doc.text`` is empty.  Documents with fewer
            sentences than ``window_size`` produce a single chunk containing
            all sentences.

        """
        text = doc.text.strip()
        if not text:
            logger.debug("SentenceWindowChunker: doc '%s' has empty text, returning []", doc.doc_id)
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # If there are fewer sentences than window_size, one chunk covers all
        chunks: list[Chunk] = []
        index = 0
        start = 0

        while start < len(sentences):
            window_sents = sentences[start : start + self.window_size]
            chunk_text = " ".join(window_sents)
            chunk_id = f"{doc.doc_id}::sentence_window::{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    index=index,
                    metadata={
                        "sent_start": start,
                        "sent_end": start + len(window_sents) - 1,
                        "num_sentences": len(window_sents),
                        "chunker": self.name,
                    },
                )
            )
            index += 1
            start += self.stride

        logger.debug(
            "SentenceWindowChunker: doc '%s' -> %d chunks (window=%d, stride=%d)",
            doc.doc_id,
            len(chunks),
            self.window_size,
            self.stride,
        )
        return chunks


__all__ = ["FixedSizeChunker", "SentenceWindowChunker"]
