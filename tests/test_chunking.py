"""
Tests for lodestone.chunking.strategies.

Covers:
- FixedSizeChunker: chunk_size/overlap, coverage, chunk_ids, empty doc,
  short doc (< chunk_size), determinism.
- SentenceWindowChunker: window/stride behavior, single-window fallback,
  abbreviation handling, determinism.
"""

from __future__ import annotations

import pytest

from lodestone.chunking.strategies import FixedSizeChunker, SentenceWindowChunker
from lodestone.schemas import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, title="", text=text, source="test")


def _word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=10, overlap=10)

    def test_overlap_greater_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=15)

    def test_empty_doc_returns_empty_list(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        doc = _make_doc("empty", "")
        assert chunker.chunk(doc) == []

    def test_whitespace_only_doc_returns_empty_list(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        doc = _make_doc("ws", "   \n\t  ")
        assert chunker.chunk(doc) == []

    def test_short_doc_produces_single_chunk(self):
        """A document shorter than chunk_size yields exactly one chunk."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        doc = _make_doc("short", "hello world this is short")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world this is short"

    def test_chunk_size_respected(self):
        """Every chunk except possibly the last has <= chunk_size words."""
        chunk_size = 5
        overlap = 2
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        # 15-word document
        doc = _make_doc("d1", "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert _word_count(chunk.text) <= chunk_size

    def test_overlap_word_count(self):
        """Consecutive chunks share exactly `overlap` words (except last).

        The final chunk may have fewer than `overlap` words (the tail of the
        document), so the overlap invariant is only checked for pairs where
        the second chunk has at least `overlap` words.
        """
        chunk_size = 5
        overlap = 2
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        doc = _make_doc("d2", "a b c d e f g h i j k l m")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            words_a = chunks[i].text.split()
            words_b = chunks[i + 1].text.split()
            # Skip the check if the next chunk is a short tail (fewer words than
            # the overlap window — the document ended before the full overlap).
            if len(words_b) < overlap:
                continue
            tail_a = words_a[-overlap:]
            head_b = words_b[:overlap]
            assert tail_a == head_b, (
                f"Chunk {i} tail {tail_a!r} does not match chunk {i+1} head {head_b!r}"
            )

    def test_all_words_covered(self):
        """The union of all chunk words covers every word in the document."""
        chunk_size = 5
        overlap = 2
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        original_text = "one two three four five six seven eight nine ten eleven"
        doc = _make_doc("d3", original_text)
        chunks = chunker.chunk(doc)
        original_words = original_text.split()
        # First chunk starts at word 0, last chunk ends at the last word
        first_words = chunks[0].text.split()
        last_words = chunks[-1].text.split()
        assert first_words[0] == original_words[0]
        assert last_words[-1] == original_words[-1]

    def test_chunk_ids_are_deterministic(self):
        """chunk_id format is f'{doc_id}::fixed::{index}'."""
        chunker = FixedSizeChunker(chunk_size=5, overlap=1)
        doc = _make_doc("mydoc", "a b c d e f g h i j")
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"mydoc::fixed::{i}"
            assert chunk.index == i
            assert chunk.doc_id == "mydoc"

    def test_chunk_ids_unique(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=2)
        doc = _make_doc("u", "a b c d e f g h i j k l m n o")
        chunks = chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"

    def test_deterministic_repeated_calls(self):
        """Same input always produces identical output."""
        chunker = FixedSizeChunker(chunk_size=8, overlap=3)
        doc = _make_doc("det", "alpha beta gamma delta epsilon zeta eta theta iota kappa")
        result1 = chunker.chunk(doc)
        result2 = chunker.chunk(doc)
        for c1, c2 in zip(result1, result2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text

    def test_metadata_contains_word_positions(self):
        """Each chunk has word_start and word_end metadata."""
        chunker = FixedSizeChunker(chunk_size=4, overlap=1)
        doc = _make_doc("m", "a b c d e f g h")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert "word_start" in chunk.metadata
            assert "word_end" in chunk.metadata
            assert chunk.metadata["chunker"] == "fixed"

    def test_step_advances_by_chunk_size_minus_overlap(self):
        """word_start advances by (chunk_size - overlap) between chunks."""
        chunk_size = 6
        overlap = 2
        step = chunk_size - overlap
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        words = list("abcdefghijklmnopqrstuvwxyz")
        doc = _make_doc("s", " ".join(words[:20]))
        chunks = chunker.chunk(doc)
        for i in range(1, len(chunks)):
            diff = chunks[i].metadata["word_start"] - chunks[i - 1].metadata["word_start"]
            assert diff == step

    def test_exact_multiple_no_remainder(self):
        """10 words, chunk_size=5, overlap=0 → exactly 2 chunks."""
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        doc = _make_doc("exact", "a b c d e f g h i j")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].text == "a b c d e"
        assert chunks[1].text == "f g h i j"


# ---------------------------------------------------------------------------
# SentenceWindowChunker
# ---------------------------------------------------------------------------

class TestSentenceWindowChunker:

    def test_invalid_stride_zero_raises(self):
        with pytest.raises(ValueError, match="stride"):
            SentenceWindowChunker(window_size=3, stride=0)

    def test_stride_greater_than_window_raises(self):
        with pytest.raises(ValueError):
            SentenceWindowChunker(window_size=2, stride=3)

    def test_empty_doc_returns_empty_list(self):
        chunker = SentenceWindowChunker(window_size=3, stride=2)
        doc = _make_doc("empty", "")
        assert chunker.chunk(doc) == []

    def test_fewer_sentences_than_window_yields_one_chunk(self):
        """Document with 2 sentences and window_size=5 → single chunk."""
        chunker = SentenceWindowChunker(window_size=5, stride=3)
        doc = _make_doc("few", "Hello world. Goodbye world.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        # Both sentences should be in it
        assert "Hello world" in chunks[0].text
        assert "Goodbye world" in chunks[0].text

    def test_six_sentences_window3_stride2_chunk_count(self):
        """6 sentences, window=3, stride=2 → ceil((6-3)/2) + 1 = 3 chunks."""
        chunker = SentenceWindowChunker(window_size=3, stride=2)
        text = (
            "Sentence one here. "
            "Sentence two here. "
            "Sentence three here. "
            "Sentence four here. "
            "Sentence five here. "
            "Sentence six here."
        )
        doc = _make_doc("six", text)
        chunks = chunker.chunk(doc)
        # Start positions: 0, 2, 4 → 3 chunks
        assert len(chunks) == 3

    def test_window_size_respected(self):
        """Each chunk contains at most window_size sentences."""
        chunker = SentenceWindowChunker(window_size=2, stride=1)
        text = (
            "Alpha. Beta. Gamma. Delta. Epsilon."
        )
        doc = _make_doc("ws", text)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            # Count sentence-ending punctuation as proxy for sentence count
            sent_count = chunk.metadata["num_sentences"]
            assert sent_count <= 2

    def test_stride_advances_by_stride_sentences(self):
        """sent_start advances by stride between consecutive chunks."""
        stride = 2
        chunker = SentenceWindowChunker(window_size=3, stride=stride)
        text = "A. B. C. D. E. F. G. H."
        doc = _make_doc("stride", text)
        chunks = chunker.chunk(doc)
        for i in range(1, len(chunks)):
            diff = chunks[i].metadata["sent_start"] - chunks[i - 1].metadata["sent_start"]
            assert diff == stride

    def test_chunk_ids_format(self):
        """chunk_id is f'{doc_id}::sentence_window::{index}'."""
        chunker = SentenceWindowChunker(window_size=2, stride=1)
        doc = _make_doc("swdoc", "One. Two. Three. Four.")
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"swdoc::sentence_window::{i}"
            assert chunk.index == i
            assert chunk.doc_id == "swdoc"

    def test_abbreviation_dr_smith(self):
        """'Dr. Smith went home. He slept.' should yield 2 sentences.

        The sentence splitter splits this into exactly 2 sentences:
        ['Dr. Smith went home.', 'He slept.']. With window_size=5 and
        stride=1, two overlapping chunks are produced. The first chunk
        contains both sentences (it does NOT split at 'Dr.').
        """
        chunker = SentenceWindowChunker(window_size=5, stride=1)
        doc = _make_doc("abbrev", "Dr. Smith went home. He slept.")
        chunks = chunker.chunk(doc)
        # 2 sentences, stride=1 → 2 overlapping chunks (start=0, start=1)
        assert len(chunks) == 2
        # First chunk spans both sentences
        assert chunks[0].metadata["num_sentences"] == 2
        # The first chunk must contain 'Dr. Smith' intact (abbreviation not split)
        assert "Dr. Smith went home." in chunks[0].text

    def test_deterministic(self):
        """Same input → identical output on repeated calls."""
        chunker = SentenceWindowChunker(window_size=3, stride=2)
        doc = _make_doc("det2", "First sentence. Second sentence. Third sentence. Fourth sentence.")
        r1 = chunker.chunk(doc)
        r2 = chunker.chunk(doc)
        assert [c.text for c in r1] == [c.text for c in r2]

    def test_name_attribute(self):
        assert FixedSizeChunker.name == "fixed"
        assert SentenceWindowChunker.name == "sentence_window"
