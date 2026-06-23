"""
Tests for lodestone.data — load_corpus and load_qa.

Coverage:
- Round-trip: write corpus.jsonl / qa.jsonl with model_dump_json lines,
  load back with explicit path, assert equality.
- Missing file raises FileNotFoundError whose message mentions "make data".
- Passing an explicit file path (not just directory) also works.
- Empty-line resilience: blank lines in JSONL are skipped gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lodestone.data import load_corpus, load_qa
from lodestone.schemas import Document, QAExample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, objects: list) -> None:
    """Write a list of Pydantic models to *path* as JSONL (one JSON object per line)."""
    with path.open("w", encoding="utf-8") as fh:
        for obj in objects:
            fh.write(obj.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            doc_id="doc_001",
            title="Introduction to Transformers",
            text="Transformers revolutionised natural language processing with attention.",
            source="test",
        ),
        Document(
            doc_id="doc_002",
            title="Gradient Descent",
            text="Gradient descent optimises the loss function iteratively.",
            source="test",
            metadata={"split": "train"},
        ),
        Document(
            doc_id="doc_003",
            title="Quantum Physics",
            text="Quantum mechanics describes subatomic particle behaviour.",
            source="test",
        ),
    ]


@pytest.fixture
def sample_qa_examples() -> list[QAExample]:
    return [
        QAExample(
            qid="q001",
            question="What revolutionised NLP?",
            answer="Transformers revolutionised NLP.",
            relevant_doc_ids=["doc_001"],
        ),
        QAExample(
            qid="q002",
            question="What does gradient descent do?",
            answer="It optimises the loss function.",
            relevant_doc_ids=["doc_002"],
        ),
    ]


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestLoadCorpus:

    def test_roundtrip_explicit_file_path(self, tmp_path: Path, sample_documents: list[Document]):
        """Write JSONL → load with explicit file path → assert exact equality."""
        corpus_file = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_file, sample_documents)

        loaded = load_corpus(path=corpus_file)

        assert len(loaded) == len(sample_documents)
        for original, result in zip(sample_documents, loaded):
            assert result.doc_id == original.doc_id
            assert result.title == original.title
            assert result.text == original.text
            assert result.source == original.source
            assert result.metadata == original.metadata

    def test_roundtrip_directory_path(self, tmp_path: Path, sample_documents: list[Document]):
        """When path is a directory, load_corpus appends 'corpus.jsonl' automatically."""
        corpus_file = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_file, sample_documents)

        loaded = load_corpus(path=tmp_path)

        assert len(loaded) == len(sample_documents)
        assert [d.doc_id for d in loaded] == [d.doc_id for d in sample_documents]

    def test_preserves_order(self, tmp_path: Path, sample_documents: list[Document]):
        """Documents must be returned in file order."""
        corpus_file = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_file, sample_documents)

        loaded = load_corpus(path=corpus_file)
        assert [d.doc_id for d in loaded] == [d.doc_id for d in sample_documents]

    def test_preserves_metadata(self, tmp_path: Path):
        """Non-empty metadata dicts must survive the round-trip."""
        doc = Document(
            doc_id="meta_doc",
            title="Meta Test",
            text="Testing metadata preservation.",
            source="unit-test",
            metadata={"language": "en", "split": "test", "count": 42},
        )
        corpus_file = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_file, [doc])

        loaded = load_corpus(path=corpus_file)
        assert loaded[0].metadata == doc.metadata

    def test_blank_lines_skipped(self, tmp_path: Path, sample_documents: list[Document]):
        """Blank lines in the JSONL file must be silently skipped."""
        corpus_file = tmp_path / "corpus.jsonl"
        with corpus_file.open("w", encoding="utf-8") as fh:
            for i, doc in enumerate(sample_documents):
                fh.write(doc.model_dump_json() + "\n")
                if i == 0:
                    fh.write("\n")  # blank line after first doc

        loaded = load_corpus(path=corpus_file)
        assert len(loaded) == len(sample_documents)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        """Non-existent corpus file → FileNotFoundError."""
        missing = tmp_path / "does_not_exist" / "corpus.jsonl"
        with pytest.raises(FileNotFoundError):
            load_corpus(path=missing)

    def test_missing_file_error_mentions_make_data(self, tmp_path: Path):
        """The FileNotFoundError message must mention 'make data'."""
        missing = tmp_path / "corpus.jsonl"  # file not created
        with pytest.raises(FileNotFoundError, match="make data"):
            load_corpus(path=missing)


class TestLoadQA:

    def test_roundtrip_explicit_file_path(
        self, tmp_path: Path, sample_qa_examples: list[QAExample]
    ):
        """Write JSONL → load with explicit file path → assert exact equality."""
        qa_file = tmp_path / "qa.jsonl"
        _write_jsonl(qa_file, sample_qa_examples)

        loaded = load_qa(path=qa_file)

        assert len(loaded) == len(sample_qa_examples)
        for original, result in zip(sample_qa_examples, loaded):
            assert result.qid == original.qid
            assert result.question == original.question
            assert result.answer == original.answer
            assert result.relevant_doc_ids == original.relevant_doc_ids

    def test_roundtrip_directory_path(
        self, tmp_path: Path, sample_qa_examples: list[QAExample]
    ):
        """When path is a directory, load_qa appends 'qa.jsonl' automatically."""
        qa_file = tmp_path / "qa.jsonl"
        _write_jsonl(qa_file, sample_qa_examples)

        loaded = load_qa(path=tmp_path)
        assert len(loaded) == len(sample_qa_examples)

    def test_preserves_order(self, tmp_path: Path, sample_qa_examples: list[QAExample]):
        qa_file = tmp_path / "qa.jsonl"
        _write_jsonl(qa_file, sample_qa_examples)

        loaded = load_qa(path=qa_file)
        assert [q.qid for q in loaded] == [q.qid for q in sample_qa_examples]

    def test_empty_relevant_doc_ids(self, tmp_path: Path):
        """QAExample with no relevant_doc_ids round-trips correctly."""
        example = QAExample(
            qid="q_empty",
            question="Who is buried in Grant's tomb?",
            answer="Grant.",
            relevant_doc_ids=[],
        )
        qa_file = tmp_path / "qa.jsonl"
        _write_jsonl(qa_file, [example])

        loaded = load_qa(path=qa_file)
        assert loaded[0].relevant_doc_ids == []

    def test_multiple_relevant_doc_ids(self, tmp_path: Path):
        """QAExample with multiple relevant_doc_ids round-trips correctly."""
        example = QAExample(
            qid="q_multi",
            question="What is ML?",
            answer="Machine learning.",
            relevant_doc_ids=["d1", "d2", "d3"],
        )
        qa_file = tmp_path / "qa.jsonl"
        _write_jsonl(qa_file, [example])

        loaded = load_qa(path=qa_file)
        assert loaded[0].relevant_doc_ids == ["d1", "d2", "d3"]

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        """Non-existent QA file → FileNotFoundError."""
        missing = tmp_path / "qa.jsonl"  # file not created
        with pytest.raises(FileNotFoundError):
            load_qa(path=missing)

    def test_missing_file_error_mentions_make_data(self, tmp_path: Path):
        """The FileNotFoundError message must mention 'make data'."""
        missing = tmp_path / "qa.jsonl"
        with pytest.raises(FileNotFoundError, match="make data"):
            load_qa(path=missing)

    def test_blank_lines_skipped(self, tmp_path: Path, sample_qa_examples: list[QAExample]):
        """Blank lines in the JSONL file must be silently skipped."""
        qa_file = tmp_path / "qa.jsonl"
        with qa_file.open("w", encoding="utf-8") as fh:
            fh.write("\n")  # leading blank line
            for q in sample_qa_examples:
                fh.write(q.model_dump_json() + "\n")
                fh.write("\n")  # blank line between records

        loaded = load_qa(path=qa_file)
        assert len(loaded) == len(sample_qa_examples)


# ---------------------------------------------------------------------------
# Combined round-trip (corpus + qa together)
# ---------------------------------------------------------------------------

class TestCombinedRoundTrip:
    """Write both files into the same directory, load both via directory path."""

    def test_both_files_round_trip_from_same_dir(
        self,
        tmp_path: Path,
        sample_documents: list[Document],
        sample_qa_examples: list[QAExample],
    ):
        _write_jsonl(tmp_path / "corpus.jsonl", sample_documents)
        _write_jsonl(tmp_path / "qa.jsonl", sample_qa_examples)

        docs = load_corpus(path=tmp_path)
        qas = load_qa(path=tmp_path)

        assert len(docs) == len(sample_documents)
        assert len(qas) == len(sample_qa_examples)
        assert {d.doc_id for d in docs} == {d.doc_id for d in sample_documents}
        assert {q.qid for q in qas} == {q.qid for q in sample_qa_examples}
