"""Lodestone core data schemas.

These Pydantic v2 models form the *single source of truth* for every data
structure that flows between components.  Later agents import these names
verbatim — do NOT rename fields without coordinating across all waves.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A raw source document before any chunking.

    Attributes:
        doc_id:   Unique identifier for the document (e.g. Wikipedia page ID,
                  filename stem, or UUID).
        title:    Human-readable title; empty string when unavailable.
        text:     Full raw text of the document.
        source:   Provenance string (URL, file path, dataset name, …).
        metadata: Arbitrary key/value pairs for downstream filtering or
                  logging (e.g. {"split": "train", "language": "en"}).

    """

    doc_id: str
    title: str = ""
    text: str
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A contiguous text span produced by a Chunker from a Document.

    Attributes:
        chunk_id: Globally unique identifier for this chunk
                  (convention: ``f"{doc_id}_{index}"``).
        doc_id:   Identifier of the parent Document.
        text:     Chunk text (the span that will be indexed and retrieved).
        index:    Zero-based position of the chunk within its parent document.
        metadata: Arbitrary key/value pairs (e.g. start/end char offsets,
                  section heading, page number).

    """

    chunk_id: str
    doc_id: str
    text: str
    index: int = 0
    metadata: dict = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A Chunk paired with a retrieval score and the name of its retriever.

    Scores are *always* higher-is-better and have no fixed range (they are
    retriever-specific).  Fusion components are responsible for normalising
    scores before combining them.

    Attributes:
        chunk:     The retrieved chunk.
        score:     Relevance score; higher values indicate greater relevance.
        retriever: Human-readable name of the retriever that produced this
                   result (e.g. ``"bm25"``, ``"dense"``, ``"hybrid"``).

    """

    chunk: Chunk
    score: float
    retriever: str = ""


class QAExample(BaseModel):
    """A single question-answer pair with ground-truth relevance labels.

    Used by the evaluation harness to compute retrieval metrics
    (Recall@k, MRR, nDCG@k, Precision@k) and answer quality metrics.

    Attributes:
        qid:              Unique question identifier.
        question:         Natural-language question string.
        answer:           Reference (gold) answer string.
        relevant_doc_ids: List of doc_ids that are considered relevant for
                          this query (used to compute retrieval metrics).

    """

    qid: str
    question: str
    answer: str
    relevant_doc_ids: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    """A generated answer, optionally annotated with faithfulness.

    Attributes:
        text:               The answer text (extractive span or generated).
        supporting_chunks:  Chunks used as evidence for the answer.
        generator:          Which generator produced this answer:
                            ``"extractive"`` (no model call) or ``"claude"``
                            (Anthropic API, requires LODESTONE_GENERATION_ENABLED).
        faithfulness:       NLI-based faithfulness score in [0, 1] — ``None``
                            if not yet computed.
        latency_ms:         End-to-end answer generation latency in
                            milliseconds (retrieval + generation).

    """

    text: str
    supporting_chunks: list[ScoredChunk] = Field(default_factory=list)
    generator: str = ""          # "extractive" or "claude"
    faithfulness: float | None = None
    latency_ms: float = 0.0


class RetrievalRunResult(BaseModel):
    """The output of a single retrieval call, with timing information.

    Attributes:
        qid:         Question identifier corresponding to a QAExample.
        retrieved:   Ordered list of scored chunks (highest score first).
        latency_ms:  Wall-clock retrieval latency in milliseconds.

    """

    qid: str
    retrieved: list[ScoredChunk]
    latency_ms: float = 0.0
