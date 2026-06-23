"""
Shared fixtures for the Lodestone offline test suite.

Provides:
- small_corpus: 8 Documents covering distinct but vocabulary-overlapping topics.
- corpus_chunks: list[Chunk] from FixedSizeChunker(chunk_size=20, overlap=4).
- fake_encoder: deterministic bag-of-words encoder (32-dim, L2-normalised).
- fake_ce_scorer: cross-encoder scorer based on token overlap.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
import pytest

from lodestone.chunking.strategies import FixedSizeChunker
from lodestone.schemas import Chunk, Document

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_CORPUS_TEXTS = [
    # 0 — machine learning / neural networks
    (
        "doc_ml",
        "Machine Learning",
        (
            "Neural networks learn by adjusting weights through backpropagation. "
            "Gradient descent optimises the loss function iteratively. "
            "Deep learning models require large datasets and significant computation. "
            "Regularisation techniques such as dropout prevent overfitting."
        ),
    ),
    # 1 — natural language processing (overlaps with ML vocabulary)
    (
        "doc_nlp",
        "Natural Language Processing",
        (
            "Transformer architectures revolutionised natural language processing. "
            "Attention mechanisms allow models to focus on relevant tokens. "
            "Pre-training on large text corpora improves downstream task performance. "
            "Fine-tuning adapts a pre-trained language model to a specific domain."
        ),
    ),
    # 2 — biology / genetics
    (
        "doc_bio",
        "Genetics",
        (
            "DNA encodes genetic information in sequences of nucleotides. "
            "Genes are transcribed into messenger RNA, which is translated into protein. "
            "Mutations alter the nucleotide sequence and may affect protein function. "
            "CRISPR enables precise genome editing at targeted DNA locations."
        ),
    ),
    # 3 — climate science
    (
        "doc_climate",
        "Climate Science",
        (
            "Global warming results from increased concentrations of greenhouse gases. "
            "Carbon dioxide and methane trap heat in the atmosphere. "
            "Rising temperatures cause glacier retreat and sea-level rise. "
            "Renewable energy sources reduce carbon emissions significantly."
        ),
    ),
    # 4 — ancient history
    (
        "doc_history",
        "Ancient History",
        (
            "The Roman Empire expanded across Europe, North Africa and the Middle East. "
            "Julius Caesar crossed the Rubicon river in 49 BC, sparking civil war. "
            "Ancient Greek philosophy laid the foundations of Western thought. "
            "The Silk Road connected China to the Mediterranean world through trade."
        ),
    ),
    # 5 — quantum physics
    (
        "doc_physics",
        "Quantum Physics",
        (
            "Quantum mechanics describes the behaviour of particles at subatomic scales. "
            "The Heisenberg uncertainty principle limits simultaneous measurement of "
            "position and momentum. "
            "Superposition allows a quantum system to exist in multiple states at once. "
            "Entanglement correlates the states of distant particles instantaneously."
        ),
    ),
    # 6 — cooking / nutrition (very different vocabulary)
    (
        "doc_cooking",
        "Cooking and Nutrition",
        (
            "Fermentation transforms sugars into alcohol and organic acids using microorganisms. "
            "The Maillard reaction browns proteins and sugars when heated above 140 degrees. "
            "Essential amino acids must be obtained from dietary sources. "
            "Vitamins and minerals regulate metabolic processes in the human body."
        ),
    ),
    # 7 — software engineering (partial vocabulary overlap with ML)
    (
        "doc_software",
        "Software Engineering",
        (
            "Version control systems track changes to source code over time. "
            "Test-driven development improves code quality and reduces defects. "
            "Continuous integration runs automated tests on every code commit. "
            "Microservices architecture decomposes applications into independent services."
        ),
    ),
]


@pytest.fixture(scope="session")
def small_corpus() -> list[Document]:
    """Eight Documents with distinct but partially overlapping vocabularies."""
    return [
        Document(doc_id=doc_id, title=title, text=text, source="test")
        for doc_id, title, text in _CORPUS_TEXTS
    ]


@pytest.fixture(scope="session")
def corpus_chunks(small_corpus: list[Document]) -> list[Chunk]:
    """Chunks produced by FixedSizeChunker(chunk_size=20, overlap=4)."""
    chunker = FixedSizeChunker(chunk_size=20, overlap=4)
    chunks: list[Chunk] = []
    for doc in small_corpus:
        chunks.extend(chunker.chunk(doc))
    return chunks


# ---------------------------------------------------------------------------
# Fake encoder — deterministic 32-dim bag-of-words, L2-normalised
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")
_DIM = 32


def _hash_to_bucket(token: str, n_buckets: int = _DIM) -> int:
    """Map a token string to a bucket index via a simple polynomial hash."""
    h = 5381
    for ch in token.lower():
        h = ((h << 5) + h) ^ ord(ch)
    return h % n_buckets


def _text_to_vector(text: str) -> np.ndarray:
    """Convert text into a 32-dim bag-of-words vector and L2-normalise it.

    Each word token is hashed into one of 32 buckets; the count is
    incremented.  The result is L2-normalised (zero vectors stay zero).
    Semantically similar texts share more tokens and thus score higher
    under cosine similarity.
    """
    tokens = _WORD_RE.findall(text.lower())
    vec = np.zeros(_DIM, dtype=np.float32)
    for token in tokens:
        bucket = _hash_to_bucket(token)
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def fake_encoder(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into a (len, 32) float32 matrix.

    Deterministic bag-of-words encoder — no models, no network.
    Texts sharing more vocabulary produce more similar vectors.
    """
    return np.stack([_text_to_vector(t) for t in texts], axis=0)


@pytest.fixture(scope="session")
def fake_encoder_fn() -> Callable[[list[str]], np.ndarray]:
    """Session-scoped fixture returning the fake_encoder callable."""
    return fake_encoder


# ---------------------------------------------------------------------------
# Fake cross-encoder scorer — token-overlap fraction
# ---------------------------------------------------------------------------

def fake_ce_scorer(pairs: list[tuple[str, str]]) -> np.ndarray:
    """Score (query, passage) pairs by token-overlap fraction.

    Returns raw logits (log-odds of overlap fraction) so that
    sigmoid(logit) maps to a probability in (0, 1).

    Higher token overlap → higher logit → higher sigmoid score.
    """
    logits = []
    for query, passage in pairs:
        q_tokens = set(_WORD_RE.findall(query.lower()))
        p_tokens = set(_WORD_RE.findall(passage.lower()))
        if not q_tokens or not p_tokens:
            logits.append(0.0)
            continue
        overlap = len(q_tokens & p_tokens) / len(q_tokens | p_tokens)
        # Map to logit space: log(p / (1-p)) clamped to avoid inf
        overlap_clamped = max(1e-6, min(1.0 - 1e-6, overlap))
        logit = math.log(overlap_clamped / (1.0 - overlap_clamped))
        logits.append(float(logit))
    return np.array(logits, dtype=np.float32)


@pytest.fixture(scope="session")
def fake_ce_scorer_fn() -> Callable[[list[tuple[str, str]]], np.ndarray]:
    """Session-scoped fixture returning the fake_ce_scorer callable."""
    return fake_ce_scorer
