"""lodestone.generation.faithfulness — NLI-based answer faithfulness scoring.

Methodology
-----------
Faithfulness measures whether the generated answer is *entailed* by the
retrieved context — i.e., the context logically implies the answer.

1. **Context assembly**: concatenate the text of the top retrieved chunks up
   to ~1 500 characters to form a single *premise*.

2. **Answer segmentation**: split the answer into individual sentences using
   the same regex-based splitter used in
   :mod:`~lodestone.generation.extractive`.

3. **NLI inference**: for each answer sentence (*hypothesis*), run the
   cross-encoder NLI model to obtain logits over the three NLI labels.
   The model used by default is ``cross-encoder/nli-deberta-v3-xsmall``
   (configurable via ``LODESTONE_NLI_MODEL_NAME``), which outputs logits in
   the order **[contradiction, entailment, neutral]**.

4. **Softmax → P(entailment)**: apply softmax to the three logits and extract
   the probability of the ``entailment`` label (index 1).

5. **Faithfulness score**: mean of P(entailment) across all answer sentences.
   Sentences that are empty or where the model is uncertain yield lower scores.

The ``scorer`` argument allows injecting a mock callable in tests — any
callable with signature ``(pairs: list[tuple[str,str]]) -> list[list[float]]``
can be used in place of the real CrossEncoder.

Heavy imports (``sentence_transformers``) are lazy — importing this module
does not trigger a model download.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence

from lodestone.schemas import ScoredChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitter (mirrors extractive.py)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a regex-based splitter.

    Args:
        text: Input string.

    Returns:
        List of non-empty sentence strings.

    """
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------


def _softmax(logits: Sequence[float]) -> list[float]:
    """Compute softmax over *logits*.

    Args:
        logits: Raw model output scores.

    Returns:
        Probability distribution that sums to 1.0.

    """
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _build_premise(chunks: list[ScoredChunk], max_chars: int = 1500) -> str:
    """Concatenate top-chunk texts into a single premise string.

    Chunks are taken in the order provided (assumed highest-score first).
    Text is truncated to *max_chars* to stay within safe NLI model input limits.

    Args:
        chunks:    Retrieved scored chunks.
        max_chars: Maximum number of characters for the premise.

    Returns:
        Concatenated chunk texts, space-separated and truncated.

    """
    parts: list[str] = []
    total = 0
    for sc in chunks:
        text = sc.chunk.text.strip()
        if not text:
            continue
        if total + len(text) > max_chars and parts:
            # Include a partial chunk rather than omitting entirely
            remaining = max_chars - total
            if remaining > 20:  # only worth including if meaningful
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text) + 1  # +1 for the space separator
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class NliFaithfulnessScorer:
    """Compute faithfulness of an answer with respect to retrieved chunks.

    Uses a sentence-transformers CrossEncoder trained on NLI to estimate the
    probability that the retrieved context *entails* each answer sentence.

    Args:
        model_name: HuggingFace model identifier for the NLI cross-encoder.
                    Defaults to ``settings.nli_model_name``
                    (``cross-encoder/nli-deberta-v3-xsmall``).
        scorer:     Optional injectable callable for testing.  Must accept
                    ``list[tuple[str, str]]`` (premise, hypothesis) pairs and
                    return ``list[list[float]]`` (raw logits per pair).
                    When provided, *model_name* is ignored and no model is
                    loaded.

    Example::

        from lodestone.generation.faithfulness import NliFaithfulnessScorer
        scorer = NliFaithfulnessScorer()
        faith = scorer.score("Paris is in France.", chunks)
        print(f"Faithfulness: {faith:.2f}")

    Testing with a mock scorer::

        def mock_scorer(pairs):
            # Always return perfect entailment
            return [[-10.0, 10.0, 0.0] for _ in pairs]

        scorer = NliFaithfulnessScorer(scorer=mock_scorer)
        assert scorer.score("Any text.", chunks) == 1.0

    """

    #: Index of the entailment label in the NLI model output.
    #: cross-encoder/nli-deberta-v3-xsmall outputs [contradiction, entailment, neutral]
    _ENTAILMENT_IDX: int = 1

    def __init__(
        self,
        model_name: str | None = None,
        scorer: Callable[[list[tuple[str, str]]], list[list[float]]] | None = None,
    ) -> None:
        self._model_name_override = model_name
        self._scorer_override = scorer
        self._model: object | None = None  # lazy-initialised CrossEncoder

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_scorer(
        self,
    ) -> Callable[[list[tuple[str, str]]], list[list[float]]]:
        """Return a callable that scores (premise, hypothesis) pairs.

        If a ``scorer`` was injected at construction time, return it directly.
        Otherwise, lazy-load the CrossEncoder model and wrap its ``predict``
        method.

        Returns:
            Callable accepting ``list[tuple[str, str]]`` and returning
            ``list[list[float]]`` of raw NLI logits.

        """
        if self._scorer_override is not None:
            return self._scorer_override

        if self._model is not None:
            return self._model.predict  # type: ignore[union-attr]

        from lodestone.config import get_settings  # noqa: PLC0415

        model_name = self._model_name_override or get_settings().nli_model_name

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformers' package is required for faithfulness scoring.\n"
                "Install it with:  pip install sentence-transformers"
            ) from exc

        logger.info("Loading NLI model: %s (this may take a moment…)", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("NLI model loaded.")
        return self._model.predict  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, answer_text: str, chunks: list[ScoredChunk]) -> float:
        """Compute a faithfulness score for *answer_text* given *chunks*.

        For each sentence in *answer_text*, the NLI model is queried with
        the pair ``(premise=context, hypothesis=sentence)`` to obtain logits.
        Softmax is applied and the entailment probability is extracted.
        The final score is the mean entailment probability across all sentences.

        Args:
            answer_text: The generated (or extractive) answer string.
            chunks:      Retrieved scored chunks used as the evidence context.

        Returns:
            Faithfulness score in ``[0.0, 1.0]``.  Returns ``0.0`` if
            *answer_text* has no parseable sentences or *chunks* is empty.

        Notes:
            - An empty *answer_text* or empty *chunks* list returns ``0.0``.
            - The premise is truncated to ~1 500 characters; very long answers
              are segmented into sentences scored independently against the
              same shared premise.

        """
        sentences = _split_sentences(answer_text)
        if not sentences:
            logger.debug("NliFaithfulnessScorer: no sentences in answer.")
            return 0.0

        if not chunks:
            logger.debug("NliFaithfulnessScorer: no chunks provided.")
            return 0.0

        premise = _build_premise(chunks)
        if not premise:
            logger.debug("NliFaithfulnessScorer: empty premise from chunks.")
            return 0.0

        pairs: list[tuple[str, str]] = [(premise, sent) for sent in sentences]

        scorer_fn = self._get_scorer()

        try:
            raw_logits: list[list[float]] = scorer_fn(pairs)
        except Exception as exc:
            logger.warning(
                "NliFaithfulnessScorer: inference failed (%s: %s). Returning 0.0.",
                type(exc).__name__,
                exc,
            )
            return 0.0

        entailment_probs: list[float] = []
        for logits in raw_logits:
            probs = _softmax(logits)
            p_entail = probs[self._ENTAILMENT_IDX] if len(probs) > self._ENTAILMENT_IDX else 0.0
            entailment_probs.append(p_entail)

        if not entailment_probs:
            return 0.0

        faithfulness = sum(entailment_probs) / len(entailment_probs)

        logger.debug(
            "NliFaithfulnessScorer: %d sentences, mean P(entailment)=%.4f.",
            len(sentences),
            faithfulness,
        )

        return faithfulness


__all__ = ["NliFaithfulnessScorer"]
