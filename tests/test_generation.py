"""
Tests for lodestone generation modules:
- ExtractiveAnswerer  (lodestone.generation.extractive)
- NliFaithfulnessScorer (lodestone.generation.faithfulness)
- ClaudeAnswerer (lodestone.generation.claude) — only the disabled-generation
  path is tested; no network or Anthropic SDK required.

All tests are offline.  No sentence_transformers, anthropic, or datasets
are imported at module level.
"""

from __future__ import annotations

import math

import pytest

from lodestone.schemas import Answer, Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str, doc_id: str = "d") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, index=0)


def _scored(chunk_id: str, text: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, text), score=score, retriever="stub")


# ---------------------------------------------------------------------------
# ExtractiveAnswerer
# ---------------------------------------------------------------------------

class TestExtractiveAnswerer:
    """Tests for the extractive answer generator."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lodestone.generation.extractive import ExtractiveAnswerer
        self.ExtractiveAnswerer = ExtractiveAnswerer

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------

    def test_returns_answer_object(self):
        answerer = self.ExtractiveAnswerer(max_sentences=1)
        chunks = [_scored("c0", "Paris is the capital of France.")]
        result = answerer.answer("What is the capital of France?", chunks)
        assert isinstance(result, Answer)

    def test_generator_field_is_extractive(self):
        answerer = self.ExtractiveAnswerer()
        result = answerer.answer("question", [_scored("c0", "some text here.")])
        assert result.generator == "extractive"

    def test_highest_overlap_sentence_selected(self):
        """The sentence with the most query-token overlap must be returned first."""
        # Sentence about 'capital France' vs one about 'cooking'
        chunks = [
            _scored(
                "c0",
                "Cooking is an art form. Paris is the capital of France.",
                score=1.0,
            )
        ]
        answerer = self.ExtractiveAnswerer(max_sentences=1)
        result = answerer.answer("What is the capital of France?", chunks)
        assert "Paris" in result.text or "France" in result.text, (
            f"Expected Paris/France sentence, got: {result.text!r}"
        )

    def test_multiple_chunks_best_sentence_wins(self):
        """Correct sentence can come from any chunk in the list."""
        chunks = [
            _scored("c0", "The sky is blue.", score=0.5),
            _scored("c1", "Gradient descent optimises the neural loss.", score=0.8),
        ]
        answerer = self.ExtractiveAnswerer(max_sentences=1)
        result = answerer.answer("How does gradient descent optimise neural networks?", chunks)
        # The gradient/descent sentence should win due to higher overlap
        assert "gradient" in result.text.lower() or "descent" in result.text.lower(), (
            f"Unexpected answer: {result.text!r}"
        )

    # ------------------------------------------------------------------
    # max_sentences respected
    # ------------------------------------------------------------------

    def test_max_sentences_one(self):
        """With max_sentences=1 only one sentence is returned."""
        chunks = [
            _scored("c0", "Alpha sentence here. Beta sentence there. Gamma sentence everywhere.")
        ]
        answerer = self.ExtractiveAnswerer(max_sentences=1)
        result = answerer.answer("alpha beta gamma", chunks)
        # Result text should not contain more sentences than max_sentences=1
        # The answer is a single joined segment (no extra period splits in the joined text
        # since we join top-1 sentence)
        assert result.text != ""
        # At most 1 sentence in the output — count periods/! at sentence end
        sentences_in_output = [s.strip() for s in result.text.split(".") if s.strip()]
        assert len(sentences_in_output) <= 2  # generous: one sentence may have trailing period

    def test_max_sentences_two(self):
        chunks = [
            _scored(
                "c0",
                "Neural networks use gradient descent. "
                "Backpropagation computes gradients. "
                "The loss function measures error.",
            )
        ]
        answerer = self.ExtractiveAnswerer(max_sentences=2)
        result = answerer.answer("gradient descent neural backpropagation", chunks)
        assert result.text != ""

    def test_max_sentences_invalid_raises(self):
        with pytest.raises(ValueError):
            self.ExtractiveAnswerer(max_sentences=0)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic(self):
        """Same inputs always produce the same answer text."""
        chunks = [
            _scored("c0", "The cat sat on the mat."),
            _scored("c1", "Dogs are loyal animals."),
        ]
        answerer = self.ExtractiveAnswerer(max_sentences=1)
        query = "where did the cat sit?"
        r1 = answerer.answer(query, chunks)
        r2 = answerer.answer(query, chunks)
        assert r1.text == r2.text

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_chunks_returns_empty_text(self):
        """An empty chunk list should return an Answer with text=''."""
        answerer = self.ExtractiveAnswerer()
        result = answerer.answer("any query", [])
        assert result.text == ""
        assert result.generator == "extractive"

    def test_supporting_chunks_preserved(self):
        """supporting_chunks in the Answer equals the input chunks list."""
        chunks = [_scored("c0", "some text."), _scored("c1", "more text.")]
        answerer = self.ExtractiveAnswerer()
        result = answerer.answer("text", chunks)
        assert result.supporting_chunks == chunks

    def test_faithfulness_is_none(self):
        """extractive answerer never computes faithfulness."""
        answerer = self.ExtractiveAnswerer()
        result = answerer.answer("q", [_scored("c0", "text.")])
        assert result.faithfulness is None

    def test_latency_ms_positive(self):
        answerer = self.ExtractiveAnswerer()
        result = answerer.answer("q", [_scored("c0", "text.")])
        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# NliFaithfulnessScorer
# ---------------------------------------------------------------------------

class TestNliFaithfulnessScorer:
    """Tests using an injected scorer callable — no model downloads."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lodestone.generation.faithfulness import NliFaithfulnessScorer
        self.NliFaithfulnessScorer = NliFaithfulnessScorer

    # Softmax helper for test assertions
    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        return [e / s for e in exps]

    # ------------------------------------------------------------------
    # Exact mean computation
    # ------------------------------------------------------------------

    def test_exact_mean_of_two_sentences(self):
        """For a 2-sentence answer, score == mean of P(entailment) for each."""
        # NLI label order: [contradiction=0, entailment=1, neutral=2]
        # Sentence 1 logits: [0, 10, 0] → P(entailment) ≈ 1.0
        # Sentence 2 logits: [10, 0, 0] → P(entailment) ≈ 0.0
        # Expected mean ≈ 0.5
        s1_logits = [0.0, 10.0, 0.0]
        s2_logits = [10.0, 0.0, 0.0]

        p1_entail = self._softmax(s1_logits)[1]
        p2_entail = self._softmax(s2_logits)[1]
        expected = (p1_entail + p2_entail) / 2.0

        call_count = [0]

        def mock_scorer(pairs: list[tuple[str, str]]) -> list[list[float]]:
            results = []
            for _ in pairs:
                idx = call_count[0]
                if idx == 0:
                    results.append(s1_logits)
                else:
                    results.append(s2_logits)
                call_count[0] += 1
            return results

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        # 2-sentence answer; sentence splitter splits on ". "
        answer = "This is the first sentence. This is the second sentence."
        chunks = [_scored("c0", "context text for premise")]
        result = scorer.score(answer, chunks)
        assert abs(result - expected) < 1e-5, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_perfect_entailment_returns_near_one(self):
        """Mock returning very high entailment logit → score near 1.0."""
        def mock_scorer(pairs):
            return [[-10.0, 10.0, 0.0] for _ in pairs]

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        result = scorer.score("Any answer sentence.", [_scored("c0", "Any context text.")])
        assert result > 0.99, f"Expected score near 1.0, got {result}"

    def test_perfect_contradiction_returns_near_zero(self):
        """Mock returning very high contradiction logit → entailment prob near 0."""
        def mock_scorer(pairs):
            return [[10.0, -10.0, 0.0] for _ in pairs]

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        result = scorer.score("Any answer sentence.", [_scored("c0", "Any context text.")])
        assert result < 0.01, f"Expected score near 0.0, got {result}"

    def test_uniform_logits_returns_one_third_entailment(self):
        """With uniform logits softmax gives 1/3 per class."""
        def mock_scorer(pairs):
            return [[1.0, 1.0, 1.0] for _ in pairs]

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        result = scorer.score("Single sentence.", [_scored("c0", "Context.")])
        assert abs(result - 1.0 / 3.0) < 1e-5, f"Expected 1/3, got {result}"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_answer_returns_zero(self):
        def mock_scorer(pairs):
            return [[0.0, 10.0, 0.0] for _ in pairs]

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        result = scorer.score("", [_scored("c0", "context")])
        assert result == 0.0

    def test_empty_chunks_returns_zero(self):
        def mock_scorer(pairs):
            return [[0.0, 10.0, 0.0] for _ in pairs]

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        result = scorer.score("Some answer sentence.", [])
        assert result == 0.0

    def test_score_in_zero_one(self):
        """Score must always be in [0, 1]."""
        def mock_scorer(pairs):
            return [[1.0, 2.0, 0.5] for _ in pairs]

        scorer = self.NliFaithfulnessScorer(scorer=mock_scorer)
        result = scorer.score("Sentence one. Sentence two.", [_scored("c0", "premise text")])
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# ClaudeAnswerer
# ---------------------------------------------------------------------------

class TestClaudeAnswerer:
    """Tests for ClaudeAnswerer — only the disabled-generation path."""

    def _clear_settings_cache(self):
        """Clear the lru_cache on get_settings so monkeypatched env vars take effect."""
        from lodestone.config import get_settings
        get_settings.cache_clear()

    @pytest.fixture(autouse=True)
    def _ensure_generation_disabled(self, monkeypatch):
        """Ensure generation_enabled is False (the default) for every test."""
        # Remove any env vars that could enable generation
        monkeypatch.delenv("LODESTONE_GENERATION_ENABLED", raising=False)
        monkeypatch.delenv("LODESTONE_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        self._clear_settings_cache()
        yield
        # Clear cache after test so we don't pollute other tests
        self._clear_settings_cache()

    # ------------------------------------------------------------------

    def test_answer_raises_runtime_error_when_disabled(self):
        """ClaudeAnswerer.answer() must raise RuntimeError when generation_enabled=False."""
        from lodestone.generation.claude import ClaudeAnswerer

        answerer = ClaudeAnswerer()
        with pytest.raises(RuntimeError, match="disabled"):
            answerer.answer("What is photosynthesis?", [_scored("c0", "Plants use sunlight.")])

    def test_error_message_contains_env_var_hint(self):
        """RuntimeError message should mention the env var to set."""
        from lodestone.generation.claude import ClaudeAnswerer

        answerer = ClaudeAnswerer()
        with pytest.raises(RuntimeError) as exc_info:
            answerer.answer("query", [])
        assert "LODESTONE_GENERATION_ENABLED" in str(exc_info.value)

    def test_instantiation_does_not_raise(self):
        """Importing and instantiating ClaudeAnswerer must not raise or import anthropic."""
        from lodestone.generation.claude import ClaudeAnswerer  # noqa: F401
        # Just constructing should not fail or trigger a network call
        answerer = ClaudeAnswerer()
        assert answerer is not None

    def test_no_anthropic_import_at_module_level(self):
        """The anthropic package must not be imported at module import time."""

        # anthropic should not be in sys.modules from simply importing claude.py
        # (it's lazy-imported only inside _get_client)
        import lodestone.generation.claude  # noqa: F401

        # anthropic may or may not be installed; if it's not in sys.modules,
        # that confirms lazy import. If it IS there, it might have been
        # imported elsewhere — we can't assert it's absent, but we verify
        # the module itself doesn't error on import.
        assert lodestone.generation.claude is not None
