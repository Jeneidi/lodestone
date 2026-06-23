"""lodestone.generation.claude — Claude-powered answer generation (optional).

This module is *off by default*.  It is activated only when:

1. ``LODESTONE_GENERATION_ENABLED=true`` in the environment / ``.env`` file.
2. A valid ``LODESTONE_ANTHROPIC_API_KEY`` (or ``ANTHROPIC_API_KEY``) is set.

If either condition is not met, :meth:`ClaudeAnswerer.answer` raises a
:exc:`RuntimeError` with a clear message explaining what to set.

The ``anthropic`` package is lazy-imported — importing this module does not
trigger a network call or require the package to be installed.

Prompt strategy
---------------
- **System prompt**: instructs the model to answer *only* from the provided
  context, never fabricate information, and say "I don't know" when the
  context is insufficient.
- **User message**: numbered list of chunk texts followed by the question.
- ``max_tokens=300`` — sufficient for concise factoid answers.
- ``temperature`` is left at the API default (1.0); the grounding constraint
  in the system prompt is the primary quality control mechanism.
"""

from __future__ import annotations

import logging
import time

from lodestone.schemas import Answer, ScoredChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant.  "
    "You will be given numbered context passages and a question.  "
    "Answer the question using ONLY information present in the provided context.  "
    "Do NOT cite, infer from, or reference any knowledge outside the context.  "
    "If the answer is not contained in the context, respond with exactly: "
    '"I don\'t know — the provided context does not contain this information."'
    "  Keep your answer concise (1–3 sentences)."
)

_USER_TEMPLATE = "{context_block}\n\nQuestion: {question}\n\nAnswer:"


def _build_context_block(chunks: list[ScoredChunk], max_chars: int = 4000) -> str:
    """Format chunk texts as a numbered list, truncated to *max_chars*.

    Args:
        chunks:    Retrieved chunks (highest score first).
        max_chars: Approximate character budget for the context block.

    Returns:
        A newline-separated numbered list of chunk texts.

    """
    lines: list[str] = []
    total = 0
    for i, sc in enumerate(chunks, start=1):
        text = sc.chunk.text.strip()
        entry = f"[{i}] {text}"
        if total + len(entry) > max_chars and lines:
            # Truncate gracefully — don't start a new entry we can't finish.
            break
        lines.append(entry)
        total += len(entry) + 1  # +1 for the newline separator
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ClaudeAnswerer:
    """Answer generator backed by the Anthropic Messages API.

    The ``anthropic`` client is created lazily on the first :meth:`answer`
    call to keep import time fast and to surface clear errors only when
    generation is actually attempted.

    Args:
        model: Anthropic model identifier.  Defaults to
               ``settings.generation_model`` (``claude-sonnet-4-6``).

    Raises:
        RuntimeError: On the first :meth:`answer` call if
            ``generation_enabled`` is ``False`` or the API key is missing.

    Example::

        import os
        os.environ["LODESTONE_GENERATION_ENABLED"] = "true"
        os.environ["LODESTONE_ANTHROPIC_API_KEY"] = "sk-ant-..."

        from lodestone.generation.claude import ClaudeAnswerer
        answerer = ClaudeAnswerer()
        ans = answerer.answer("What is photosynthesis?", chunks)
        print(ans.text)

    """

    def __init__(self, model: str | None = None) -> None:
        self._model_override: str | None = model
        self._client: object | None = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_client(self) -> object:
        """Return an ``anthropic.Anthropic`` client, creating it on first call.

        Raises:
            RuntimeError: If generation is disabled or the API key is absent.

        """
        if self._client is not None:
            return self._client

        from lodestone.config import get_settings  # noqa: PLC0415

        settings = get_settings()

        if not settings.generation_enabled:
            raise RuntimeError(
                "Claude generation is disabled.\n"
                "\n"
                "  To enable it, set the following in your .env or environment:\n"
                "\n"
                "      LODESTONE_GENERATION_ENABLED=true\n"
                "      LODESTONE_ANTHROPIC_API_KEY=sk-ant-...\n"
                "\n"
                "  Then re-instantiate ClaudeAnswerer (or restart your process)."
            )

        api_key = settings.anthropic_api_key
        if not api_key:
            # Also check the bare ANTHROPIC_API_KEY environment variable as a
            # convenience (the anthropic SDK itself respects this variable, but
            # we want a clear error message if neither is set).
            import os  # noqa: PLC0415

            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError(
                "No Anthropic API key found.\n"
                "\n"
                "  Provide the key via one of:\n"
                "\n"
                "      LODESTONE_ANTHROPIC_API_KEY=sk-ant-...   (in .env or environment)\n"
                "      ANTHROPIC_API_KEY=sk-ant-...             (bare env var)\n"
                "\n"
                "  Also ensure LODESTONE_GENERATION_ENABLED=true."
            )

        try:
            import anthropic  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for Claude generation.\n"
                "Install it with:  pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        logger.debug("Anthropic client initialised.")
        return self._client

    def _get_model(self) -> str:
        """Return the model identifier to use for this call."""
        if self._model_override:
            return self._model_override
        from lodestone.config import get_settings  # noqa: PLC0415

        return get_settings().generation_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, query: str, chunks: list[ScoredChunk]) -> Answer:
        """Generate a grounded answer for *query* using the Anthropic API.

        The model is instructed to respond **only** from the provided chunks.
        If no relevant information is found, it returns a polite "I don't know"
        response rather than hallucinating.

        Args:
            query:  The natural-language question.
            chunks: Retrieved chunks (highest-scored first).  At most the top
                    chunks fitting within ~4 000 characters are sent to the API.

        Returns:
            An :class:`~lodestone.schemas.Answer` with:
            - ``text``: the model's answer string.
            - ``supporting_chunks``: the input *chunks* unchanged.
            - ``generator``: ``"claude"``.
            - ``faithfulness``: ``None`` (not computed here).
            - ``latency_ms``: wall-clock time for the API call in ms.

        Raises:
            RuntimeError: If generation is disabled or the API key is missing.
            anthropic.APIError: On any Anthropic API-level error.

        """
        t_start = time.perf_counter()

        client = self._get_client()  # may raise RuntimeError
        model = self._get_model()

        context_block = _build_context_block(chunks)
        user_content = _USER_TEMPLATE.format(
            context_block=context_block,
            question=query,
        )

        logger.debug(
            "ClaudeAnswerer: calling model=%s, context_chars=%d.",
            model,
            len(context_block),
        )

        # Anthropic Messages API call
        response = client.messages.create(  # type: ignore[union-attr]
            model=model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        answer_text: str = response.content[0].text.strip() if response.content else ""

        latency = (time.perf_counter() - t_start) * 1000.0

        logger.debug(
            "ClaudeAnswerer: received %d chars in %.0f ms.",
            len(answer_text),
            latency,
        )

        return Answer(
            text=answer_text,
            supporting_chunks=list(chunks),
            generator="claude",
            faithfulness=None,
            latency_ms=latency,
        )


__all__ = ["ClaudeAnswerer"]
