"""lodestone.generation — answer generation backends.

Public surface
--------------
- :class:`~lodestone.generation.extractive.ExtractiveAnswerer` — free, LLM-free
  extractive generation using lexical sentence scoring.  Always available.

- :class:`~lodestone.generation.claude.ClaudeAnswerer` — optional Claude
  generation via the Anthropic Messages API.  Requires
  ``LODESTONE_GENERATION_ENABLED=true`` and a valid API key.

- :class:`~lodestone.generation.faithfulness.NliFaithfulnessScorer` — NLI-based
  faithfulness scoring.  Lazy-loads ``sentence-transformers``.

- :func:`get_answerer` — factory that returns a :class:`ClaudeAnswerer` when
  generation is enabled and the API key is present, otherwise falls back to
  :class:`ExtractiveAnswerer`.

Generation strategies
---------------------
``"extractive"`` (default, free, no model call)
    Selects the most relevant sentence(s) from the top-ranked chunks using
    Jaccard overlap scoring.  Deterministic, instant, no dependencies.

``"claude"`` (optional, requires LODESTONE_GENERATION_ENABLED=true)
    Calls the Anthropic API with the retrieved chunks as context.  Requires
    a valid ``LODESTONE_ANTHROPIC_API_KEY`` environment variable.

Both generators return an :class:`~lodestone.schemas.Answer`.

Usage::

    from lodestone.generation import get_answerer

    answerer = get_answerer()          # auto-selects based on settings
    ans = answerer.answer(query, chunks)
"""

from __future__ import annotations

from lodestone.generation.claude import ClaudeAnswerer
from lodestone.generation.extractive import ExtractiveAnswerer
from lodestone.generation.faithfulness import NliFaithfulnessScorer


def get_answerer(
    settings: object | None = None,
) -> ClaudeAnswerer | ExtractiveAnswerer:
    """Return the appropriate answerer based on current settings.

    Returns a :class:`ClaudeAnswerer` when:
    - ``settings.generation_enabled`` is ``True``, **and**
    - either ``settings.anthropic_api_key`` is set or the ``ANTHROPIC_API_KEY``
      environment variable is present.

    Otherwise returns an :class:`ExtractiveAnswerer` with default parameters.

    Args:
        settings: Optional :class:`~lodestone.config.Settings` instance.
                  When ``None``, the singleton from
                  :func:`~lodestone.config.get_settings` is used.

    Returns:
        Either a :class:`ClaudeAnswerer` or an :class:`ExtractiveAnswerer`.

    Example::

        from lodestone.generation import get_answerer

        answerer = get_answerer()
        ans = answerer.answer("What is photosynthesis?", chunks)
        print(ans.generator)   # "extractive" or "claude"

    """
    import os  # noqa: PLC0415

    if settings is None:
        from lodestone.config import get_settings  # noqa: PLC0415
        settings = get_settings()

    generation_enabled: bool = getattr(settings, "generation_enabled", False)
    api_key: str | None = (
        getattr(settings, "anthropic_api_key", None)
        or os.environ.get("ANTHROPIC_API_KEY")
    )

    if generation_enabled and api_key:
        return ClaudeAnswerer()

    return ExtractiveAnswerer()


__all__ = [
    "ClaudeAnswerer",
    "ExtractiveAnswerer",
    "NliFaithfulnessScorer",
    "get_answerer",
]
