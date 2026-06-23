"""Lodestone runtime configuration.

All settings are read from environment variables with the ``LODESTONE_``
prefix.  Secrets (e.g. ``ANTHROPIC_API_KEY``) can also be placed in a
``.env`` file at the project root — pydantic-settings will load it
automatically.

Usage
-----
Import the singleton accessor anywhere in the codebase::

    from lodestone.config import get_settings

    settings = get_settings()
    print(settings.embedding_model_name)

The :func:`get_settings` function is cached via ``functools.lru_cache``, so
the settings object is constructed exactly once per process.

Environment variable reference
-------------------------------
See ``.env.example`` at the project root for a commented template.
"""

from __future__ import annotations

import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic-settings model for all Lodestone runtime parameters.

    All fields can be overridden via environment variables using the
    ``LODESTONE_`` prefix (e.g. ``LODESTONE_TOP_K=20``).

    Model configuration
    -------------------
    - ``env_prefix = "LODESTONE_"``
    - ``.env`` file is loaded automatically if present (``env_file = ".env"``,
      ``env_file_encoding = "utf-8"``).
    - Extra fields are forbidden to catch typos early.
    """

    model_config = SettingsConfigDict(
        env_prefix="LODESTONE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Embedding / retrieval models
    # ------------------------------------------------------------------

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "HuggingFace model identifier for the bi-encoder used in dense "
            "retrieval.  Must be compatible with sentence-transformers."
        ),
    )

    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description=(
            "HuggingFace model identifier for the cross-encoder reranker.  "
            "Must expose a classification head compatible with "
            "sentence-transformers CrossEncoder."
        ),
    )

    nli_model_name: str = Field(
        default="cross-encoder/nli-deberta-v3-xsmall",
        description=(
            "HuggingFace model identifier for the NLI cross-encoder used to "
            "score answer faithfulness.  The model must output entailment / "
            "neutral / contradiction logits."
        ),
    )

    # ------------------------------------------------------------------
    # Retrieval hyper-parameters
    # ------------------------------------------------------------------

    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of chunks returned by the retriever to downstream components.",
    )

    rerank_top_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of chunks to retain after cross-encoder reranking.  "
            "Must be ≤ top_k."
        ),
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "Reciprocal Rank Fusion smoothing constant.  "
            "Score for rank r is 1 / (rrf_k + r).  "
            "Higher values reduce the impact of top-ranked results."
        ),
    )

    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Interpolation weight for weighted score fusion: "
            "``score = alpha * dense_score + (1 - alpha) * bm25_score``.  "
            "0.0 → pure BM25, 1.0 → pure dense."
        ),
    )

    # ------------------------------------------------------------------
    # Generation / LLM settings
    # ------------------------------------------------------------------

    generation_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for Claude-powered answer generation.  "
            "When False (default), the extractive fallback is used and no "
            "Anthropic API calls are made.  "
            "Set LODESTONE_GENERATION_ENABLED=true to enable."
        ),
    )

    anthropic_api_key: str | None = Field(
        default=None,
        description=(
            "Anthropic API key.  Required only when generation_enabled=True.  "
            "Can also be provided via the standard ANTHROPIC_API_KEY env var."
        ),
    )

    generation_model: str = Field(
        default="claude-sonnet-4-6",
        description=(
            "Anthropic model identifier used for answer generation when "
            "generation_enabled=True."
        ),
    )

    # ------------------------------------------------------------------
    # File system paths
    # ------------------------------------------------------------------

    data_dir: str = Field(
        default="data",
        description=(
            "Root directory for raw and processed dataset files.  "
            "Relative paths are resolved from the project root."
        ),
    )

    reports_dir: str = Field(
        default="evals/reports",
        description=(
            "Directory where evaluation harness writes metric reports, "
            "ablation plots, and the markdown summary."
        ),
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) global :class:`Settings` instance.

    The settings object is constructed on first call and reused on every
    subsequent call.  Use this function throughout the codebase instead of
    constructing ``Settings()`` directly.

    Returns:
        The singleton :class:`Settings` instance.

    Example::

        from lodestone.config import get_settings
        cfg = get_settings()
        model = cfg.embedding_model_name

    """
    return Settings()
