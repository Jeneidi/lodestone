"""Lodestone — a hybrid retrieval engine built from first principles.

This package provides:

- :mod:`lodestone.schemas`   — core data models (Document, Chunk, ScoredChunk, …)
- :mod:`lodestone.config`    — runtime settings via pydantic-settings
- :mod:`lodestone.retrieval` — retriever contract + concrete backends (wave 2+)
- :mod:`lodestone.chunking`  — chunker protocol + concrete strategies (wave 2+)
- :mod:`lodestone.generation`— answer generation (extractive + Claude) (wave 3+)
- :mod:`lodestone.api`       — FastAPI server (wave 3+)
- :mod:`lodestone.cli`       — Typer CLI (wave 2+)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("lodestone")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
