"""lodestone.data — corpus and QA dataset loaders.

Public surface
--------------
- :func:`load_corpus` — load ``corpus.jsonl`` → ``list[Document]``
- :func:`load_qa`     — load ``qa.jsonl``     → ``list[QAExample]``

Both functions default to the ``data_dir`` configured via
:func:`~lodestone.config.get_settings`.  Pass an explicit *path* to override.

If the requested file does not exist, a :exc:`FileNotFoundError` is raised
with a clear message telling the user to run ``make data``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # keep the import block tidy; real imports below

from lodestone.schemas import Document, QAExample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_path(path: str | Path | None, filename: str) -> Path:
    """Return an absolute :class:`~pathlib.Path` for *filename* under *path*.

    If *path* is ``None``, the ``data_dir`` from settings is used.

    Args:
        path:     Explicit directory or file path, or ``None`` to use settings.
        filename: File name to append when *path* is a directory.

    Returns:
        Resolved :class:`~pathlib.Path`.

    """
    if path is None:
        # Lazy import to avoid circular imports and to allow the module to be
        # imported even when pydantic-settings is not yet installed.
        from lodestone.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        resolved = Path(settings.data_dir) / filename
    else:
        resolved = Path(path)
        # If the caller supplied a directory, append the expected filename.
        if resolved.is_dir():
            resolved = resolved / filename

    return resolved


def _missing_file_error(path: Path, filename: str) -> FileNotFoundError:
    """Return a :exc:`FileNotFoundError` with a helpful message."""
    return FileNotFoundError(
        f"Dataset file not found: {path}\n"
        f"\n"
        f"  The file '{filename}' has not been built yet.  "
        f"Generate it by running:\n"
        f"\n"
        f"      make data\n"
        f"\n"
        f"  or directly:\n"
        f"\n"
        f"      python scripts/build_dataset.py\n"
        f"\n"
        f"  Use --force to rebuild if files already exist."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_corpus(path: str | Path | None = None) -> list[Document]:
    """Load the document corpus from a JSONL file.

    Each line of the file must be a valid JSON object matching the
    :class:`~lodestone.schemas.Document` schema.

    Args:
        path: Path to ``corpus.jsonl``, a directory that contains it, or
              ``None`` to use the ``data_dir`` from
              :func:`~lodestone.config.get_settings`.

    Returns:
        Ordered list of :class:`~lodestone.schemas.Document` objects.

    Raises:
        FileNotFoundError: If the corpus file does not exist.  The error
            message instructs the user to run ``make data``.
        json.JSONDecodeError: If any line is not valid JSON.
        pydantic.ValidationError: If any record does not match the
            Document schema.

    Example::

        from lodestone.data import load_corpus
        docs = load_corpus()          # uses settings.data_dir
        docs = load_corpus("data/")   # explicit directory

    """
    filename = "corpus.jsonl"
    resolved = _resolve_path(path, filename)

    if not resolved.exists():
        raise _missing_file_error(resolved, filename)

    documents: list[Document] = []
    with resolved.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {lineno} of {resolved}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
            documents.append(Document.model_validate(data))

    logger.info("Loaded %d documents from %s.", len(documents), resolved)
    return documents


def load_qa(path: str | Path | None = None) -> list[QAExample]:
    """Load the QA evaluation set from a JSONL file.

    Each line of the file must be a valid JSON object matching the
    :class:`~lodestone.schemas.QAExample` schema.

    Args:
        path: Path to ``qa.jsonl``, a directory that contains it, or
              ``None`` to use the ``data_dir`` from
              :func:`~lodestone.config.get_settings`.

    Returns:
        Ordered list of :class:`~lodestone.schemas.QAExample` objects.

    Raises:
        FileNotFoundError: If the QA file does not exist.  The error
            message instructs the user to run ``make data``.
        json.JSONDecodeError: If any line is not valid JSON.
        pydantic.ValidationError: If any record does not match the
            QAExample schema.

    Example::

        from lodestone.data import load_qa
        examples = load_qa()         # uses settings.data_dir
        examples = load_qa("data/")  # explicit directory

    """
    filename = "qa.jsonl"
    resolved = _resolve_path(path, filename)

    if not resolved.exists():
        raise _missing_file_error(resolved, filename)

    examples: list[QAExample] = []
    with resolved.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {lineno} of {resolved}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
            examples.append(QAExample.model_validate(data))

    logger.info("Loaded %d QA examples from %s.", len(examples), resolved)
    return examples


__all__ = ["load_corpus", "load_qa"]
