"""lodestone.chunking — text chunking strategies package.

Contract
--------
A *Chunker* is any object that satisfies the :class:`Chunker` Protocol below.
Later waves register concrete implementations here; this wave only defines
the contract so that retrieval and evaluation code can type-check against it.

Protocol fields
~~~~~~~~~~~~~~~
- ``name: str``
    Human-readable identifier for the strategy (e.g. ``"fixed_512"``,
    ``"sentence"``, ``"recursive"``).  Used in evaluation reports and
    ablation sweep keys.

- ``chunk(self, doc: Document) -> list[Chunk]``
    Split ``doc`` into a list of :class:`~lodestone.schemas.Chunk` objects.

    Guarantees the implementation MUST satisfy:

    * Every returned ``Chunk`` has ``doc_id == doc.doc_id``.
    * ``chunk_id`` values are unique within the returned list (convention:
      ``f"{doc.doc_id}_{index}"``).
    * ``index`` values are zero-based and contiguous.
    * The concatenation of all ``chunk.text`` values covers all semantically
      significant content in ``doc.text`` (no silent data loss).
    * If ``doc.text`` is empty, return ``[]``.

Thread safety
-------------
Chunkers are not required to be thread-safe.

Example concrete implementation (illustrative, not in this file)::

    class FixedWindowChunker:
        name = "fixed_512"

        def __init__(self, window: int = 512, overlap: int = 64) -> None:
            self.window = window
            self.overlap = overlap

        def chunk(self, doc: Document) -> list[Chunk]:
            tokens = doc.text.split()
            step = self.window - self.overlap
            chunks = []
            for i, start in enumerate(range(0, len(tokens), step)):
                span = " ".join(tokens[start : start + self.window])
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_{i}",
                        doc_id=doc.doc_id,
                        text=span,
                        index=i,
                    )
                )
            return chunks
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lodestone.schemas import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Structural protocol for text-chunking strategies.

    Any class that exposes a ``name: str`` attribute and a ``chunk`` method
    with the correct signature satisfies this protocol — no explicit
    inheritance required.

    Attributes:
        name: Human-readable identifier for this chunking strategy.
              Used in ablation sweep keys and evaluation report headers.

    """

    name: str

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into an ordered list of non-overlapping (or
        overlapping, depending on strategy) text chunks.

        Args:
            doc: The source document to split.

        Returns:
            A list of :class:`~lodestone.schemas.Chunk` objects.
            Returns an empty list if ``doc.text`` is empty.

        """
        ...


from lodestone.chunking.strategies import FixedSizeChunker, SentenceWindowChunker

__all__ = ["Chunker", "FixedSizeChunker", "SentenceWindowChunker"]
