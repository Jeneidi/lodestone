"""lodestone.api.server — FastAPI application for the Lodestone hybrid RAG engine.

Endpoints
---------
``GET  /``          Redirect / welcome with name, version, and docs link.
``GET  /health``    Liveness probe: reports whether the corpus is loaded.
``POST /search``    Retrieve ranked chunks for a query.
``POST /ask``       Full RAG pipeline: retrieve + generate + optional faithfulness.

The engine is initialised lazily on the first request via
:func:`~lodestone.engine.get_engine`.  If the corpus has not been built yet,
a ``503 Service Unavailable`` response is returned with a hint to run
``make data``.

Start the server::

    make serve
    # or: uvicorn lodestone.api.server:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lodestone.schemas import Answer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Lodestone",
    version="1.0.0",
    description=(
        "A hybrid BM25 + Dense retrieval engine with optional cross-encoder "
        "reranking and Claude-powered answer generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for ``POST /search``.

    Attributes:
        query: Natural-language search query.
        k:     Number of results to return.  Defaults to 10.

    """

    query: str = Field(..., min_length=1, description="Search query string.")
    k: int = Field(default=10, ge=1, le=100, description="Number of results.")


class ChunkResult(BaseModel):
    """A single result item returned by ``POST /search``.

    Attributes:
        chunk_id:  Unique identifier for the chunk.
        doc_id:    Parent document identifier.
        text:      Full chunk text.
        score:     Retrieval / reranker score (higher is better).
        retriever: Name of the retriever that produced this result.

    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    retriever: str


class SearchResponse(BaseModel):
    """Response body for ``POST /search``.

    Attributes:
        results:    Ranked list of chunk results.
        latency_ms: End-to-end retrieval latency in milliseconds.

    """

    results: list[ChunkResult]
    latency_ms: float


class AskRequest(BaseModel):
    """Request body for ``POST /ask``.

    Attributes:
        query:       Natural-language question.
        k:           Number of supporting chunks to retrieve.  Defaults to 5.
        faithfulness: When ``True``, compute and return an NLI faithfulness
                      score.  Adds latency due to model inference.

    """

    query: str = Field(..., min_length=1, description="Question string.")
    k: int = Field(default=5, ge=1, le=50, description="Number of supporting chunks.")
    faithfulness: bool = Field(default=False, description="Compute faithfulness score.")


# ---------------------------------------------------------------------------
# Engine dependency
# ---------------------------------------------------------------------------

def _get_engine() -> object:
    """FastAPI dependency that returns the loaded engine or raises 503.

    Returns:
        The :class:`~lodestone.engine.LodestoneEngine` singleton.

    Raises:
        HTTPException: 503 if the corpus has not been built yet.

    """
    try:
        from lodestone.engine import get_engine  # noqa: PLC0415
        return get_engine()
    except FileNotFoundError as exc:
        logger.warning("Engine not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Corpus not loaded.  Build the dataset first:\n\n"
                "    make data\n\n"
                "Then restart the server."
            ),
        ) from exc


EngineDep = Annotated[object, Depends(_get_engine)]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Welcome endpoint returning service name, version, and docs link.

    Returns:
        JSON with ``name``, ``version``, and ``docs`` fields.

    """
    return JSONResponse(
        content={
            "name": "Lodestone",
            "version": "1.0.0",
            "docs": "/docs",
        }
    )


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Liveness and readiness probe.

    Returns ``{"status": "ok", "corpus_loaded": bool}`` without loading the
    engine (so the check is always fast).

    Returns:
        JSON body with ``status`` and ``corpus_loaded`` keys.

    """
    try:
        from lodestone.engine import get_engine  # noqa: PLC0415
        engine = get_engine()
        corpus_loaded: bool = getattr(engine, "_loaded", False)
    except FileNotFoundError:
        corpus_loaded = False

    return JSONResponse(content={"status": "ok", "corpus_loaded": corpus_loaded})


@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, engine: EngineDep) -> SearchResponse:
    """Retrieve ranked chunks for a query.

    Args:
        body:   Request body with ``query`` and optional ``k``.
        engine: Injected engine dependency.

    Returns:
        :class:`SearchResponse` with ranked ``results`` and ``latency_ms``.

    """
    import time  # noqa: PLC0415

    t0 = time.perf_counter()

    from lodestone.engine import LodestoneEngine  # noqa: PLC0415
    eng: LodestoneEngine = engine  # type: ignore[assignment]
    hits = eng.search(body.query, k=body.k)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    results = [
        ChunkResult(
            chunk_id=sc.chunk.chunk_id,
            doc_id=sc.chunk.doc_id,
            text=sc.chunk.text,
            score=sc.score,
            retriever=sc.retriever,
        )
        for sc in hits
    ]

    return SearchResponse(results=results, latency_ms=latency_ms)


@app.post("/ask", response_model=Answer)
async def ask(body: AskRequest, engine: EngineDep) -> Answer:
    """Full RAG pipeline: retrieve, generate, and optionally score faithfulness.

    Args:
        body:   Request body with ``query``, optional ``k``, and
                ``faithfulness`` flag.
        engine: Injected engine dependency.

    Returns:
        :class:`~lodestone.schemas.Answer` with all fields populated,
        including ``latency_ms`` for end-to-end timing.

    """
    from lodestone.engine import LodestoneEngine  # noqa: PLC0415
    eng: LodestoneEngine = engine  # type: ignore[assignment]
    return eng.ask(
        query=body.query,
        k=body.k,
        score_faithfulness=body.faithfulness,
    )


__all__ = ["app"]
