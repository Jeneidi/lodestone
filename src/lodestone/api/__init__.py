"""lodestone.api — FastAPI server package.

This package is scaffolded in wave 1; the full server implementation
(``server.py``) is added in wave 3.

Planned endpoints (wave 3):

``POST /search``
    Accept a query string and optional retriever config; return ranked chunks.

``POST /answer``
    Accept a query string; return an :class:`~lodestone.schemas.Answer` with
    faithfulness score.

``GET /health``
    Lightweight liveness probe.

The server is started via::

    make serve
    # equivalent to: uvicorn lodestone.api.server:app --reload
"""
