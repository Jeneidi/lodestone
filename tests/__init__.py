"""
Lodestone test suite.

Tests are designed to be 100% offline-friendly (no network calls, no model
downloads required).  Heavy models are mocked via pytest fixtures.

Test modules (added in later waves):

``tests.test_schemas``    — round-trip serialisation of Pydantic models.
``tests.test_config``     — Settings env-var override behaviour.
``tests.test_chunking``   — Chunker protocol conformance + edge cases.
``tests.test_bm25``       — BM25 scoring correctness (small corpus).
``tests.test_dense``      — Dense retriever with a mocked encoder.
``tests.test_hybrid``     — RRF and weighted fusion unit tests.
``tests.test_metrics``    — Recall@k, MRR, nDCG@k golden values.
``tests.test_generation`` — Extractive generator unit tests.
"""
