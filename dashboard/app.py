"""Lodestone — Streamlit portfolio dashboard.

Provides three tabs:

- **Ask**:    Full RAG pipeline.  Enter a question and see the answer,
              faithfulness metric, latency, and per-chunk evidence.
- **Search**: Lexical + dense hybrid retrieval results rendered as a dataframe.
- **Eval results**: Renders evaluation reports from ``evals/reports/`` when
              available; otherwise prompts the user to run ``make eval``.

Run::

    streamlit run dashboard/app.py

The ``src/`` directory is inserted into :data:`sys.path` at module load time so
the dashboard works when executed directly from the project root without
requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the src layout is importable when running from the project root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Standard-library imports (no lodestone yet — keep import order clean)
# ---------------------------------------------------------------------------
import json
import logging

import streamlit as st

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lodestone",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Engine singleton (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Lodestone engine (this may take a moment)…")
def _load_engine(use_rerank: bool) -> object:
    """Load and return the :class:`~lodestone.engine.LodestoneEngine`.

    Args:
        use_rerank: Whether to load the cross-encoder reranker.

    Returns:
        A loaded :class:`~lodestone.engine.LodestoneEngine` instance.

    Raises:
        FileNotFoundError: If the corpus has not been built yet.

    """
    from lodestone.engine import LodestoneEngine  # noqa: PLC0415

    engine = LodestoneEngine()
    engine.load(use_rerank=use_rerank)
    return engine


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Lodestone")
    st.caption("Hybrid BM25 + Dense RAG engine")
    st.divider()

    k_val: int = st.slider("Top-k chunks", min_value=1, max_value=20, value=5)
    rerank_on: bool = st.toggle("Cross-encoder reranking", value=True)
    faith_on: bool = st.toggle("Faithfulness scoring (NLI)", value=False)

    st.divider()
    st.caption(
        "Run `make data` to build the corpus,  \n"
        "`make eval` to generate eval reports."
    )

# ---------------------------------------------------------------------------
# Load the engine (or show an error banner)
# ---------------------------------------------------------------------------

_engine_error: str | None = None
_engine = None

try:
    _engine = _load_engine(use_rerank=rerank_on)
except FileNotFoundError as _exc:
    _engine_error = str(_exc)
except Exception as _exc:
    _engine_error = f"Unexpected error loading engine: {_exc}"

if _engine_error:
    st.error(
        f"**Engine not ready.**\n\n```\n{_engine_error}\n```\n\n"
        "Run `make data` to build the corpus, then reload this page."
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ask, tab_search, tab_eval = st.tabs(["Ask", "Search", "Eval results"])

# ===========================================================================
# Tab: Ask
# ===========================================================================

with tab_ask:
    st.header("Ask a question")

    st.info(
        "**Lodestone is specialized in the following topics:** "
        "Super Bowl 50 · Nikola Tesla · The Normans · Warsaw · "
        "Computational Complexity Theory · The Teaching Profession  \n"
        "These topics were intentionally chosen from unrelated domains to stress-test "
        "the retrieval engine's ability to distinguish and retrieve across diverse subject areas."
    )

    ask_query = st.text_input(
        "Question",
        placeholder="e.g. Ask something related to any of the above topics...",
        key="ask_query",
    )
    ask_btn = st.button("Ask", key="ask_btn", type="primary", disabled=_engine is None)

    if ask_btn and ask_query.strip() and _engine is not None:
        with st.spinner("Thinking…"):
            from lodestone.engine import LodestoneEngine  # noqa: PLC0415

            engine: LodestoneEngine = _engine  # type: ignore[assignment]
            answer = engine.ask(
                ask_query.strip(), k=k_val, score_faithfulness=faith_on
            )

        # Answer
        st.subheader("Answer")
        st.markdown(
            f'<div style="background:#1e1e2e;padding:1.2rem 1.4rem;border-radius:8px;'
            f'border-left:4px solid #7c9dff;font-size:1.05rem;">{answer.text or "<em>(no answer)</em>"}</div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Generator", answer.generator or "—")
        col2.metric("Latency (ms)", f"{answer.latency_ms:.1f}")
        if faith_on and answer.faithfulness is not None:
            col3.metric("Faithfulness", f"{answer.faithfulness:.3f}")
        else:
            col3.metric("Faithfulness", "—")

        # Supporting chunks
        if answer.supporting_chunks:
            st.subheader("Supporting chunks")
            for idx, sc in enumerate(answer.supporting_chunks, start=1):
                with st.expander(
                    f"Chunk {idx} — {sc.chunk.doc_id}  |  score {sc.score:.4f}  |  {sc.retriever}",
                    expanded=(idx == 1),
                ):
                    st.text(sc.chunk.text)

# ===========================================================================
# Tab: Search
# ===========================================================================

with tab_search:
    st.header("Search")

    st.info(
        "**Lodestone is specialized in the following topics:** "
        "Super Bowl 50 · Nikola Tesla · The Normans · Warsaw · "
        "Computational Complexity Theory · The Teaching Profession  \n"
        "These topics were intentionally chosen from unrelated domains to stress-test "
        "the retrieval engine's ability to distinguish and retrieve across diverse subject areas."
    )

    search_query = st.text_input(
        "Query",
        placeholder="e.g. Ask something related to any of the above topics...",
        key="search_query",
    )
    search_btn = st.button(
        "Search", key="search_btn", type="primary", disabled=_engine is None
    )

    if search_btn and search_query.strip() and _engine is not None:
        with st.spinner("Retrieving…"):
            from lodestone.engine import LodestoneEngine  # noqa: PLC0415

            eng2: LodestoneEngine = _engine  # type: ignore[assignment]
            hits = eng2.search(search_query.strip(), k=k_val)

        if not hits:
            st.info("No results found.")
        else:
            import pandas as pd  # noqa: PLC0415

            rows = [
                {
                    "rank": i,
                    "score": round(sc.score, 5),
                    "retriever": sc.retriever,
                    "doc_id": sc.chunk.doc_id,
                    "snippet": sc.chunk.text[:200].replace("\n", " "),
                }
                for i, sc in enumerate(hits, start=1)
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

# ===========================================================================
# Tab: Eval results
# ===========================================================================

with tab_eval:
    st.header("Evaluation results")

    reports_dir = _REPO_ROOT / "evals" / "reports"
    results_json = reports_dir / "results.json"

    if not results_json.exists():
        st.info(
            "No evaluation report found.  "
            "Run `make eval` to generate retrieval + generation metrics,  \n"
            "then `make ablate` for the ablation study.  \n\n"
            f"Expected path: `{results_json}`"
        )
    else:
        # ------------------------------------------------------------------
        # Parse results.json defensively
        # ------------------------------------------------------------------
        try:
            with results_json.open("r", encoding="utf-8") as fh:
                report: dict = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read `results.json`: {exc}")
            report = {}

        if report:
            import pandas as pd  # noqa: PLC0415

            # Metadata section (optional)
            metadata: dict = report.get("metadata", {})
            if metadata:
                st.subheader("Run metadata")
                meta_rows = [{"key": k, "value": str(v)} for k, v in metadata.items()]
                st.dataframe(
                    pd.DataFrame(meta_rows), use_container_width=True, hide_index=True
                )

            # Per-pipeline summaries (optional)
            pipelines: dict = report.get("pipelines", {})
            if not pipelines:
                # Fallback: maybe the top-level IS the summary
                pipelines = {
                    k: v
                    for k, v in report.items()
                    if isinstance(v, dict) and k != "metadata"
                }

            if pipelines:
                st.subheader("Pipeline metrics")

                summary_rows = []
                for pipeline_name, metrics in pipelines.items():
                    if not isinstance(metrics, dict):
                        continue
                    row: dict[str, object] = {"pipeline": pipeline_name}
                    for metric_name, metric_val in metrics.items():
                        if isinstance(metric_val, dict):
                            # Structured as {mean, ci_lo, ci_hi}
                            mean = metric_val.get("mean")
                            ci_lo = metric_val.get("ci_lo")
                            ci_hi = metric_val.get("ci_hi")
                            if mean is not None:
                                formatted = f"{mean:.4f}"
                                if ci_lo is not None and ci_hi is not None:
                                    formatted += f" [{ci_lo:.4f}, {ci_hi:.4f}]"
                                row[metric_name] = formatted
                            else:
                                row[metric_name] = str(metric_val)
                        else:
                            try:
                                row[metric_name] = (
                                    f"{float(metric_val):.4f}"  # type: ignore[arg-type]
                                    if metric_val is not None
                                    else "—"
                                )
                            except (TypeError, ValueError):
                                row[metric_name] = str(metric_val)
                    summary_rows.append(row)

                if summary_rows:
                    st.dataframe(
                        pd.DataFrame(summary_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        # ------------------------------------------------------------------
        # Ablation plots
        # ------------------------------------------------------------------
        ablation_ndcg = reports_dir / "ablation_ndcg.png"
        latency_quality = reports_dir / "latency_quality.png"

        has_plots = ablation_ndcg.exists() or latency_quality.exists()
        if has_plots:
            st.subheader("Ablation plots")

        if ablation_ndcg.exists():
            st.image(str(ablation_ndcg), caption="Ablation — nDCG", use_column_width=True)

        if latency_quality.exists():
            st.image(
                str(latency_quality), caption="Latency vs quality", use_column_width=True
            )

        if not has_plots:
            st.info(
                "No ablation plots found.  Run `make ablate` to generate them.  \n"
                f"Expected: `{ablation_ndcg}` and `{latency_quality}`"
            )
