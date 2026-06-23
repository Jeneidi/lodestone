"""Lodestone command-line interface.

Entry point: ``lodestone`` (installed via pyproject.toml console_scripts).

Usage::

    lodestone --help
    lodestone search "What is retrieval-augmented generation?" --k 10
    lodestone ask "What is backpropagation?" --k 5 --faithfulness
    lodestone info
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="lodestone",
    help=(
        "Lodestone — a hybrid retrieval engine built from first principles.\n\n"
        "Run 'lodestone COMMAND --help' for command-specific help."
    ),
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

console = Console()

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Root callback
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed Lodestone version."""
    from lodestone import __version__  # noqa: PLC0415

    console.print(f"lodestone [bold]{__version__}[/bold]")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="Query string to search."),
    k: int = typer.Option(10, "--k", "-k", help="Number of results to return."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable cross-encoder reranking."),
) -> None:
    """Search the indexed corpus and display ranked results.

    Retrieves the top-k chunks and renders them as a rich table with rank,
    score, doc_id, and a snippet (first ~120 characters of chunk text).

    Args:
        query:     Natural-language query string.
        k:         Number of results to return.
        no_rerank: When set, skips cross-encoder reranking.

    """
    try:
        from lodestone.engine import LodestoneEngine  # noqa: PLC0415

        engine = LodestoneEngine()
        engine.load(use_rerank=not no_rerank)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    with console.status("[bold green]Searching...[/bold green]"):
        results = engine.search(query, k=k)

    table = Table(
        title=f"Search results for: [italic]{query}[/italic]",
        show_lines=True,
        highlight=True,
    )
    table.add_column("Rank", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green")
    table.add_column("Doc ID", style="yellow", no_wrap=True)
    table.add_column("Retriever", style="dim")
    table.add_column("Snippet")

    for rank, sc in enumerate(results, start=1):
        snippet = sc.chunk.text[:120].replace("\n", " ")
        if len(sc.chunk.text) > 120:
            snippet += "…"
        table.add_row(
            str(rank),
            f"{sc.score:.4f}",
            sc.chunk.doc_id,
            sc.retriever,
            snippet,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    k: int = typer.Option(5, "--k", "-k", help="Number of supporting chunks."),
    faithfulness: bool = typer.Option(
        False, "--faithfulness", help="Compute NLI faithfulness score."
    ),
) -> None:
    """Answer a question using the full RAG pipeline.

    Retrieves supporting evidence, generates an answer, and optionally scores
    faithfulness.  Displays the answer in a rich Panel followed by a table of
    supporting chunks.

    Args:
        question:    Natural-language question string.
        k:           Number of supporting chunks to retrieve.
        faithfulness: When set, compute and display the NLI faithfulness score.

    """
    try:
        from lodestone.engine import LodestoneEngine  # noqa: PLC0415

        engine = LodestoneEngine()
        engine.load()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    with console.status("[bold green]Thinking...[/bold green]"):
        answer = engine.ask(question, k=k, score_faithfulness=faithfulness)

    # Answer panel
    meta_parts = [
        f"Generator: [bold]{answer.generator}[/bold]",
        f"Latency: [bold]{answer.latency_ms:.1f} ms[/bold]",
    ]
    if faithfulness and answer.faithfulness is not None:
        meta_parts.append(f"Faithfulness: [bold]{answer.faithfulness:.3f}[/bold]")

    meta_line = "  |  ".join(meta_parts)
    console.print(
        Panel(
            f"[white]{answer.text}[/white]\n\n[dim]{meta_line}[/dim]",
            title=f"[bold blue]Answer[/bold blue]: [italic]{question}[/italic]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    if not answer.supporting_chunks:
        return

    # Supporting chunks table
    chunks_table = Table(
        title="Supporting chunks",
        show_lines=True,
        highlight=True,
    )
    chunks_table.add_column("Rank", justify="right", style="bold cyan", no_wrap=True)
    chunks_table.add_column("Score", justify="right", style="green")
    chunks_table.add_column("Doc ID", style="yellow", no_wrap=True)
    chunks_table.add_column("Retriever", style="dim")
    chunks_table.add_column("Snippet")

    for rank, sc in enumerate(answer.supporting_chunks, start=1):
        snippet = sc.chunk.text[:120].replace("\n", " ")
        if len(sc.chunk.text) > 120:
            snippet += "…"
        chunks_table.add_row(
            str(rank),
            f"{sc.score:.4f}",
            sc.chunk.doc_id,
            sc.retriever,
            snippet,
        )

    console.print(chunks_table)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Display current Lodestone settings and data-file status.

    Shows all configuration values (masking the Anthropic API key) and
    whether the corpus and QA data files exist, with row counts if available.
    """
    from lodestone.config import get_settings  # noqa: PLC0415

    settings = get_settings()

    # Settings table
    settings_table = Table(
        title="Lodestone settings",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    settings_table.add_column("Setting", style="cyan", no_wrap=True)
    settings_table.add_column("Value")

    masked_key: str | None = None
    raw_key = settings.anthropic_api_key
    if raw_key:
        visible = raw_key[:6] if len(raw_key) >= 6 else raw_key
        masked_key = visible + "*" * max(0, len(raw_key) - 6)

    rows: list[tuple[str, str]] = [
        ("embedding_model_name", settings.embedding_model_name),
        ("reranker_model_name", settings.reranker_model_name),
        ("nli_model_name", settings.nli_model_name),
        ("top_k", str(settings.top_k)),
        ("rerank_top_k", str(settings.rerank_top_k)),
        ("rrf_k", str(settings.rrf_k)),
        ("hybrid_alpha", str(settings.hybrid_alpha)),
        ("generation_enabled", str(settings.generation_enabled)),
        ("anthropic_api_key", masked_key or "[dim](not set)[/dim]"),
        ("generation_model", settings.generation_model),
        ("data_dir", settings.data_dir),
        ("reports_dir", settings.reports_dir),
    ]

    for name, value in rows:
        settings_table.add_row(name, value)

    console.print(settings_table)

    # Data files table
    data_table = Table(
        title="Data files",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    data_table.add_column("File", style="cyan", no_wrap=True)
    data_table.add_column("Exists", justify="center")
    data_table.add_column("Rows", justify="right")

    data_root = Path(settings.data_dir)

    for filename in ("corpus.jsonl", "qa.jsonl"):
        fpath = data_root / filename
        if fpath.exists():
            try:
                with fpath.open("r", encoding="utf-8") as fh:
                    row_count = sum(1 for line in fh if line.strip())
                count_str = str(row_count)
            except OSError:
                count_str = "[dim]?[/dim]"
            data_table.add_row(str(fpath), "[green]yes[/green]", count_str)
        else:
            data_table.add_row(str(fpath), "[red]no[/red]", "[dim]—[/dim]")

    console.print(data_table)


if __name__ == "__main__":
    app()
