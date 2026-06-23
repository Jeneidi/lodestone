"""
scripts/build_dataset.py
------------------------
Build the Lodestone evaluation dataset into data/.

Writes:
  data/corpus.jsonl  — one Document per line (model_dump JSON)
  data/qa.jsonl      — one QAExample per line (model_dump JSON)

Primary source: HuggingFace SQuAD validation split.
Fallback source: built-in factual paragraphs (when download fails).

Usage
-----
    python scripts/build_dataset.py [--docs N] [--questions M] [--force]
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Ensure the src/ layout is importable even without pip install
# ---------------------------------------------------------------------------
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import hashlib
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Lodestone imports
# ---------------------------------------------------------------------------
from lodestone.config import get_settings
from lodestone.schemas import Document, QAExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in fallback corpus (used when HuggingFace download fails)
# ---------------------------------------------------------------------------

_FALLBACK_PARAGRAPHS: list[dict] = [
    {
        "title": "The Solar System",
        "text": (
            "The Solar System consists of the Sun and the objects that orbit it, "
            "including eight planets, their moons, dwarf planets, and countless "
            "smaller bodies such as asteroids and comets.  The four inner rocky "
            "planets—Mercury, Venus, Earth, and Mars—are separated from the four "
            "giant outer planets by the asteroid belt."
        ),
    },
    {
        "title": "Photosynthesis",
        "text": (
            "Photosynthesis is the process by which green plants and certain other "
            "organisms use sunlight to synthesise nutrients from carbon dioxide and "
            "water.  The overall chemical equation is: 6CO2 + 6H2O + light energy "
            "→ C6H12O6 + 6O2.  Chlorophyll in the chloroplasts absorbs red and "
            "blue light while reflecting green light, which is why plants appear green."
        ),
    },
    {
        "title": "The French Revolution",
        "text": (
            "The French Revolution (1789–1799) was a period of radical political and "
            "societal transformation in France.  It began with the convening of the "
            "Estates-General in 1789 and ended with Napoleon Bonaparte's coup on "
            "18 Brumaire.  The revolution abolished the absolute monarchy, established "
            "republican principles, and produced the Declaration of the Rights of "
            "Man and of the Citizen."
        ),
    },
    {
        "title": "DNA and Genetics",
        "text": (
            "Deoxyribonucleic acid (DNA) is the hereditary material in humans and "
            "almost all other organisms.  The structure of DNA is a double helix "
            "formed by base pairs attached to a sugar-phosphate backbone.  The four "
            "nitrogenous bases are adenine (A), thymine (T), guanine (G), and "
            "cytosine (C); A pairs with T and G pairs with C."
        ),
    },
    {
        "title": "The Internet",
        "text": (
            "The Internet is a global system of interconnected computer networks that "
            "use the Internet protocol suite (TCP/IP) to communicate.  It was developed "
            "from ARPANET, a U.S. Department of Defense project in the late 1960s.  "
            "The World Wide Web, invented by Tim Berners-Lee in 1989, is the most "
            "widely used service running on top of the Internet."
        ),
    },
    {
        "title": "Plate Tectonics",
        "text": (
            "Plate tectonics is the scientific theory describing the large-scale motion "
            "of seven large plates and the movements of a larger number of smaller "
            "plates of Earth's lithosphere.  The theory was developed during the first "
            "half of the twentieth century by Alfred Wegener and others.  Earthquakes, "
            "volcanic activity, mountain-building, and oceanic trench formation occur "
            "along plate boundaries."
        ),
    },
    {
        "title": "Python Programming Language",
        "text": (
            "Python is a high-level, general-purpose programming language created by "
            "Guido van Rossum and first released in 1991.  Python's design emphasises "
            "code readability with the use of significant indentation.  It supports "
            "multiple programming paradigms, including structured, object-oriented, "
            "and functional programming."
        ),
    },
    {
        "title": "The Roman Empire",
        "text": (
            "The Roman Empire was the post-Republican state of ancient Rome that "
            "lasted from 27 BC, when Augustus became the first emperor, until the "
            "fall of the Western Roman Empire in AD 476.  At its height under Trajan "
            "in AD 117 the empire encompassed about 5 million km² and had a population "
            "of approximately 70 million people."
        ),
    },
    {
        "title": "Black Holes",
        "text": (
            "A black hole is a region of spacetime where gravity is so strong that "
            "nothing—not even light or other electromagnetic waves—has enough speed "
            "to escape it.  The boundary of a black hole is called the event horizon.  "
            "Black holes are formed when very massive stars collapse at the end of "
            "their life cycles, and supermassive black holes are found at the centre "
            "of most large galaxies."
        ),
    },
    {
        "title": "Antibiotics",
        "text": (
            "Antibiotics are a type of antimicrobial substance active against bacteria.  "
            "The first widely used antibiotic, penicillin, was discovered by Alexander "
            "Fleming in 1928.  Antibiotics work by inhibiting bacterial cell-wall "
            "synthesis, protein synthesis, DNA replication, or other vital processes.  "
            "Overuse of antibiotics has led to the emergence of antibiotic-resistant bacteria."
        ),
    },
    {
        "title": "Machine Learning",
        "text": (
            "Machine learning is a branch of artificial intelligence that gives computers "
            "the ability to learn from data without being explicitly programmed.  "
            "Supervised learning trains models on labelled data; unsupervised learning "
            "finds patterns in unlabelled data; reinforcement learning trains agents "
            "via reward signals.  Neural networks, decision trees, and support vector "
            "machines are among the most popular algorithms."
        ),
    },
    {
        "title": "The Water Cycle",
        "text": (
            "The water cycle (hydrological cycle) describes the continuous movement of "
            "water within Earth and its atmosphere.  Water evaporates from oceans and "
            "land, rises as water vapour, condenses into clouds, falls as precipitation "
            "(rain or snow), and flows back to the sea via rivers and groundwater.  "
            "The Sun's energy and gravity are the primary drivers of the cycle."
        ),
    },
]

_FALLBACK_QA: list[dict] = [
    # Solar System
    {
        "qid": "fb_001",
        "question": "How many planets are in the Solar System?",
        "answer": "eight",
        "title": "The Solar System",
    },
    {
        "qid": "fb_002",
        "question": "What separates the inner and outer planets?",
        "answer": "the asteroid belt",
        "title": "The Solar System",
    },
    # Photosynthesis
    {
        "qid": "fb_003",
        "question": "What gas is produced during photosynthesis?",
        "answer": "O2",
        "title": "Photosynthesis",
    },
    {
        "qid": "fb_004",
        "question": "Why do plants appear green?",
        "answer": "chlorophyll reflects green light",
        "title": "Photosynthesis",
    },
    # French Revolution
    {
        "qid": "fb_005",
        "question": "When did the French Revolution begin?",
        "answer": "1789",
        "title": "The French Revolution",
    },
    {
        "qid": "fb_006",
        "question": "What document did the French Revolution produce?",
        "answer": "Declaration of the Rights of Man and of the Citizen",
        "title": "The French Revolution",
    },
    # DNA
    {
        "qid": "fb_007",
        "question": "What are the four nitrogenous bases in DNA?",
        "answer": "adenine, thymine, guanine, and cytosine",
        "title": "DNA and Genetics",
    },
    {
        "qid": "fb_008",
        "question": "What is the shape of the DNA molecule?",
        "answer": "double helix",
        "title": "DNA and Genetics",
    },
    # Internet
    {
        "qid": "fb_009",
        "question": "Who invented the World Wide Web?",
        "answer": "Tim Berners-Lee",
        "title": "The Internet",
    },
    {
        "qid": "fb_010",
        "question": "What protocol suite does the Internet use?",
        "answer": "TCP/IP",
        "title": "The Internet",
    },
    # Plate Tectonics
    {
        "qid": "fb_011",
        "question": "Who developed the theory of plate tectonics?",
        "answer": "Alfred Wegener",
        "title": "Plate Tectonics",
    },
    # Python
    {
        "qid": "fb_012",
        "question": "Who created the Python programming language?",
        "answer": "Guido van Rossum",
        "title": "Python Programming Language",
    },
    {
        "qid": "fb_013",
        "question": "In what year was Python first released?",
        "answer": "1991",
        "title": "Python Programming Language",
    },
    # Roman Empire
    {
        "qid": "fb_014",
        "question": "Who was the first Roman Emperor?",
        "answer": "Augustus",
        "title": "The Roman Empire",
    },
    {
        "qid": "fb_015",
        "question": "When did the Western Roman Empire fall?",
        "answer": "AD 476",
        "title": "The Roman Empire",
    },
    # Black Holes
    {
        "qid": "fb_016",
        "question": "What is the boundary of a black hole called?",
        "answer": "event horizon",
        "title": "Black Holes",
    },
    # Antibiotics
    {
        "qid": "fb_017",
        "question": "Who discovered penicillin?",
        "answer": "Alexander Fleming",
        "title": "Antibiotics",
    },
    {
        "qid": "fb_018",
        "question": "When was penicillin discovered?",
        "answer": "1928",
        "title": "Antibiotics",
    },
    # Machine Learning
    {
        "qid": "fb_019",
        "question": "What type of learning uses labelled data?",
        "answer": "supervised learning",
        "title": "Machine Learning",
    },
    # Water Cycle
    {
        "qid": "fb_020",
        "question": "What are the primary drivers of the water cycle?",
        "answer": "the Sun's energy and gravity",
        "title": "The Water Cycle",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_id_from_context(context: str) -> str:
    """Return a 12-character hex SHA-1 hash of *context* for use as doc_id."""
    return hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]


def _build_from_squad(n_docs: int, n_questions: int) -> tuple[list[Document], list[QAExample]]:
    """Download SQuAD validation split and build corpus + QA pairs.

    Args:
        n_docs:      Maximum number of unique contexts to include.
        n_questions: Maximum number of QA examples to collect across all docs.

    Returns:
        Tuple of (documents, qa_examples).

    Raises:
        Exception: Re-raises any error so the caller can fall back.
    """
    from datasets import load_dataset  # type: ignore[import]

    logger.info("Downloading SQuAD validation split from HuggingFace…")
    # Try the canonical namespaced id first; fall back to the legacy short id
    try:
        dataset = load_dataset("rajpurkar/squad", split="validation")
    except Exception:
        dataset = load_dataset("squad", split="validation")

    # Group by unique context text to form documents
    seen_contexts: dict[str, Document] = {}
    qa_examples: list[QAExample] = []

    for row in dataset:
        context: str = row["context"]
        title: str = row["title"]
        doc_id = _doc_id_from_context(context)

        if doc_id not in seen_contexts:
            if len(seen_contexts) >= n_docs:
                continue  # already have enough docs; still collect QA below
            seen_contexts[doc_id] = Document(
                doc_id=doc_id,
                title=title,
                text=context,
                source="squad-validation",
                metadata={"squad_title": title},
            )

        # Only collect QA for doc_ids we kept
        if doc_id in seen_contexts and len(qa_examples) < n_questions:
            answer_text: str = row["answers"]["text"][0] if row["answers"]["text"] else ""
            qa_examples.append(
                QAExample(
                    qid=row["id"],
                    question=row["question"],
                    answer=answer_text,
                    relevant_doc_ids=[doc_id],
                )
            )

    documents = list(seen_contexts.values())
    logger.info(
        "SQuAD build complete: %d documents, %d QA pairs.",
        len(documents),
        len(qa_examples),
    )
    return documents, qa_examples


def _build_fallback() -> tuple[list[Document], list[QAExample]]:
    """Build a small built-in corpus from the hard-coded paragraphs.

    Returns:
        Tuple of (documents, qa_examples).
    """
    import warnings

    warnings.warn(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  WARNING: HuggingFace download failed.                          ║\n"
        "║  Falling back to the built-in 12-paragraph corpus.              ║\n"
        "║  This corpus is tiny and suitable ONLY for smoke-testing.       ║\n"
        "║  To use the real SQuAD dataset, ensure 'datasets' is installed  ║\n"
        "║  and internet access is available, then rerun with --force.     ║\n"
        "╚══════════════════════════════════════════════════════════════════╝",
        stacklevel=2,
    )

    # Build title → doc_id mapping for linking QA pairs
    title_to_doc_id: dict[str, str] = {}
    documents: list[Document] = []
    for para in _FALLBACK_PARAGRAPHS:
        doc_id = _doc_id_from_context(para["text"])
        title_to_doc_id[para["title"]] = doc_id
        documents.append(
            Document(
                doc_id=doc_id,
                title=para["title"],
                text=para["text"],
                source="builtin-fallback",
                metadata={"fallback": True},
            )
        )

    qa_examples: list[QAExample] = []
    for i, qa in enumerate(_FALLBACK_QA):
        doc_id = title_to_doc_id.get(qa["title"], "")
        qa_examples.append(
            QAExample(
                qid=qa["qid"],
                question=qa["question"],
                answer=qa["answer"],
                relevant_doc_ids=[doc_id] if doc_id else [],
            )
        )

    logger.info(
        "Fallback build complete: %d documents, %d QA pairs.", len(documents), len(qa_examples)
    )
    return documents, qa_examples


def _write_jsonl(records: list, path: Path) -> None:
    """Write *records* to *path* as newline-delimited JSON (model_dump)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    logger.debug("Wrote %d records to %s.", len(records), path)


def _print_rich_summary(
    documents: list[Document], qa_examples: list[QAExample], source: str
) -> None:
    """Print a human-readable summary table of the built dataset."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Lodestone Dataset Summary  [dim]({source})[/dim]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        avg_len = (sum(len(d.text) for d in documents) / len(documents)) if documents else 0.0

        table.add_row("Source", source)
        table.add_row("Documents (corpus.jsonl)", str(len(documents)))
        table.add_row("QA examples (qa.jsonl)", str(len(qa_examples)))
        table.add_row("Avg document length (chars)", f"{avg_len:,.0f}")
        console.print(table)

    except ImportError:
        # Rich not installed; plain fallback
        avg_len = (sum(len(d.text) for d in documents) / len(documents)) if documents else 0.0
        print("=" * 50)
        print(f"  Lodestone Dataset Summary  ({source})")
        print("=" * 50)
        print(f"  Documents (corpus.jsonl)        : {len(documents)}")
        print(f"  QA examples (qa.jsonl)          : {len(qa_examples)}")
        print(f"  Avg document length (chars)     : {avg_len:,.0f}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and orchestrate dataset construction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the Lodestone evaluation dataset into data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=300,
        metavar="N",
        help="Maximum number of unique context paragraphs to include as documents.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=400,
        metavar="M",
        help="Maximum number of QA examples to collect.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output files already exist.",
    )
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    qa_path = data_dir / "qa.jsonl"

    if corpus_path.exists() and qa_path.exists() and not args.force:
        print(f"Dataset already exists at {data_dir}/ — skipping build.\nUse --force to rebuild.")
        return

    # Try HuggingFace first; fall back to built-in corpus on any failure.
    source_label: str
    try:
        documents, qa_examples = _build_from_squad(
            n_docs=args.docs,
            n_questions=args.questions,
        )
        source_label = "squad-validation"
    except Exception as exc:
        logger.warning(
            "HuggingFace download failed (%s: %s). Using built-in fallback corpus.",
            type(exc).__name__,
            exc,
        )
        documents, qa_examples = _build_fallback()
        source_label = "builtin-fallback"

    _write_jsonl(documents, corpus_path)
    _write_jsonl(qa_examples, qa_path)

    logger.info("Wrote corpus  → %s", corpus_path)
    logger.info("Wrote QA data → %s", qa_path)

    _print_rich_summary(documents, qa_examples, source_label)


if __name__ == "__main__":
    main()
