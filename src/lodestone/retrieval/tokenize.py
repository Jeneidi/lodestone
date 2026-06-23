"""lodestone.retrieval.tokenize — shared lightweight text preprocessing pipeline.

This module provides pure, deterministic functions for converting raw text
into a bag-of-tokens suitable for sparse retrieval (BM25, RM3).

Pipeline
--------
1. Unicode NFKD normalisation and ASCII lowercasing.
2. Regex word tokenisation (alphanumeric sequences only).
3. Stopword removal using a hardcoded ~50-word English set.
4. Optional Porter-lite suffix-stripping stemmer.

All functions are stateless and side-effect-free.  They import only the
Python standard library (``re``, ``unicodedata``).

Porter-lite stemmer limitations
--------------------------------
The stemmer handles the most common English suffixes in a single pass:

- ``-ing``  : "running" → "runn"  (may over-stem; "ring" stays "ring")
- ``-ed``   : "walked"  → "walk"  (may over-stem short roots)
- ``-er``   : "faster"  → "fast"
- ``-ly``   : "quickly" → "quick"
- ``-ness`` : "darkness" → "dark"
- ``-tion`` / ``-sion`` : "action" → "act"
- ``-es``   : "watches" → "watch"  (strips only after ch/sh/s/x/z)
- ``-s``    : plain plural (only when > 4 chars after strip)

It is NOT a full Porter stemmer and will produce incorrect stems for
irregular forms.  Use it only when exact-match recall matters more than
precision.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Stopwords — 55 common English function words
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "then",
        "there",
        "up",
        "out",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "own",
        "same",
        "their",
        "our",
        "your",
        "his",
        "her",
        "my",
    }
)

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

# Matches sequences of Unicode word characters (letters + digits + underscore).
# The ``\w`` class is Unicode-aware by default in Python 3.
_WORD_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Unicode NFKD normalise *text* and return a lowercase ASCII-like string.

    Characters that cannot be ASCII-encoded after normalisation (e.g.
    accented letters whose base letter exists in ASCII) retain their
    normalised Unicode form — only the case is folded.

    Args:
        text: Raw input string.

    Returns:
        Lowercased, NFKD-normalised string.

    """
    return unicodedata.normalize("NFKD", text).lower()


def word_tokenize(text: str) -> list[str]:
    r"""Extract word tokens from *text* using a regex.

    Tokens are non-empty sequences of alphanumeric characters and
    underscores (``\w+``).  Punctuation, whitespace, and other
    non-word characters are treated as delimiters and discarded.

    Args:
        text: Input string (should already be lowercased / normalised).

    Returns:
        List of token strings; empty list for empty *text*.

    """
    return _WORD_RE.findall(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter stopwords from *tokens*.

    Uses the module-level :data:`STOPWORDS` frozenset.  Comparison is
    case-sensitive, so callers should lowercase tokens first (or use
    :func:`normalize` + :func:`word_tokenize`).

    Args:
        tokens: List of token strings.

    Returns:
        New list with stopword tokens removed; preserves order.

    """
    return [t for t in tokens if t not in STOPWORDS]


def _porter_lite_stem(token: str) -> str:
    """Apply a minimal Porter-inspired suffix stripper to *token*.

    This is NOT a full Porter stemmer.  It handles the most common
    English derivational suffixes in a single pass, checking that the
    resulting stem has at least 3 characters to avoid over-truncation.

    Rules (applied in order of specificity — longest match wins):

    1. ``-ness``  → strip if stem >= 4 chars
    2. ``-tion`` / ``-sion`` → strip if stem >= 3 chars
    3. ``-ing``  → strip if stem >= 3 chars; handle ``-e`` elision
    4. ``-ed``   → strip if stem >= 3 chars; handle ``-e`` elision
    5. ``-er``   → strip if stem >= 3 chars
    6. ``-ly``   → strip if stem >= 3 chars
    7. ``-ies``  → replace with ``-y`` if stem >= 3 chars
    8. ``-es``   → strip after sibilant (ch/sh/s/x/z) if stem >= 3 chars
    9. ``-s``    → strip if stem >= 4 chars (plain plural)

    Args:
        token: Single lowercase token.

    Returns:
        Stemmed token string (at least 3 characters).

    """
    if len(token) <= 3:
        return token

    # 1. -ness
    if token.endswith("ness") and len(token) - 4 >= 4:
        return token[:-4]

    # 2. -tion / -sion
    if token.endswith("tion") and len(token) - 4 >= 3:
        return token[:-4]
    if token.endswith("sion") and len(token) - 4 >= 3:
        return token[:-4]

    # 3. -ing  (e.g. "running" → "run", "taking" → "take")
    if token.endswith("ing") and len(token) - 3 >= 3:
        stem = token[:-3]
        # If the stem ends in a doubled consonant, remove one (runn → run)
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        # Restore -e only when the final pattern is vowel+consonant (VxC),
        # i.e. exactly one consonant after the last vowel.  This handles
        # "taking" → "take" but not "walking" → "walk" (two consonants: lk).
        if len(stem) >= 3 and stem[-1] not in "aeiou":
            # find position of last vowel
            last_vowel = max((i for i, c in enumerate(stem) if c in "aeiou"), default=-1)
            trailing_consonants = len(stem) - 1 - last_vowel
            if trailing_consonants == 1:
                return stem + "e"
        return stem

    # 4. -ed  (e.g. "walked" → "walk", "hoped" → "hope")
    if token.endswith("ed") and len(token) - 2 >= 3:
        stem = token[:-2]
        # doubled consonant: e.g. "stopped" → "stop"
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            return stem[:-1]
        # Restore -e only for VxC pattern (single trailing consonant after vowel).
        if len(stem) >= 3 and stem[-1] not in "aeiou":
            last_vowel = max((i for i, c in enumerate(stem) if c in "aeiou"), default=-1)
            trailing_consonants = len(stem) - 1 - last_vowel
            if trailing_consonants == 1:
                return stem + "e"
        return stem

    # 5. -er
    if token.endswith("er") and len(token) - 2 >= 3:
        return token[:-2]

    # 6. -ly
    if token.endswith("ly") and len(token) - 2 >= 3:
        return token[:-2]

    # 7. -ies → -y  (e.g. "countries" → "country")
    if token.endswith("ies") and len(token) - 3 >= 3:
        return token[:-3] + "y"

    # 8. -es after sibilant  (e.g. "watches" → "watch")
    if token.endswith("es") and len(token) - 2 >= 3:
        stem = token[:-2]
        if stem and stem[-1] in "sxz":
            return stem
        if stem.endswith("ch") or stem.endswith("sh"):
            return stem
        # Fall through to -s rule for plain -es

    # 9. plain -s  (e.g. "cats" → "cat")
    if token.endswith("s") and not token.endswith("ss") and len(token) - 1 >= 4:
        return token[:-1]

    return token


def tokenize(
    text: str,
    remove_stops: bool = True,
    stem: bool = False,
) -> list[str]:
    """Full text-preprocessing pipeline.

    Applies: normalise → word-tokenise → (optional) stopword removal
    → (optional) stemming.

    Args:
        text:          Raw input string.
        remove_stops:  If ``True`` (default), remove stopwords.
        stem:          If ``True``, apply Porter-lite suffix stripping.

    Returns:
        Ordered list of processed token strings.

    Example::

        >>> tokenize("The quick brown foxes are running")
        ['quick', 'brown', 'foxes', 'running']
        >>> tokenize("The quick brown foxes are running", stem=True)
        ['quick', 'brown', 'foxe', 'run']

    """
    tokens = word_tokenize(normalize(text))
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if stem:
        tokens = [_porter_lite_stem(t) for t in tokens]
    return tokens


__all__ = [
    "STOPWORDS",
    "normalize",
    "word_tokenize",
    "remove_stopwords",
    "tokenize",
    "_porter_lite_stem",  # exposed for unit testing
]
