"""
Tests for lodestone.retrieval.tokenize.

Covers:
- normalize: Unicode NFKD, lowercasing.
- word_tokenize: alphanumeric splitting, punctuation stripping.
- remove_stopwords: stopword removal, case sensitivity.
- _porter_lite_stem: documented suffix rules (tested against actual behavior).
- tokenize: full pipeline, stemming flag, determinism.
"""

from __future__ import annotations

import pytest

from lodestone.retrieval.tokenize import (
    STOPWORDS,
    _porter_lite_stem,
    normalize,
    remove_stopwords,
    tokenize,
    word_tokenize,
)

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases_ascii(self):
        assert normalize("Hello World") == "hello world"

    def test_all_uppercase(self):
        assert normalize("PYTHON") == "python"

    def test_already_lower(self):
        assert normalize("already") == "already"

    def test_unicode_nfkd_accented_e(self):
        # é → NFKD decomposes to 'e' + combining accent → lowercased
        result = normalize("café")
        assert result.startswith("cafe") or "cafe" in result

    def test_unicode_nfkd_normalisation(self):
        # Full-width characters → ASCII equivalents after NFKD
        full_width_a = "Ａ"  # FULLWIDTH LATIN CAPITAL LETTER A
        result = normalize(full_width_a)
        assert result == "a"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_numbers_preserved(self):
        assert normalize("Version 2.0") == "version 2.0"

    def test_mixed_unicode_and_ascii(self):
        result = normalize("Über fast")
        assert "fast" in result
        assert result == result.lower()


# ---------------------------------------------------------------------------
# word_tokenize
# ---------------------------------------------------------------------------


class TestWordTokenize:
    def test_basic_split(self):
        assert word_tokenize("hello world") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert word_tokenize("hello, world!") == ["hello", "world"]

    def test_hyphen_splits(self):
        # Hyphens are not \w characters, so they split
        result = word_tokenize("well-known")
        assert "well" in result and "known" in result

    def test_empty_string(self):
        assert word_tokenize("") == []

    def test_only_punctuation(self):
        assert word_tokenize("...!!!???") == []

    def test_numbers_are_tokens(self):
        result = word_tokenize("version 2 point 0")
        assert "2" in result

    def test_underscore_included(self):
        # \w+ includes underscore
        result = word_tokenize("snake_case_variable")
        assert "snake_case_variable" in result

    def test_preserves_order(self):
        tokens = word_tokenize("cat dog bird fish")
        assert tokens == ["cat", "dog", "bird", "fish"]


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------


class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        tokens = ["the", "quick", "brown", "fox"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_preserves_order(self):
        tokens = ["cat", "is", "a", "small", "animal"]
        result = remove_stopwords(tokens)
        assert result == ["cat", "small", "animal"]

    def test_empty_list(self):
        assert remove_stopwords([]) == []

    def test_all_stopwords(self):
        tokens = ["the", "a", "an", "and", "or"]
        assert remove_stopwords(tokens) == []

    def test_no_stopwords(self):
        tokens = ["python", "machine", "learning"]
        assert remove_stopwords(tokens) == tokens

    def test_case_sensitive(self):
        # STOPWORDS are lowercase; "The" is NOT a stopword (case-sensitive)
        tokens = ["The", "quick", "the"]
        result = remove_stopwords(tokens)
        assert "The" in result  # capital T → not in stopwords
        assert "the" not in result  # lowercase → removed

    def test_stopwords_frozenset_contains_common_words(self):
        for word in ("a", "the", "and", "or", "is", "in", "of", "to", "it"):
            assert word in STOPWORDS


# ---------------------------------------------------------------------------
# _porter_lite_stem — tested against actual implementation behaviour
# ---------------------------------------------------------------------------


class TestPorterLiteStem:
    # Documented in module docstring (tested against actual output)
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("running", "run"),  # -ing with doubled consonant nn→n
            ("walked", "walk"),  # -ed
            ("faster", "fast"),  # -er
            ("quickly", "quick"),  # -ly
            ("darkness", "dark"),  # -ness
            ("watches", "watch"),  # -es after ch (sibilant)
        ],
    )
    def test_documented_cases(self, token, expected):
        assert _porter_lite_stem(token) == expected

    # Cases where the docstring differs from reality (noted as doc bugs)
    # The actual implementation: foxes→fox (via -s rule, not -e elision)
    def test_foxes_stems_to_fox(self):
        assert _porter_lite_stem("foxes") == "fox"

    # action: ends in -tion but stem would be 'ac' (len 2 < 3) → no strip
    def test_action_unchanged(self):
        assert _porter_lite_stem("action") == "action"

    def test_traction_strips_tion(self):
        # 'traction' → 'trac' (stem 'trac' len 4 >= 3)
        assert _porter_lite_stem("traction") == "trac"

    def test_short_token_unchanged(self):
        # <= 3 chars → returned as-is
        assert _porter_lite_stem("run") == "run"
        assert _porter_lite_stem("go") == "go"
        assert _porter_lite_stem("is") == "is"

    def test_no_over_truncation(self):
        # Result must be >= 3 chars for longer inputs
        for token in ["running", "walked", "faster", "quickly", "darkness"]:
            stemmed = _porter_lite_stem(token)
            assert len(stemmed) >= 3, f"Stem of {token!r} is too short: {stemmed!r}"

    def test_countries_ies_to_y(self):
        # -ies → -y: 'countries' → 'country'
        assert _porter_lite_stem("countries") == "country"

    def test_plain_s_plural(self):
        # -s strip only when the resulting stem has >= 4 chars.
        # 'cats'[:-1] = 'cat' which has len 3 < 4, so 'cats' is NOT stripped.
        # (See test_cats_not_stripped below.)
        # 'items'[:-1] = 'item' which has len 4 >= 4, so it IS stripped.
        assert _porter_lite_stem("cats") == "cats"  # stem 'cat' len 3 < 4 → unchanged

    def test_plain_s_strip_condition(self):
        # stem must be >= 4 chars: 'dogs' → stem 'dog' len 3 < 4 → unchanged
        # 'items' → stem 'item' len 4 >= 4 → 'item'
        assert _porter_lite_stem("items") == "item"

    def test_cats_not_stripped(self):
        # 'cats' -> strip -s gives 'cat' (len 3), 3 < 4 → unchanged
        assert _porter_lite_stem("cats") == "cats"

    def test_deterministic(self):
        tokens = ["running", "walked", "faster", "learning", "neural", "networks"]
        results1 = [_porter_lite_stem(t) for t in tokens]
        results2 = [_porter_lite_stem(t) for t in tokens]
        assert results1 == results2


# ---------------------------------------------------------------------------
# tokenize (full pipeline)
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_default_pipeline_removes_stops(self):
        # From the docstring (actual output verified):
        result = tokenize("The quick brown foxes are running")
        assert result == ["quick", "brown", "foxes", "running"]

    def test_stem_pipeline(self):
        # With stem=True (actual output verified):
        result = tokenize("The quick brown foxes are running", stem=True)
        assert result == ["quick", "brown", "fox", "run"]

    def test_no_stop_removal(self):
        result = tokenize("the cat sat", remove_stops=False)
        assert "the" in result
        assert "cat" in result

    def test_empty_string(self):
        assert tokenize("") == []

    def test_all_stopwords_returns_empty(self):
        result = tokenize("the and or is are")
        assert result == []

    def test_unicode_normalised(self):
        # NFKD normalisation decomposes 'Ü' into 'u' + combining diaeresis.
        # The \w+ regex does not match combining diaeritic characters, so the
        # decomposed form yields two tokens: ['u', 'ber'].
        result = tokenize("Über", remove_stops=False)
        assert len(result) == 2
        assert result == ["u", "ber"]
        # All tokens are lowercase
        for token in result:
            assert token == token.lower()

    def test_deterministic(self):
        text = "Machine learning models require gradient descent optimisation."
        r1 = tokenize(text)
        r2 = tokenize(text)
        assert r1 == r2

    def test_punctuation_removed(self):
        result = tokenize("Hello, world! How are you?", remove_stops=False)
        for token in result:
            assert token.isalnum() or "_" in token

    @pytest.mark.parametrize(
        "text,remove_stops,stem",
        [
            ("Neural networks backpropagate gradients", True, False),
            ("Neural networks backpropagate gradients", True, True),
            ("Neural networks backpropagate gradients", False, False),
            ("Neural networks backpropagate gradients", False, True),
        ],
    )
    def test_parametrized_combinations(self, text, remove_stops, stem):
        result = tokenize(text, remove_stops=remove_stops, stem=stem)
        # Just ensure it doesn't raise and returns a list
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)
