"""Tests for the text splitting logic in split_into_best_sentences."""

import pytest

from pocket_tts.conditioners.text import get_default_tokenizer
from pocket_tts.models.tts_model import _protect_atoms, _restore_atoms, split_into_best_sentences


@pytest.fixture(scope="session")
def tokenizer():
    return get_default_tokenizer()


def test_short_text_single_chunk(tokenizer):
    """Short text should produce a single chunk."""
    chunks = split_into_best_sentences(
        tokenizer,
        "Hello world.",
        50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    assert len(chunks) == 1


def test_multiple_sentences_split(tokenizer):
    """Multiple sentences should be split when they exceed max_tokens."""
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = split_into_best_sentences(
        tokenizer, text, 10, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) > 1


def test_long_sentence_with_commas_is_split(tokenizer):
    """A long sentence with only commas (no periods) should be split on commas."""
    # This is the core bug from issue #38 - the Tale of Two Cities example
    text = (
        "It was the best of times, it was the worst of times, "
        "it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity, "
        "it was the season of Light, it was the season of Darkness, "
        "it was the spring of hope, it was the winter of despair"
    )
    chunks = split_into_best_sentences(
        tokenizer, text, 50, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) > 1, "Long comma-separated text should be split into multiple chunks"

    # Verify all content is preserved (no words should be lost in splitting)
    rejoined = " ".join(chunks).lower()
    for phrase in ["best of times", "worst of times", "age of foolishness", "winter of despair"]:
        assert phrase in rejoined, f"'{phrase}' should be preserved after splitting"


def test_long_sentence_with_commas_respects_max_tokens(tokenizer):
    """Each chunk from comma splitting should respect max_tokens (when possible)."""
    text = (
        "It was the best of times, it was the worst of times, "
        "it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity"
    )
    max_tokens = 20
    chunks = split_into_best_sentences(
        tokenizer, text, max_tokens, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    for chunk in chunks:
        token_count = len(tokenizer(chunk.strip()).tokens[0].tolist())
        # Allow some tolerance since comma clauses may vary in size
        assert token_count <= max_tokens * 2, (
            f"Chunk '{chunk[:50]}...' has {token_count} tokens, expected ~{max_tokens}"
        )


def test_mixed_sentences_and_commas(tokenizer):
    """Text with both sentence boundaries and long comma-separated clauses."""
    text = (
        "Short sentence. "
        "This is a very long sentence with many clauses, separated by commas, "
        "that goes on and on, and on some more, without any periods at all, "
        "until it finally reaches a period. "
        "Another short one."
    )
    chunks = split_into_best_sentences(
        tokenizer, text, 20, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) >= 3


def test_no_commas_no_periods_stays_single_chunk(tokenizer):
    """Text with no splitting characters stays as a single chunk."""
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = split_into_best_sentences(
        tokenizer, text, 5, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    # Should be 1 chunk since there are no split points
    assert len(chunks) == 1


def test_semicolons_and_colons_also_split(tokenizer):
    """Semicolons and colons should also serve as fallback split points."""
    text = (
        "First clause here; second clause here; third clause here; "
        "fourth clause here; fifth clause here; sixth clause here"
    )
    chunks = split_into_best_sentences(
        tokenizer, text, 15, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) > 1


def test_short_sentence_not_affected_by_comma_splitting(tokenizer):
    """Sentences under max_tokens should not be affected by comma logic."""
    text = "Hello, world."
    chunks = split_into_best_sentences(
        tokenizer, text, 50, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) == 1
    assert "hello" in chunks[0].lower()
    assert "world" in chunks[0].lower()


class TestProtectAtoms:
    """Unit tests for _protect_atoms / _restore_atoms (issue #162 + follow-up)."""

    # ---- decimals: period and comma forms ----

    def test_decimal_period_replaced(self):
        protected, atoms = _protect_atoms("98.6")
        assert "98.6" not in protected
        assert "98.6" in atoms.values()
        assert _restore_atoms(protected, atoms) == "98.6"

    def test_decimal_comma_replaced(self):
        # German-style decimal — protected regardless of language because the
        # splitter treats commas as fallback boundaries in every language.
        protected, atoms = _protect_atoms("37,0")
        assert "37,0" not in protected
        assert "37,0" in atoms.values()
        assert _restore_atoms(protected, atoms) == "37,0"

    def test_thousands_separator_replaced(self):
        # Both English-style ("1,000") and German-style ("1.000") thousands
        # must be opaque so the splitter doesn't break them apart.
        for raw in ("1,000", "1.000", "1,000.50", "1.000,50"):
            protected, atoms = _protect_atoms(raw)
            assert raw not in protected, f"{raw!r} should be replaced"
            assert _restore_atoms(protected, atoms) == raw

    def test_prose_period_untouched(self):
        protected, atoms = _protect_atoms("Hello world.")
        assert protected == "Hello world."
        assert atoms == {}

    def test_prose_comma_untouched(self):
        protected, atoms = _protect_atoms("Hello, world.")
        assert protected == "Hello, world."
        assert atoms == {}

    def test_integer_period_untouched(self):
        # "Section 4." has a period after a digit but no digit after.
        # Must remain a sentence boundary.
        protected, atoms = _protect_atoms("See section 4. It is important.")
        assert protected == "See section 4. It is important."

    # ---- URLs ----

    def test_url_with_scheme_replaced(self):
        text = "Visit https://example.com today"
        protected, atoms = _protect_atoms(text)
        assert "https://example.com" not in protected
        assert _restore_atoms(protected, atoms) == text

    def test_url_without_scheme_replaced(self):
        text = "Visit example.com today"
        protected, atoms = _protect_atoms(text)
        assert "example.com" not in protected
        assert _restore_atoms(protected, atoms) == text

    def test_subdomain_url_replaced(self):
        text = "See docs.python.org for help"
        protected, atoms = _protect_atoms(text)
        assert "docs.python.org" not in protected
        assert _restore_atoms(protected, atoms) == text

    # ---- emails ----

    def test_email_replaced(self):
        text = "Email me at alice@example.com please"
        protected, atoms = _protect_atoms(text)
        assert "alice@example.com" not in protected
        assert _restore_atoms(protected, atoms) == text

    def test_email_takes_precedence_over_url(self):
        # The URL pattern would otherwise match example.com inside the email.
        # Email must be detected first so the whole address becomes one atom.
        text = "alice@example.com"
        protected, atoms = _protect_atoms(text)
        assert text in atoms.values()
        assert len(atoms) == 1

    # ---- honorifics (per language) ----

    def test_english_honorific_replaced(self):
        text = "Dr. Smith arrived at noon."
        protected, atoms = _protect_atoms(text, language="english")
        assert "Dr." not in protected
        assert "Dr." in atoms.values()
        assert _restore_atoms(protected, atoms) == text

    def test_german_honorific_replaced(self):
        text = "Hr. Schmidt kam um zwölf."
        protected, atoms = _protect_atoms(text, language="german")
        assert "Hr." not in protected
        assert _restore_atoms(protected, atoms) == text

    def test_french_honorific_replaced(self):
        text = "M. Dupont est arrivé."
        protected, atoms = _protect_atoms(text, language="french")
        assert "M." not in protected
        assert _restore_atoms(protected, atoms) == text

    def test_24l_variant_uses_base_language_honorifics(self):
        text = "Dr. Smith arrived."
        protected, _ = _protect_atoms(text, language="english_24l")
        assert "Dr." not in protected

    # ---- abbreviations (per language) ----

    def test_english_abbreviation_replaced(self):
        text = "Apples, e.g. Granny Smith, are tasty."
        protected, atoms = _protect_atoms(text, language="english")
        assert "e.g." not in protected
        assert "e.g." in atoms.values()
        assert _restore_atoms(protected, atoms) == text

    def test_german_abbreviation_replaced(self):
        text = "Obst, z.B. Äpfel, schmecken gut."
        protected, _ = _protect_atoms(text, language="german")
        assert "z.B." not in protected

    # ---- multiple atoms in one text ----

    def test_multiple_atoms_all_protected_and_restored(self):
        text = (
            "Dr. Smith said it's 98.6°F at 3.14 PM, "
            "see https://example.com or email alice@example.com for details."
        )
        protected, atoms = _protect_atoms(text, language="english")
        assert "Dr." not in protected
        assert "98.6" not in protected
        assert "3.14" not in protected
        assert "https://example.com" not in protected
        assert "alice@example.com" not in protected
        assert _restore_atoms(protected, atoms) == text

    # ---- placeholders are safe for the splitter ----

    def test_placeholders_contain_no_boundary_chars(self):
        text = "Dr. Smith says it's 98.6°F, see example.com."
        _, atoms = _protect_atoms(text, language="english")
        for placeholder in atoms:
            for ch in ".,;:!?":
                assert ch not in placeholder, (
                    f"Placeholder {placeholder!r} must not contain boundary char {ch!r}"
                )

    # ---- empty / no-op cases ----

    def test_empty_text_returns_empty(self):
        protected, atoms = _protect_atoms("")
        assert protected == ""
        assert atoms == {}

    def test_text_with_no_atoms(self):
        text = "Just regular words here."
        protected, atoms = _protect_atoms(text)
        assert protected == text
        assert atoms == {}

    def test_restore_handles_no_atoms(self):
        assert _restore_atoms("Just regular text.", {}) == "Just regular text."


def test_decimal_not_split_into_separate_chunks(tokenizer):
    """Decimals must not be treated as sentence boundaries (issue #162).

    The mask-based splitter wraps decimals in placeholders that contain no
    boundary characters, runs the splitter, then restores the originals.
    The decimal is preserved literally in the output chunk.
    """
    text = "The average human body temperature is 98.6°F, which is normal."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    assert len(chunks) == 1
    assert "98.6" in chunks[0]


def test_german_decimal_comma_not_split(tokenizer):
    """German decimal-comma (memchr's case) must survive splitting in long sentences."""
    text = (
        "Die durchschnittliche Körpertemperatur eines erwachsenen Menschen "
        "beträgt 37,0 Grad Celsius, was als normal angesehen wird, "
        "während eine Temperatur von 38,5 Grad bereits als Fieber gilt, "
        "und eine Temperatur über 40,0 Grad ist gefährlich."
    )
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=30,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
        language="german",
    )
    rejoined = " ".join(chunks)
    assert "37,0" in rejoined
    assert "38,5" in rejoined
    assert "40,0" in rejoined
    # No chunk fragments a decimal across a boundary.
    for chunk in chunks:
        assert not chunk.rstrip().endswith(("37,", "38,", "40,"))


def test_honorific_not_split(tokenizer):
    """A sentence with 'Dr.' must not be split between the honorific and the name."""
    text = "Dr. Smith and Dr. Jones discussed the patient. They agreed on the diagnosis."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=15,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    for chunk in chunks:
        assert not chunk.rstrip().endswith("Dr.")


def test_url_not_split(tokenizer):
    """A URL with internal dots must not be split inside the domain."""
    text = "Please visit https://example.com for more information about the product."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    rejoined = " ".join(chunks)
    assert "https://example.com" in rejoined


def test_english_thousands_not_split(tokenizer):
    """English '1,000' (thousands separator) must not be split in long sentences."""
    text = (
        "The crowd of 1,000 people gathered at the square, "
        "the police estimated 2,500 attendees, "
        "while the organizers claimed 5,000 supporters showed up."
    )
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=20,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    rejoined = " ".join(chunks)
    assert "1,000" in rejoined
    assert "2,500" in rejoined
    assert "5,000" in rejoined
    for chunk in chunks:
        assert not chunk.rstrip().endswith(("1,", "2,", "5,"))


def test_multiple_decimals_preserved(tokenizer):
    """Multiple decimals in one sentence are all preserved literally."""
    text = "Pi is 3.14 and e is 2.718."
    chunks = split_into_best_sentences(
        tokenizer,
        text,
        max_tokens=50,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    assert len(chunks) == 1
    assert "3.14" in chunks[0]
    assert "2.718" in chunks[0]


def test_empty_string_raises(tokenizer):
    """Empty input should raise ValueError from prepare_text_prompt."""
    with pytest.raises(ValueError, match="empty"):
        split_into_best_sentences(
            tokenizer, "", 50, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
        )


def test_oversized_clause_without_commas_still_returns(tokenizer):
    """A long clause with no split points should still be returned (not dropped)."""
    # 20 words with no punctuation at all - no way to split
    text = " ".join(f"word{i}" for i in range(20))
    chunks = split_into_best_sentences(
        tokenizer, text, 5, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    assert len(chunks) == 1
    # prepare_text_prompt capitalizes the first char and adds a trailing period,
    # so compare case-insensitively and strip punctuation
    assert chunks[0].lower().rstrip(".") == text.lower()
