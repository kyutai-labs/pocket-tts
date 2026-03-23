"""Tests for the text splitting logic in split_into_best_sentences."""

import pytest

from pocket_tts.conditioners.text import get_default_tokenizer
from pocket_tts.models.tts_model import split_into_best_sentences


@pytest.fixture(scope="session")
def tokenizer():
    return get_default_tokenizer()


def test_short_text_single_chunk(tokenizer):
    """Short text should produce a single chunk."""
    chunks = split_into_best_sentences(tokenizer, "Hello world.", 50)
    assert len(chunks) == 1


def test_multiple_sentences_split(tokenizer):
    """Multiple sentences should be split when they exceed max_tokens."""
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = split_into_best_sentences(tokenizer, text, 10)
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
    chunks = split_into_best_sentences(tokenizer, text, 50)
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
    chunks = split_into_best_sentences(tokenizer, text, max_tokens)
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
    chunks = split_into_best_sentences(tokenizer, text, 20)
    assert len(chunks) >= 3


def test_no_commas_no_periods_stays_single_chunk(tokenizer):
    """Text with no splitting characters stays as a single chunk."""
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = split_into_best_sentences(tokenizer, text, 5)
    # Should be 1 chunk since there are no split points
    assert len(chunks) == 1


def test_semicolons_and_colons_also_split(tokenizer):
    """Semicolons and colons should also serve as fallback split points."""
    text = (
        "First clause here; second clause here; third clause here; "
        "fourth clause here; fifth clause here; sixth clause here"
    )
    chunks = split_into_best_sentences(tokenizer, text, 15)
    assert len(chunks) > 1


def test_short_sentence_not_affected_by_comma_splitting(tokenizer):
    """Sentences under max_tokens should not be affected by comma logic."""
    text = "Hello, world."
    chunks = split_into_best_sentences(tokenizer, text, 50)
    assert len(chunks) == 1
    assert "hello" in chunks[0].lower()
    assert "world" in chunks[0].lower()


def test_empty_string_raises(tokenizer):
    """Empty input should raise ValueError from prepare_text_prompt."""
    with pytest.raises(ValueError, match="empty"):
        split_into_best_sentences(tokenizer, "", 50)


def test_oversized_clause_without_commas_still_returns(tokenizer):
    """A long clause with no split points should still be returned (not dropped)."""
    # 20 words with no punctuation at all - no way to split
    text = " ".join(f"word{i}" for i in range(20))
    chunks = split_into_best_sentences(tokenizer, text, 5)
    assert len(chunks) == 1
    # prepare_text_prompt capitalizes the first char and adds a trailing period,
    # so compare case-insensitively and strip punctuation
    assert chunks[0].lower().rstrip(".") == text.lower()
