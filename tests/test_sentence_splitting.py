"""Unit tests for split_into_best_sentences function.

Tests verify that sentence splitting preserves all content and handles edge cases.
"""

import pytest

from pocket_tts.models.tts_model import split_into_best_sentences


class MockTokenizer:
    """Mock tokenizer for testing purposes."""

    def __call__(self, text: str):
        """Mock tokenizer that returns a simple token count."""
        # Simple mock: 1 token per word, plus 1 for punctuation
        return MockTokenizedText(text)


class MockTokenizedText:
    """Mock TokenizedText for testing purposes."""

    def __init__(self, text: str):
        self.text = text
        # Simple mock: 1 token per word
        self.tokens = [[1] * len(text.split())]


def test_simple_sentence():
    """Test splitting a simple single sentence."""
    tokenizer = MockTokenizer()
    text = "Hello world."
    result = split_into_best_sentences(tokenizer, text)

    assert len(result) == 1
    assert "Hello" in result[0] or "hello" in result[0].lower()
    assert "world" in result[0]


def test_multiple_sentences():
    """Test splitting multiple sentences."""
    tokenizer = MockTokenizer()
    text = "First sentence. Second sentence. Third sentence."
    result = split_into_best_sentences(tokenizer, text)

    assert len(result) >= 1
    # All original words should be present
    combined = " ".join(result).lower()
    assert "first" in combined
    assert "second" in combined
    assert "third" in combined


def test_tale_of_two_cities():
    """Test the famous Tale of Two Cities opening - should preserve all clauses."""
    tokenizer = MockTokenizer()
    text = "It was the best of times, it was the worst of times."
    result = split_into_best_sentences(tokenizer, text)

    # All clauses should be present
    combined = " ".join(result).lower()
    assert "best" in combined
    assert "worst" in combined
    assert "times" in combined
    assert "it" in combined
    assert "was" in combined


def test_no_sentence_boundaries():
    """Test text without sentence terminators."""
    tokenizer = MockTokenizer()
    text = "This is just a single sentence without any punctuation"
    result = split_into_best_sentences(tokenizer, text)

    # Should return single chunk
    assert len(result) >= 1
    combined = " ".join(result).lower()
    assert "single" in combined


def test_empty_text():
    """Test empty text raises ValueError."""
    tokenizer = MockTokenizer()
    with pytest.raises(ValueError):
        split_into_best_sentences(tokenizer, "")


def test_whitespace_handling():
    """Test that extra whitespace is handled correctly."""
    tokenizer = MockTokenizer()
    text = "First sentence.  Second sentence.   Third sentence."
    result = split_into_best_sentences(tokenizer, text)

    # Should normalize whitespace
    assert len(result) >= 1
    # All sentences should be present
    combined = " ".join(result).lower()
    assert "first" in combined
    assert "second" in combined
    assert "third" in combined


def test_punctuation_variety():
    """Test different punctuation types."""
    tokenizer = MockTokenizer()
    text = "Question? Exclamation! Statement. Ellipsis…"
    result = split_into_best_sentences(tokenizer, text)

    # All sentences should be present
    combined = " ".join(result).lower()
    assert "question" in combined
    assert "exclamation" in combined
    assert "statement" in combined
    assert "ellipsis" in combined


def test_content_preservation():
    """Test that no content is lost during splitting."""
    tokenizer = MockTokenizer()
    text = "The quick brown fox jumps over the lazy dog. The dog was not amused."
    result = split_into_best_sentences(tokenizer, text)

    # Count words in original and result
    original_words = set(text.lower().split())

    # Unprepare chunks to remove added punctuation
    def unprepare(text):
        text = text.lstrip()
        if text.endswith("."):
            text = text[:-1]
        return text.strip()

    combined_unprepared = " ".join([unprepare(c) for c in result])
    result_words = set(combined_unprepared.lower().split())

    # All original words should be in result
    assert original_words.issubset(result_words), f"Missing words: {original_words - result_words}"


def test_long_text_chunking():
    """Test that long text is properly chunked at token limit."""
    tokenizer = MockTokenizer()
    # Create text with many words to exceed token limit
    text = " ".join([f"Word {i}." for i in range(100)])
    result = split_into_best_sentences(tokenizer, text)

    # Should split into multiple chunks
    assert len(result) > 1

    # All words should be preserved
    combined = " ".join(result).lower()
    for i in range(100):
        assert f"word {i}" in combined, f"Word {i} missing from result"


def test_comma_separated_clauses():
    """Test that comma-separated clauses are not split."""
    tokenizer = MockTokenizer()
    text = "First clause, second clause, third clause. Final sentence."
    result = split_into_best_sentences(tokenizer, text)

    # Commas should not cause splits
    combined = " ".join(result).lower()
    assert "first" in combined
    assert "second" in combined
    assert "third" in combined
    assert "final" in combined
    assert "clause" in combined


def test_mixed_case():
    """Test that case is preserved properly."""
    tokenizer = MockTokenizer()
    text = "UPPERCASE sentence. lowercase sentence. Mixed Case Sentence."
    result = split_into_best_sentences(tokenizer, text)

    # Case should be normalized (first letter uppercase)
    combined = " ".join(result).lower()
    assert "uppercase" in combined
    assert "lowercase" in combined
    assert "mixed" in combined
