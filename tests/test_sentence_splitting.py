# Tests for the sentence splitting fix (Issue #3)
# Run with: pytest tests/test_sentence_splitting.py -v

from unittest.mock import MagicMock

import pytest


class MockTokenizer:
    """Mock tokenizer that simulates the real tokenizer behavior."""

    def __init__(self):
        self.sp = MagicMock()

    def __call__(self, text: str):
        """Return mock tokens - count based on word count for simplicity."""
        # Simulate ~1.5 tokens per word on average
        word_count = len(text.split())
        token_count = max(1, int(word_count * 1.5))
        mock_result = MagicMock()
        mock_result.tokens = [MagicMock()]
        mock_result.tokens[0].tolist.return_value = list(range(token_count))
        return mock_result


@pytest.fixture
def tokenizer():
    return MockTokenizer()


class TestSplitIntoBestSentences:
    """Tests for the split_into_best_sentences function."""

    def test_tale_of_two_cities_no_skipping(self, tokenizer):
        """
        Test case from Issue #3 - should not skip 'age of foolishness'.

        The original bug caused comma-separated clauses to be dropped
        when using token-based decoding.
        """
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = (
            "It was the best of times, it was the worst of times, "
            "it was the age of wisdom, it was the age of foolishness, "
            "it was the epoch of belief, it was the epoch of incredulity."
        )

        chunks = split_into_best_sentences(tokenizer, text)
        combined = " ".join(chunks)

        # These specific phrases were being dropped in the original bug
        assert "age of foolishness" in combined.lower()
        assert "age of wisdom" in combined.lower()
        assert "epoch of belief" in combined.lower()
        assert "epoch of incredulity" in combined.lower()

    def test_no_content_loss_simple(self, tokenizer):
        """All words from input should appear in output."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = "One two three. Four five six. Seven eight nine."

        chunks = split_into_best_sentences(tokenizer, text)
        combined = " ".join(chunks)

        # Normalize for comparison (remove punctuation, lowercase)
        import re

        original_words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        combined_words = set(re.sub(r"[^\w\s]", "", combined.lower()).split())

        assert original_words == combined_words, f"Missing words: {original_words - combined_words}"

    def test_no_content_loss_with_commas(self, tokenizer):
        """Comma-separated clauses should all be preserved."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = "Alpha, beta, gamma, delta. Epsilon, zeta, eta, theta."

        chunks = split_into_best_sentences(tokenizer, text)
        combined = " ".join(chunks)

        for word in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]:
            assert word in combined.lower(), f"Missing word: {word}"

    def test_single_sentence(self, tokenizer):
        """Single sentence without punctuation should work."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = "This is a single sentence without ending punctuation"

        chunks = split_into_best_sentences(tokenizer, text)

        assert len(chunks) >= 1
        combined = " ".join(chunks)
        # Should preserve the text (prepare_text_prompt will add period)
        assert "single sentence" in combined.lower()

    def test_multiple_sentences(self, tokenizer):
        """Multiple sentences should be properly chunked."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = "First sentence. Second sentence. Third sentence."

        chunks = split_into_best_sentences(tokenizer, text)

        combined = " ".join(chunks)
        assert "first" in combined.lower()
        assert "second" in combined.lower()
        assert "third" in combined.lower()

    def test_exclamation_and_question_marks(self, tokenizer):
        """Exclamation and question marks should work as sentence boundaries."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = "Hello! How are you? I am fine."

        chunks = split_into_best_sentences(tokenizer, text)

        combined = " ".join(chunks)
        assert "hello" in combined.lower()
        assert "how are you" in combined.lower()
        assert "i am fine" in combined.lower()

    def test_ellipsis(self, tokenizer):
        """Ellipsis should work as sentence boundary."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        text = "Wait for it… Here it comes. And done!"

        chunks = split_into_best_sentences(tokenizer, text)

        combined = " ".join(chunks)
        assert "wait" in combined.lower()
        assert "here it comes" in combined.lower()
        assert "done" in combined.lower()

    def test_empty_text_raises(self, tokenizer):
        """Empty text should raise ValueError (from prepare_text_prompt)."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        with pytest.raises(ValueError):
            split_into_best_sentences(tokenizer, "")

    def test_whitespace_only_raises(self, tokenizer):
        """Whitespace-only text should raise ValueError."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        with pytest.raises(ValueError):
            split_into_best_sentences(tokenizer, "   ")

    def test_long_text_chunking(self, tokenizer):
        """Long text should be split into multiple chunks."""
        from pocket_tts.models.tts_model import split_into_best_sentences

        # Create a long text with many sentences
        sentences = [f"This is sentence number {i}." for i in range(20)]
        text = " ".join(sentences)

        chunks = split_into_best_sentences(tokenizer, text)

        # Should have multiple chunks
        assert len(chunks) > 1

        # All content should be preserved
        combined = " ".join(chunks)
        for i in range(20):
            assert f"sentence number {i}" in combined.lower() or f"sentence number {i}" in combined


class TestPrepareTextPrompt:
    """Tests for the prepare_text_prompt function."""

    def test_adds_period_if_missing(self):
        """Text ending in alphanumeric should get a period added."""
        from pocket_tts.models.tts_model import prepare_text_prompt

        text, _ = prepare_text_prompt("Hello world")
        assert text.endswith(".")

    def test_capitalizes_first_letter(self):
        """First letter should be capitalized."""
        from pocket_tts.models.tts_model import prepare_text_prompt

        text, _ = prepare_text_prompt("hello world")
        assert text.lstrip()[0].isupper()

    def test_preserves_existing_punctuation(self):
        """Existing trailing punctuation should be preserved."""
        from pocket_tts.models.tts_model import prepare_text_prompt

        text, _ = prepare_text_prompt("Hello world!")
        assert text.endswith("!")
        assert not text.endswith("!.")

    def test_short_text_padding(self):
        """Short text should get padding for better generation."""
        from pocket_tts.models.tts_model import prepare_text_prompt

        text, _ = prepare_text_prompt("Hi")
        # Short text gets spaces prepended
        assert len(text) > 2
