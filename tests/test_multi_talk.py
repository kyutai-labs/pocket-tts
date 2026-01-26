"""Tests for multi-talk functionality."""

import pytest

from pocket_tts.main import parse_script


class TestParseScript:
    """Tests for the parse_script function."""

    def test_basic_two_speakers(self):
        """Test parsing a basic script with two speakers."""
        script = "{Alice} Hello Bob! {Bob} Hi Alice!"
        speakers = ["Alice", "Bob"]

        segments = parse_script(script, speakers)

        assert len(segments) == 2
        assert segments[0].speaker_name == "Alice"
        assert segments[0].text == "Hello Bob!"
        assert segments[1].speaker_name == "Bob"
        assert segments[1].text == "Hi Alice!"

    def test_multiline_script(self):
        """Test parsing a multiline script."""
        script = """{Alice} Hello there!
{Bob} How are you doing today?
{Alice} I'm doing great, thanks for asking!"""
        speakers = ["Alice", "Bob"]

        segments = parse_script(script, speakers)

        assert len(segments) == 3
        assert segments[0].speaker_name == "Alice"
        assert segments[0].text == "Hello there!"
        assert segments[1].speaker_name == "Bob"
        assert segments[1].text == "How are you doing today?"
        assert segments[2].speaker_name == "Alice"
        assert segments[2].text == "I'm doing great, thanks for asking!"

    def test_case_insensitive_speaker_names(self):
        """Test that speaker names are matched case-insensitively."""
        script = "{alice} Hello! {ALICE} Goodbye!"
        speakers = ["Alice"]

        segments = parse_script(script, speakers)

        assert len(segments) == 2
        assert segments[0].speaker_name == "Alice"
        assert segments[1].speaker_name == "Alice"

    def test_unknown_speaker_raises_error(self):
        """Test that unknown speaker names raise a ValueError."""
        script = "{Alice} Hello! {Charlie} Hi!"
        speakers = ["Alice", "Bob"]

        with pytest.raises(ValueError) as exc_info:
            parse_script(script, speakers)

        assert "Unknown speaker" in str(exc_info.value)
        assert "Charlie" in str(exc_info.value)

    def test_empty_text_segments_skipped(self):
        """Test that empty text segments are skipped."""
        script = "{Alice} {Bob} Hello!"
        speakers = ["Alice", "Bob"]

        segments = parse_script(script, speakers)

        assert len(segments) == 1
        assert segments[0].speaker_name == "Bob"
        assert segments[0].text == "Hello!"

    def test_whitespace_handling(self):
        """Test that whitespace is properly trimmed."""
        script = "  {Alice}   Hello there!   {Bob}    Hi!   "
        speakers = ["Alice", "Bob"]

        segments = parse_script(script, speakers)

        assert len(segments) == 2
        assert segments[0].text == "Hello there!"
        assert segments[1].text == "Hi!"

    def test_no_tags_returns_empty(self):
        """Test that a script without tags returns empty list."""
        script = "Hello there! How are you?"
        speakers = ["Alice", "Bob"]

        segments = parse_script(script, speakers)

        assert len(segments) == 0

    def test_single_speaker(self):
        """Test parsing a script with a single speaker."""
        script = "{Narrator} Once upon a time, there was a story."
        speakers = ["Narrator"]

        segments = parse_script(script, speakers)

        assert len(segments) == 1
        assert segments[0].speaker_name == "Narrator"
        assert segments[0].text == "Once upon a time, there was a story."

    def test_speaker_with_spaces_in_name(self):
        """Test parsing a speaker name with spaces."""
        script = "{Alice Smith} Hello! {Bob Jones} Hi!"
        speakers = ["Alice Smith", "Bob Jones"]

        segments = parse_script(script, speakers)

        assert len(segments) == 2
        assert segments[0].speaker_name == "Alice Smith"
        assert segments[1].speaker_name == "Bob Jones"
