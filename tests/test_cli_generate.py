"""Integration tests for the CLI generate command using real implementation."""

import os

import pytest
from typer.testing import CliRunner

from pocket_tts.data.audio import audio_read
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.main import cli_app

other_voice = "https://huggingface.co/kyutai/tts-voices/resolve/main/expresso/ex01-ex02_default_001_channel1_168s.wav"

runner = CliRunner()

IS_CI = os.environ.get("CI") == "true"
CI_SKIP_REASON = "Voice cloning is not publicly available, skipping in the CI"


def test_generate_basic_usage(tmp_path):
    """Test basic generate command with default parameters."""
    output_file = tmp_path / "test_output.wav"

    result = runner.invoke(
        cli_app,
        ["generate", "--text", "Hello world, this is a test.", "--output-path", str(output_file)],
    )

    assert result.exit_code == 0
    # Verify output file was created and contains audio
    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Verify it's a valid audio file
    audio, sample_rate = audio_read(str(output_file))
    assert audio.shape[0] == 1  # Mono channel
    assert audio.shape[1] > 0  # Has audio samples
    assert sample_rate == 24000  # Expected sample rate


@pytest.mark.skipif(IS_CI, reason=CI_SKIP_REASON)
def test_generate_with_custom_voice(tmp_path):
    """Test generate command with custom voice prompt."""
    output_file = tmp_path / "custom_voice_test.wav"

    result = runner.invoke(
        cli_app,
        [
            "generate",
            "--text",
            "Testing custom voice.",
            "--voice",
            other_voice,
            "--output-path",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    # Verify audio content
    audio, sample_rate = audio_read(str(output_file))
    assert audio.shape[0] == 1  # Mono channel
    assert audio.shape[1] > 0  # Has audio samples
    assert sample_rate == 24000


def test_generate_with_custom_parameters(tmp_path):
    """Test generate command with custom generation parameters."""
    output_file = tmp_path / "custom_params_test.wav"

    result = runner.invoke(
        cli_app,
        [
            "generate",
            "--text",
            "Testing custom parameters.",
            "--variant",
            DEFAULT_VARIANT,
            "--temperature",
            "0.8",
            "--lsd-decode-steps",
            "2",
            "--eos-threshold",
            "-3.0",
            "--frames-after-eos",
            "7",
            "--output-path",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    audio, sample_rate = audio_read(str(output_file))
    assert audio.shape[0] == 1  # Mono channel
    assert audio.shape[1] > 0  # Has audio samples
    assert sample_rate == 24000


def test_generate_verbose_mode(tmp_path):
    """Test generate command with verbose logging."""
    output_file = tmp_path / "verbose_test.wav"

    result = runner.invoke(
        cli_app,
        ["generate", "--text", "Testing verbose mode.", "-q", "--output-path", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()


def test_generate_default_text(tmp_path):
    """Test generate command with default text when no text provided."""
    output_file = tmp_path / "default_text_test.wav"

    result = runner.invoke(cli_app, ["generate", "--output-path", str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()

    audio, sample_rate = audio_read(str(output_file))
    assert audio.shape[0] == 1  # Mono channel
    assert audio.shape[1] > 0  # Has audio samples
    assert sample_rate == 24000


def test_generate_long_text(tmp_path):
    """Test generate command with longer text."""
    long_text = "This is a longer text to test the TTS system. " * 5
    output_file = tmp_path / "long_text_test.wav"

    result = runner.invoke(
        cli_app, ["generate", "--text", long_text, "--output-path", str(output_file)]
    )

    assert result.exit_code == 0
    assert output_file.exists()

    audio, sample_rate = audio_read(str(output_file))
    assert audio.shape[0] == 1  # Mono channel
    assert audio.shape[1] > 0  # Has audio samples
    assert sample_rate == 24000
    # Longer text should produce longer audio
    assert audio.shape[1] > 24000 * 10  # At least 10 second of audio


def test_generate_multiple_runs(tmp_path):
    """Test multiple consecutive generate commands."""
    for i in range(3):
        output_file = tmp_path / f"test_run_{i + 1}.wav"

        result = runner.invoke(
            cli_app,
            [
                "generate",
                "--text",
                f"This is test run number {i + 1}.",
                "--output-path",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        audio, sample_rate = audio_read(str(output_file))
        assert audio.shape[0] == 1  # Mono channel
        assert audio.shape[1] > 0  # Has audio samples
        assert sample_rate == 24000
