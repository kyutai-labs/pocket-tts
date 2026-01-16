"""Tests for the Python API, including audio conversion functionality."""

import subprocess

import pytest

from pocket_tts.data.audio import audio_read, convert_audio_to_wav


def check_ffmpeg_available():
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def sample_wav_file(tmp_path):
    """Create a sample WAV file for testing."""
    # Download a test WAV file or create one using ffmpeg
    # For simplicity, we'll create a test tone using ffmpeg if available
    wav_file = tmp_path / "test_audio.wav"

    if check_ffmpeg_available():
        # Create a 1-second 440Hz sine wave at 24kHz, mono, 16-bit
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-ar",
                "24000",
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                "-acodec",
                "pcm_s16le",
                str(wav_file),
            ],
            capture_output=True,
            check=True,
        )
    else:
        # If ffmpeg is not available, create a minimal WAV file manually
        import wave

        with wave.open(str(wav_file), "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # 24kHz
            # Write 1 second of silence
            wf.writeframes(b"\x00\x00" * 24000)

    return wav_file


@pytest.fixture
def sample_m4a_file(tmp_path, sample_wav_file):
    """Create a sample M4A file by converting the WAV file."""
    if not check_ffmpeg_available():
        pytest.skip("ffmpeg not available, cannot create M4A test file")

    m4a_file = tmp_path / "test_audio.m4a"

    # Convert WAV to M4A
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(sample_wav_file),
            "-acodec",
            "aac",
            "-b:a",
            "128k",
            str(m4a_file),
        ],
        capture_output=True,
        check=True,
    )

    return m4a_file


@pytest.fixture
def sample_mp3_file(tmp_path, sample_wav_file):
    """Create a sample MP3 file by converting the WAV file."""
    if not check_ffmpeg_available():
        pytest.skip("ffmpeg not available, cannot create MP3 test file")

    mp3_file = tmp_path / "test_audio.mp3"

    # Convert WAV to MP3
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(sample_wav_file),
            "-acodec",
            "libmp3lame",
            "-b:a",
            "128k",
            str(mp3_file),
        ],
        capture_output=True,
        check=True,
    )

    return mp3_file


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
class TestFfmpegConversion:
    """Tests for ffmpeg-based audio conversion."""

    def test_convert_wav_to_wav(self, sample_wav_file, tmp_path):
        """Test that WAV files are returned as-is when no output path specified."""
        result = convert_audio_to_wav(sample_wav_file)
        assert result == sample_wav_file
        assert result.exists()

    def test_convert_wav_to_wav_with_output(self, sample_wav_file, tmp_path):
        """Test converting WAV to WAV with specified output path."""
        output_file = tmp_path / "output.wav"
        result = convert_audio_to_wav(sample_wav_file, output_file)

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_convert_m4a_to_wav(self, sample_m4a_file, tmp_path):
        """Test converting M4A file to WAV format."""
        output_file = tmp_path / "converted.wav"
        result = convert_audio_to_wav(sample_m4a_file, output_file)

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify it's a valid WAV file with correct format
        audio, sample_rate = audio_read(output_file)
        assert audio.shape[0] == 1  # Mono
        assert audio.shape[1] > 0  # Has samples
        assert sample_rate == 24000  # 24kHz

    def test_convert_mp3_to_wav(self, sample_mp3_file, tmp_path):
        """Test converting MP3 file to WAV format."""
        output_file = tmp_path / "converted_mp3.wav"
        result = convert_audio_to_wav(sample_mp3_file, output_file)

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify it's a valid WAV file with correct format
        audio, sample_rate = audio_read(output_file)
        assert audio.shape[0] == 1  # Mono
        assert audio.shape[1] > 0  # Has samples
        assert sample_rate == 24000  # 24kHz

    def test_convert_with_temp_output(self, sample_m4a_file):
        """Test conversion with automatic temporary output file."""
        result = convert_audio_to_wav(sample_m4a_file)

        assert result.exists()
        assert result.suffix == ".wav"
        assert result.stat().st_size > 0

        # Verify it's a valid WAV file
        audio, sample_rate = audio_read(result)
        assert audio.shape[0] == 1  # Mono
        assert sample_rate == 24000  # 24kHz

        # Clean up
        result.unlink()

    def test_convert_nonexistent_file(self, tmp_path):
        """Test that converting a non-existent file raises FileNotFoundError."""
        fake_file = tmp_path / "nonexistent.m4a"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            convert_audio_to_wav(fake_file)

    def test_audio_read_with_m4a(self, sample_m4a_file):
        """Test that audio_read automatically converts M4A files."""
        audio, sample_rate = audio_read(sample_m4a_file)

        assert audio.shape[0] == 1  # Mono
        assert audio.shape[1] > 0  # Has samples
        assert sample_rate == 24000  # 24kHz

    def test_audio_read_with_mp3(self, sample_mp3_file):
        """Test that audio_read automatically converts MP3 files."""
        audio, sample_rate = audio_read(sample_mp3_file)

        assert audio.shape[0] == 1  # Mono
        assert audio.shape[1] > 0  # Has samples
        assert sample_rate == 24000  # 24kHz


class TestFfmpegErrorHandling:
    """Tests for error handling when ffmpeg is not available."""

    def test_convert_without_ffmpeg(self, sample_wav_file, tmp_path, monkeypatch):
        """Test that conversion raises FileNotFoundError when ffmpeg is not available."""

        # Mock subprocess.run to simulate ffmpeg not being found
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("ffmpeg not found")

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Create a fake non-WAV file
        fake_m4a = tmp_path / "test.m4a"
        fake_m4a.write_bytes(b"fake m4a content")

        with pytest.raises(FileNotFoundError, match="ffmpeg is required"):
            convert_audio_to_wav(fake_m4a)

    def test_convert_invalid_audio_file(self, tmp_path):
        """Test that converting an invalid audio file raises ValueError."""
        if not check_ffmpeg_available():
            pytest.skip("ffmpeg not available")

        # Create a file that looks like audio but isn't
        invalid_file = tmp_path / "invalid.m4a"
        invalid_file.write_text("This is not a valid audio file")

        with pytest.raises(ValueError, match="Failed to convert"):
            convert_audio_to_wav(invalid_file)
