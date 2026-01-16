"""
Audio IO methods are defined in this module (info, read, write),
We rely on av library for faster read when possible, otherwise on torchaudio.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from beartype.typing import Iterator

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def convert_audio_to_wav(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert audio file to WAV format using ffmpeg.

    Supports various audio formats (MP3, M4A, FLAC, OGG, etc.) by using ffmpeg
    for conversion. Output is 16-bit PCM, mono, 24kHz WAV format.

    Args:
        input_path: Path to input audio file (any format supported by ffmpeg)
        output_path: Optional path for output WAV file. If None, creates a temporary file.

    Returns:
        Path to the converted WAV file.

    Raises:
        FileNotFoundError: If ffmpeg is not found or input file doesn't exist.
        ValueError: If conversion fails.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    # If already WAV, return as-is (unless output_path is specified and different)
    if input_path.suffix.lower() == ".wav":
        if output_path is None or Path(output_path) == input_path:
            return input_path
        # Copy to output_path if specified
        shutil.copy2(input_path, output_path)
        return Path(output_path)

    # Determine output path
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    output_path = Path(output_path)

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise FileNotFoundError(
            "ffmpeg is required to convert audio files. "
            "Please install ffmpeg from https://ffmpeg.org/download.html. "
            "On Windows, download the 'full-shared' build and ensure ffmpeg.exe is in your PATH."
        )

    # Convert using ffmpeg: 16-bit PCM, mono, 24kHz (matching model sample rate)
    # -y: overwrite output file if it exists
    # -i: input file
    # -ar 24000: sample rate 24kHz
    # -ac 1: mono (1 channel)
    # -sample_fmt s16: 16-bit signed integer PCM
    # -acodec pcm_s16le: PCM codec, 16-bit little-endian
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-i",
                str(input_path),
                "-ar",
                "24000",  # Sample rate: 24kHz
                "-ac",
                "1",  # Mono
                "-sample_fmt",
                "s16",  # 16-bit signed integer
                "-acodec",
                "pcm_s16le",  # PCM codec
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        logger.info(f"Converted {input_path.suffix} to WAV: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        raise ValueError(
            f"Failed to convert audio file {input_path} to WAV. ffmpeg error: {error_msg}"
        ) from e
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"Audio conversion timed out for {input_path}. "
            f"The file might be too large or corrupted."
        )


def audio_read(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Read audio using Python's wave module.

    If the file is not a WAV file, it will be automatically converted to WAV first.
    Note: If conversion creates a temporary file, it will be cleaned up automatically
    after reading (unless it's the same as the input file).
    """
    filepath = Path(filepath)
    converted_path = None

    # If not WAV, convert it first
    if filepath.suffix.lower() != ".wav":
        converted_path = convert_audio_to_wav(filepath)
        filepath = converted_path

    try:
        with wave.open(str(filepath), "rb") as wav_file:
            sample_rate = wav_file.getframerate()

            # Read all audio data as 16-bit signed integers
            raw_data = wav_file.readframes(-1)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Return as mono tensor (channels, samples)
            wav = torch.from_numpy(samples.reshape(1, -1))
            return wav, sample_rate
    finally:
        # Clean up converted temp file if it was created
        if converted_path and converted_path != filepath and converted_path.exists():
            try:
                os.unlink(converted_path)
            except (PermissionError, FileNotFoundError):
                # On Windows, file might still be locked. Log and continue.
                # The OS will clean it up eventually.
                logger.debug(f"Could not immediately delete converted temp file: {converted_path}")


class StreamingWAVWriter:
    """WAV writer using Python's standard library wave module."""

    def __init__(self, output_stream, sample_rate: int):
        self.output_stream = output_stream
        self.sample_rate = sample_rate
        self.wave_writer = None
        self.first_chunk_buffer = []

    def write_header(self, sample_rate: int):
        """Initialize WAV writer with header."""
        # For stdout streaming, we need to handle the unseekable stream case
        # The wave module supports unseekable streams since Python 3.4
        self.wave_writer = wave.open(self.output_stream, "wb")
        self.wave_writer.setnchannels(1)  # Mono
        self.wave_writer.setsampwidth(2)  # 16-bit
        self.wave_writer.setframerate(sample_rate)
        self.wave_writer.setnframes(1_000_000_000)

    def write_pcm_data(self, audio_chunk: torch.Tensor):
        """Write PCM data using wave module."""
        # Convert to int16 PCM bytes
        chunk_int16 = (audio_chunk.clamp(-1, 1) * 32767).short()
        chunk_bytes = chunk_int16.detach().cpu().numpy().tobytes()

        if self.first_chunk_buffer is not None:
            self.first_chunk_buffer.append(chunk_bytes)
            total_length = sum(len(c) for c in self.first_chunk_buffer)
            target_length = (
                int(self.sample_rate * FIRST_CHUNK_LENGTH_SECONDS) * 2
            )  # 2 bytes per sample
            if total_length < target_length:
                return
            self._flush()
            return

        # Use writeframesraw to avoid frame count validation for streaming
        self.wave_writer.writeframesraw(chunk_bytes)

    def _flush(self):
        if self.first_chunk_buffer is not None:
            self.wave_writer.writeframesraw(b"".join(self.first_chunk_buffer))
            self.first_chunk_buffer = None

    def finalize(self):
        """Close the wave writer."""
        self._flush()

        # Let's add 200ms of silence to ensure proper playback
        silence_duration_sec = 0.2
        num_silence_samples = int(self.sample_rate * silence_duration_sec)

        self.wave_writer.writeframesraw(bytes(num_silence_samples * 2))

        if self.wave_writer:
            # do not update the header for unseekable streams
            self.wave_writer._patchheader = lambda: None
            self.wave_writer.close()


def is_file_like(obj):
    """Check if object has basic file-like methods."""
    return all(hasattr(obj, attr) for attr in ["write", "close"])


def stream_audio_chunks(
    path: str | Path | None | Any, audio_chunks: Iterator[torch.Tensor], sample_rate: int
):
    """Stream audio chunks to a WAV file or stdout, optionally playing them."""
    if path == "-":
        f = sys.stdout.buffer
    elif path is None:
        f = nullcontext()
    elif is_file_like(path):
        f = path
    else:
        f = open(path, "wb")

    with f:
        if path is not None:
            writer = StreamingWAVWriter(f, sample_rate)
            writer.write_header(sample_rate)

        for chunk in audio_chunks:
            # Then write to file
            if path is not None:
                writer.write_pcm_data(chunk)

        if path is not None:
            writer.finalize()
