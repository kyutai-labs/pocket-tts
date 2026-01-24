"""
Audio IO methods are defined in this module (info, read, write),
We rely on av library for faster read when possible, otherwise on torchaudio.
"""

import logging
import os
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Import numpy_rs for NumPy replacement
import numpy as np_rs
import torch
from beartype.typing import Iterator

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def audio_read(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Read audio using Python's wave module.

    Args:
        filepath: Path to the WAV audio file.

    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor is shape [1, num_samples]
        with values in [-1, 1] range.
    """
    with wave.open(str(filepath), "rb") as wav_file:
        sample_rate = wav_file.getframerate()

        # Read all audio data as 16-bit signed integers
        raw_data = wav_file.readframes(-1)
        samples = (
            np_rs.frombuffer(raw_data, dtype=np_rs.int16).astype(np_rs.float32)
            / 32768.0
        )

        # Return as mono tensor (channels, samples)
        wav = torch.from_numpy(samples.reshape(1, -1))
        return wav, sample_rate


# Industry-standard alias for audio_read (matches Coqui TTS, VITS, etc.)
def load_wav(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio file from path.

    This is an alias for audio_read() to match common TTS library conventions.

    Args:
        filepath: Path to the WAV audio file.

    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor is shape [1, num_samples]
        with values in [-1, 1] range.

    Examples:
        >>> from pocket_tts.data.audio import load_wav
        >>> audio, sr = load_wav("voice_sample.wav")
        >>> print(f"Loaded audio: {audio.shape}, sample rate: {sr}")
    """
    return audio_read(filepath)


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
        self.wave_writer = wave.open(self.output_stream, "wb")  # noqa: SIM115 - managed by finalize()
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
    path: str | Path | None | Any,
    audio_chunks: Iterator[torch.Tensor],
    sample_rate: int,
):
    """Stream audio chunks to a WAV file or stdout, optionally playing them."""
    # Handle file path case separately to use context manager directly
    if not is_file_like(path) and path not in ("-", None):
        with open(path, "wb") as f:
            writer = StreamingWAVWriter(f, sample_rate)
            writer.write_header(sample_rate)
            for chunk in audio_chunks:
                writer.write_pcm_data(chunk)
            writer.finalize()
        return

    # Handle stdout, null context, and file-like objects
    if path == "-":
        f = sys.stdout.buffer
    elif path is None:
        f = nullcontext()
    else:
        f = path

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
