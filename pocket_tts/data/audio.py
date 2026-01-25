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

import numpy as np
import torch
from beartype.typing import Iterator

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def load_wav(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio file with automatic format detection.

    This function provides a unified interface for loading audio files.
    It supports WAV files natively and other formats through soundfile.
    The audio is automatically converted to mono and normalized to float32.

    Args:
        filepath: Path to the audio file. Can be a string or Path object.
            Supported formats include WAV (native), FLAC, MP3, OGG (via soundfile).

    Returns:
        tuple[torch.Tensor, int]: A tuple containing:
            - Audio tensor with shape [1, samples] (mono, float32, normalized to [-1, 1])
            - Sample rate as integer

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        ImportError: If soundfile is required but not installed for non-WAV formats.
        ValueError: If the audio file is corrupted or invalid.

    Example:
        >>> audio, sr = load_wav("speech.wav")
        >>> print(f"Loaded {audio.shape[1]} samples at {sr}Hz")
        >>> # Output: Loaded 24000 samples at 24000Hz

    Note:
        - WAV files are loaded using Python's built-in wave module
        - Other formats require soundfile: `pip install soundfile`
        - Multi-channel audio is automatically mixed to mono
        - Audio is normalized to float32 range [-1, 1]
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        # Use built-in wave module for WAV files
        with wave.open(str(filepath), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            raw_data = wav_file.readframes(-1)
            samples = (
                np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return torch.from_numpy(samples).unsqueeze(0), sample_rate

    # For non-WAV formats, use soundfile (optional dependency)
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    return wav, sample_rate


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
    path: str | Path | None | Any,
    audio_chunks: Iterator[torch.Tensor],
    sample_rate: int,
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
