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


# Integer PCM sample widths, in bytes, that map directly onto a numpy dtype.
# 24-bit (3 bytes) has no numpy equivalent and is handled separately.
_PCM_DTYPES = {1: np.uint8, 2: "<i2", 4: "<i4"}


def _decode_pcm(raw_data: bytes, sample_width: int) -> np.ndarray:
    """Decode little-endian integer PCM bytes to float32 samples in [-1, 1]."""
    usable = len(raw_data) - len(raw_data) % sample_width
    raw_data = raw_data[:usable]

    if sample_width == 3:
        # No 24-bit numpy dtype exists, so widen each sample to 32 bits by
        # padding the low-order byte, which preserves both value and sign.
        as_bytes = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((as_bytes.shape[0], 4), dtype=np.uint8)
        widened[:, 1:] = as_bytes
        samples = widened.view("<i4").reshape(-1)
        return samples.astype(np.float32) / 2147483648.0

    samples = np.frombuffer(raw_data, dtype=_PCM_DTYPES[sample_width])
    if sample_width == 1:
        # 8-bit WAV is the odd one out: unsigned, centered on 128.
        return (samples.astype(np.float32) - 128.0) / 128.0
    return samples.astype(np.float32) / float(2 ** (8 * sample_width - 1))


def _read_wav_with_stdlib(filepath: Path) -> tuple[torch.Tensor, int]:
    """Read an integer-PCM WAV file using the standard library only."""
    with wave.open(str(filepath), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width not in (1, 2, 3, 4):
            raise wave.Error(f"unsupported sample width: {sample_width} bytes")
        raw_data = wav_file.readframes(-1)

    samples = _decode_pcm(raw_data, sample_width)
    if n_channels > 1:
        samples = samples[: len(samples) - len(samples) % n_channels]
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(samples).unsqueeze(0), sample_rate


def _read_with_soundfile(filepath: Path) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            f"soundfile is required to read {filepath.name}. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    return wav, sample_rate


def audio_read(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Read audio file. WAV uses built-in wave module; other formats require soundfile."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        try:
            return _read_wav_with_stdlib(filepath)
        except wave.Error:
            # The stdlib only handles plain integer PCM. Float-encoded and
            # WAVE_FORMAT_EXTENSIBLE files still work through soundfile.
            logger.debug("Falling back to soundfile for %s", filepath)

    return _read_with_soundfile(filepath)


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
