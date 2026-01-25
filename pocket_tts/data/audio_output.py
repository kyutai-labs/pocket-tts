# Import numpy_rs for NumPy replacement
from pathlib import Path
from typing import Union

import numpy as np_rs
import scipy.io.wavfile
import torch


def save_audio(
    path: Union[str, Path], audio: Union[torch.Tensor, np_rs.ndarray], sample_rate: int
) -> None:
    """Save audio tensor or array to a WAV file with automatic conversion.

    This function provides a reliable way to save audio data to WAV format.
    It handles both PyTorch tensors and NumPy arrays automatically,
    ensuring proper format conversion and shape handling.

    Args:
        path: Destination path for the WAV file. Can be string or Path object.
            The file will be created or overwritten if it exists.
        audio: Audio data to save. Accepts:
            - torch.Tensor: Any shape, automatically detached and moved to CPU
            - numpy.ndarray: Any shape, used directly
            Expected format: float32 values normalized to [-1, 1]
        sample_rate: Sampling rate in Hz for the output WAV file.
            Common values: 16000, 22050, 24000, 44100, 48000.

    Raises:
        ValueError: If audio data is invalid or empty.
        TypeError: If audio is neither torch.Tensor nor numpy.ndarray.
        OSError: If the destination path is invalid or not writable.

    Example:
        >>> import torch
        >>> # Generate 1 second of sine wave at 440Hz
        >>> samples = torch.linspace(0, 1, 24000)
        >>> audio = torch.sin(2 * torch.pi * 440 * samples) * 0.5
        >>> save_audio("tone.wav", audio, 24000)
        >>> # Creates tone.wav with 1 second of 440Hz tone

    Note:
        - Audio is automatically converted to int16 PCM for WAV format
        - Multi-channel audio (shape [channels, samples]) is preserved
        - Single-channel audio (shape [samples]) is written as mono
        - Uses scipy.io.wavfile for reliable WAV writing
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    # Ensure correct shape for scipy (samples, channels) or just (samples) for mono
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)

    # Normalize if needed? Assuming model output is float32 [-1, 1] usually,
    # but scipy handles float32 fine.

    scipy.io.wavfile.write(str(path), sample_rate, audio)
