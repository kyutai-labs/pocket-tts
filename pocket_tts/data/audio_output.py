# Import numpy_rs for NumPy replacement
from pathlib import Path
from typing import Union

import numpy as np_rs
import scipy.io.wavfile
import torch


def save_audio(
    path: Union[str, Path], audio: Union[torch.Tensor, np_rs.ndarray], sample_rate: int
) -> None:
    """Save audio to a WAV file with automatic type conversion.

    Args:
        path: Destination path for the WAV file.
        audio: Audio data as a torch.Tensor or numpy.ndarray.
        sample_rate: Sampling rate in Hz.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    # Ensure correct shape for scipy (samples, channels) or just (samples) for mono
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)

    # Normalize if needed? Assuming model output is float32 [-1, 1] usually,
    # but scipy handles float32 fine.

    scipy.io.wavfile.write(str(path), sample_rate, audio)
