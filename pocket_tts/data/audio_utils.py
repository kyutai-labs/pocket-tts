"""Various utilities for audio conversion (pcm format, sample rate and channels),
and volume normalization."""

import torch
from scipy.signal import resample_poly


def convert_audio(
    wav: torch.Tensor, from_rate: int | float, to_rate: int | float, to_channels: int
) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    if from_rate != to_rate:
        # Convert to numpy for scipy resampling
        wav_np = wav.detach().cpu().numpy()

        # Calculate resampling parameters
        gcd = int(torch.gcd(torch.tensor(from_rate), torch.tensor(to_rate)).item())
        up = int(to_rate // gcd)
        down = int(from_rate // gcd)

        # Resample using scipy
        resampled_np = resample_poly(wav_np, up, down, axis=-1)

        # Convert back to torch tensor
        wav = torch.from_numpy(resampled_np).to(wav.device).to(wav.dtype)

    assert wav.shape[-2] == to_channels
    return wav
