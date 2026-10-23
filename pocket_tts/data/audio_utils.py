"""Various utilities for audio conversion (pcm format, sample rate and channels),
and volume normalization."""

import torch
from scipy.signal import resample_poly


def convert_audio(
    wav: torch.Tensor, from_rate: float, to_rate: float, to_channels: int
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


def end_on_pause(
    wav: torch.Tensor,
    sample_rate: int,
    pause_sec: float = 0.08,
    fade_sec: float = 0.02,
    floor_db: float = 35.0,
) -> torch.Tensor:
    """End a voice prompt [..., T] on exactly `pause_sec` of silence.

    Training prompts end inside the pause between two words. A prompt that stops on speech
    makes the model continue that utterance (a burst at the start of every chunk, or a wrong
    first word), and one that ends on a long silence delays the onset. The trailing silence
    (20 ms frames more than `floor_db` below the loudest one) is cut, the last `fade_sec` of
    what remains is faded out, and `pause_sec` of zeros is appended.
    """
    frame = max(1, int(0.02 * sample_rate))
    n = wav.shape[-1] // frame
    if n == 0:
        return wav
    rms = wav[..., : n * frame].reshape(-1, n, frame).square().mean(dim=(0, 2)).sqrt()
    db = 20 * torch.log10(rms + 1e-12)
    loud = torch.nonzero(db > db.max() - floor_db).flatten()
    end = (int(loud[-1]) + 1) * frame
    wav = wav[..., :end].clone()
    fade = min(int(fade_sec * sample_rate), end)
    wav[..., end - fade :] *= torch.linspace(1, 0, fade, device=wav.device, dtype=wav.dtype)
    pause = wav.new_zeros(*wav.shape[:-1], int(pause_sec * sample_rate))
    return torch.cat([wav, pause], dim=-1)
