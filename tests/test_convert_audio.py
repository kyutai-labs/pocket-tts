"""Resampling in convert_audio must honour its float sample-rate contract."""

import pytest
import torch

from pocket_tts.data.audio_utils import convert_audio


@pytest.mark.parametrize(
    "from_rate, to_rate, wav, expected",
    [
        # Integer rates: the resampling ratio is exact.
        (48000, 24000, torch.zeros(1, 48000), torch.Size([1, 24000])),
        (24000, 48000, torch.zeros(1, 24000), torch.Size([1, 48000])),
        # The signature accepts floats (config frame_rate, soundfile metadata); these
        # must resample too, not blow up in torch.gcd on a float tensor.
        (48000.0, 24000.0, torch.zeros(1, 48000), torch.Size([1, 24000])),
        (24000.0, 48000.0, torch.zeros(1, 24000), torch.Size([1, 48000])),
        (44100.0, 48000.0, torch.zeros(1, 44100), torch.Size([1, 48000])),
        # Equal rates take the no-op branch whatever the numeric type.
        (24000, 24000, torch.zeros(1, 12345), torch.Size([1, 12345])),
        (24000.0, 24000.0, torch.zeros(1, 12345), torch.Size([1, 12345])),
    ],
)
def test_convert_audio_resamples_at_the_requested_ratio(
    from_rate: float, to_rate: float, wav: torch.Tensor, expected: torch.Size
):
    out = convert_audio(wav, from_rate, to_rate, to_channels=1)
    assert out.shape == expected


def test_convert_audio_preserves_signal_on_float_rates():
    """An actual tone must survive a float-rate conversion, not just the shape."""
    t = torch.arange(24000, dtype=torch.float32) / 24000
    tone = torch.sin(2 * torch.pi * 440 * t)[None]
    out = convert_audio(tone, 24000.0, 48000.0, to_channels=1)
    assert out.shape == (1, 48000)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0.5


def test_convert_audio_keeps_a_2d_channel_layout():
    """The channel assertion expects [channels, T]; mono input must stay 2D."""
    stereo = torch.zeros(2, 24000)
    out = convert_audio(stereo, 24000.0, 48000.0, to_channels=2)
    assert out.shape == (2, 48000)
