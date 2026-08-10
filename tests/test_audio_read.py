import wave

import numpy as np
import pytest

from pocket_tts.data.audio import audio_read

SAMPLE_RATE = 24000


def make_signal(n_samples: int = 4096) -> np.ndarray:
    """A sine sweep, kept just below full scale so no width clips it."""
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    return 0.9 * np.sin(2 * np.pi * 440 * t * (1 + t))


def write_wav(path, signal: np.ndarray, sample_width: int, n_channels: int = 1):
    if n_channels > 1:
        signal = np.repeat(signal, n_channels)

    if sample_width == 1:
        raw = np.clip(signal * 128.0 + 128.0, 0, 255).astype(np.uint8).tobytes()
    elif sample_width == 3:
        scaled = np.clip(signal * 2147483648.0, -(2**31), 2**31 - 1).astype("<i4")
        # Drop the low-order byte to get 24-bit little-endian samples.
        raw = scaled.view(np.uint8).reshape(-1, 4)[:, 1:].tobytes()
    else:
        limit = 2 ** (8 * sample_width - 1)
        dtype = "<i2" if sample_width == 2 else "<i4"
        raw = np.clip(signal * limit, -limit, limit - 1).astype(dtype).tobytes()

    with wave.open(str(path), "wb") as f:
        f.setnchannels(n_channels)
        f.setsampwidth(sample_width)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(raw)


@pytest.mark.parametrize("sample_width", [1, 2, 3, 4])
def test_reads_every_pcm_bit_depth(tmp_path, sample_width):
    signal = make_signal()
    path = tmp_path / f"tone_{sample_width * 8}bit.wav"
    write_wav(path, signal, sample_width)

    wav, sample_rate = audio_read(path)

    assert sample_rate == SAMPLE_RATE
    assert wav.shape == (1, len(signal))
    # 8-bit quantization is coarse; the wider depths should be near-exact.
    tolerance = 1 / 128 if sample_width == 1 else 1e-4
    assert np.allclose(wav.numpy()[0], signal, atol=tolerance)


@pytest.mark.parametrize("sample_width", [2, 3])
def test_downmixes_stereo_to_mono(tmp_path, sample_width):
    signal = make_signal(1024)
    path = tmp_path / f"stereo_{sample_width * 8}bit.wav"
    write_wav(path, signal, sample_width, n_channels=2)

    wav, _ = audio_read(path)

    assert wav.shape == (1, len(signal))
    assert np.allclose(wav.numpy()[0], signal, atol=1e-4)


def test_falls_back_to_soundfile_for_float_wav(tmp_path):
    """Float WAVs are rejected by the wave module but readable via soundfile."""
    sf = pytest.importorskip("soundfile")

    signal = make_signal(1024)
    path = tmp_path / "float32.wav"
    sf.write(str(path), signal.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")

    wav, sample_rate = audio_read(path)

    assert sample_rate == SAMPLE_RATE
    assert np.allclose(wav.numpy()[0], signal, atol=1e-6)
