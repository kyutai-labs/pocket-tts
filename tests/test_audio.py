import wave
from pathlib import Path

import torch

from pocket_tts.data.audio import audio_read, stream_audio_chunks


def test_stream_audio_chunks_patches_seekable_wav_header(tmp_path: Path):
    output_file = tmp_path / "output.wav"
    samples = torch.zeros(24000)

    stream_audio_chunks(output_file, iter([samples]), sample_rate=24000)

    with wave.open(str(output_file), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 28800


def test_audio_read_uses_soundfile_for_8_bit_wav(tmp_path: Path):
    output_file = tmp_path / "silence_u8.wav"
    with wave.open(str(output_file), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(8000)
        wav_file.writeframes(bytes([128] * 16))

    audio, sample_rate = audio_read(output_file)

    assert sample_rate == 8000
    assert audio.shape == (1, 16)
    assert torch.allclose(audio, torch.zeros_like(audio))
