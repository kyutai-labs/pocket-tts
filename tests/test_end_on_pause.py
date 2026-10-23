import torch

from pocket_tts.data.audio_utils import end_on_pause

SR = 24000


def test_end_on_pause_trims_fades_and_pads():
    speech = torch.sin(torch.arange(SR) / 5.0)[None]  # 1 s of tone
    fade = int(0.02 * SR)
    for tail in (0, SR // 2):  # a prompt stopping on speech, and one with 0.5 s of silence
        wav = torch.cat([speech, torch.zeros(1, tail)], dim=-1)
        out = end_on_pause(wav, SR, pause_sec=0.2)
        assert out.shape == (1, SR + int(0.2 * SR))
        assert torch.equal(out[:, : SR - fade], speech[:, : SR - fade])
        assert out[:, SR - 1].abs() < 1e-6
        assert out[:, SR:].abs().max() == 0


def test_end_on_pause_silent_input_is_returned_whole():
    wav = torch.zeros(1, 100)
    assert end_on_pause(wav, SR).shape[-1] >= 100
