"""
Tests for int8 quantization.

Verifies:
1. Quantized model produces valid audio (not silence, not NaN)
2. load_model(quantize=True) applies quantization
3. CLI --quantize flag is accepted
4. Backend detection works
"""

import torch
from typer.testing import CliRunner

from pocket_tts import TTSModel
from pocket_tts.main import cli_app
from pocket_tts.quantization import _get_backend

SHORT_TEXT = "Hello, this is a test."
TEST_VOICE = "alba"

runner = CliRunner()


def test_quantized_model_produces_audio():
    model = TTSModel.load_model(quantize=True)
    voice_state = model.get_state_for_audio_prompt(TEST_VOICE)
    audio = model.generate_audio(voice_state, SHORT_TEXT)
    assert audio is not None
    assert len(audio) > 0
    assert not torch.isnan(audio).any(), "Audio contains NaN values"
    assert not torch.isinf(audio).any(), "Audio contains Inf values"
    assert audio.abs().max() > 0, "Audio is silent"


def test_quantize_flag_applies_quantization():
    model_q = TTSModel.load_model(quantize=True)
    model_b = TTSModel.load_model(quantize=False)
    # Quantized model's Linear layers should have different weight types than baseline
    layer_q = model_q.flow_lm.transformer.layers[0]
    layer_b = model_b.flow_lm.transformer.layers[0]
    weight_type_q = type(layer_q.self_attn.in_proj.weight).__name__
    weight_type_b = type(layer_b.self_attn.in_proj.weight).__name__
    assert weight_type_q != weight_type_b, (
        f"Attention weights should differ after quantization, got {weight_type_q} for both"
    )


def test_cli_quantize_flag(tmp_path):
    output_file = tmp_path / "quantized_output.wav"
    result = runner.invoke(
        cli_app,
        ["generate", "--quantize", "--text", SHORT_TEXT, "--output-path", str(output_file), "-q"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"


def test_backend_detection():
    backend = _get_backend()
    assert backend in ("torchao", "torch.ao")
