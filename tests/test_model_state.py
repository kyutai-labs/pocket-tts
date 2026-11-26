from pathlib import Path

import pytest
import safetensors.torch
import torch

from pocket_tts import TTSModel, export_model_state
from pocket_tts.modules.stateful_module import init_states


def test_a_voice_state_of_another_model_is_rejected_when_loading(tmp_path: Path):
    model = TTSModel.load_model()
    state = init_states(model.flow_lm, batch_size=1, sequence_length=3)
    export_model_state(state, tmp_path / "same_model.safetensors")
    model.get_state_for_audio_prompt(tmp_path / "same_model.safetensors")

    # A model with fewer key/value heads, one with more layers, and a pocket-tts 1.0 voice.
    one_kv_head = {
        name: {**s, "cache": s["cache"][:, :, :, :1].contiguous()} for name, s in state.items()
    }
    export_model_state(one_kv_head, tmp_path / "one_kv_head.safetensors")
    first_layer = state["transformer.layers.0.self_attn"]
    more_layers = {
        **state,
        f"transformer.layers.{len(state)}.self_attn": {
            k: v.clone() for k, v in first_layer.items()
        },
    }
    export_model_state(more_layers, tmp_path / "more_layers.safetensors")
    safetensors.torch.save_file(
        {"audio_prompt": torch.zeros(1, 125, 1024)}, tmp_path / "v1.safetensors"
    )
    for name in ["one_kv_head", "more_layers", "v1"]:
        with pytest.raises(ValueError, match="is not a voice state of this model"):
            model.get_state_for_audio_prompt(tmp_path / f"{name}.safetensors")
