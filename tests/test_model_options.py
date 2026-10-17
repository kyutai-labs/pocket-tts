"""Architecture options, on small randomly initialized models."""

from pathlib import Path
from typing import TypeVar

import safetensors.torch
import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import build_mimi
from pocket_tts.models.tts_model import TTSModel, stamp_state_names
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule, increment_steps, init_states
from pocket_tts.utils.config import CONFIGS_DIR, FlowLMConfig, load_config
from pocket_tts.utils.weights_loading import get_mimi_state_dict

M = TypeVar("M", bound=torch.nn.Module)


def build_flow_lm(**overrides: object) -> FlowLMModel:
    """A 2-layer, 64-wide english flow LM with `overrides` ("transformer.num_kv_heads": 2, ...)."""
    torch.manual_seed(0)
    config = load_config(CONFIGS_DIR / "english.yaml").flow_lm.model_dump()
    config["transformer"].update(d_model=64, num_heads=4, num_layers=2)
    config["lookup_table"]["dim"] = 64
    config["flow"].update(dim=32, depth=1)
    for key, value in overrides.items():
        *path, leaf = key.split(".")
        section = config
        for part in path:
            section = section[part]
        section[leaf] = value
    flow_lm = FlowLMModel.from_pydantic_config(
        FlowLMConfig(**config), latent_dim=32, insert_bos_before_voice=True
    )
    return stamp(flow_lm).eval()


def stamp(model: M) -> M:
    for name, module in model.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name
    return model


@torch.no_grad()
def sample_latents(
    flow_lm: FlowLMModel, num_frames: int = 3, voice: int | None = None
) -> torch.Tensor:
    """[num_frames, ldim] latents generated after a short text, streaming one frame at a time."""
    torch.manual_seed(1)
    state = init_states(flow_lm, batch_size=1, sequence_length=32)
    if voice is not None:
        assert flow_lm.voice_table is not None
        flow_lm.voice_table.select(state, voice)
    text = flow_lm.conditioner(flow_lm.conditioner.prepare("Hello world."))
    latent = torch.full((1, 1, flow_lm.ldim), float("nan"))
    latents = []
    for _ in range(num_frames):
        latent, _ = flow_lm(latent, text, state, 1, 0.3, None, -4.0)
        latent = latent[:, None]
        increment_steps(flow_lm, state, increment=text.shape[1] + 1)
        text = text[:, :0]
        latents.append(latent[0, 0])
    return torch.stack(latents)


def test_z_dim_projects_backbone_output_for_both_heads():
    flow_lm = build_flow_lm(z_dim=16)
    assert flow_lm.z_proj is not None
    assert flow_lm.z_proj.weight.shape == (16, 64)
    assert flow_lm.out_eos.in_features == 16
    assert flow_lm.flow_net.cond_embed.in_features == 16
    assert sample_latents(flow_lm).isfinite().all()


def test_z_dim_defaults_to_no_projection():
    flow_lm = build_flow_lm()
    assert flow_lm.z_proj is None
    assert not any(k.startswith("z_proj") for k in flow_lm.state_dict())


def test_grouped_query_attention_matches_repeated_kv_heads():
    torch.manual_seed(0)
    rope = RotaryEmbedding()
    gqa = StreamingMultiheadAttention(64, 4, rope, attn_dim=32, num_kv_heads=2)
    mha = StreamingMultiheadAttention(64, 4, rope, attn_dim=32)
    q, k, v = gqa.in_proj.weight.detach().split([32, 16, 16])

    def repeat_heads(w: torch.Tensor) -> torch.Tensor:
        return w.view(2, 8, 64).repeat_interleave(2, dim=0).reshape(32, 64)

    mha.in_proj.weight.data = torch.cat([q, repeat_heads(k), repeat_heads(v)])
    mha.out_proj.weight.data = gqa.out_proj.weight.data
    x = torch.randn(1, 5, 64)
    torch.testing.assert_close(gqa(x, None), mha(x, None))


def test_attn_dim_and_kv_heads_in_the_flow_lm():
    flow_lm = build_flow_lm(**{"transformer.attn_dim": 32, "transformer.num_kv_heads": 2})
    weights = flow_lm.state_dict()
    assert weights["transformer.layers.0.self_attn.in_proj.weight"].shape == (32 + 2 * 16, 64)
    assert weights["transformer.layers.0.self_attn.out_proj.weight"].shape == (64, 32)
    state = init_states(flow_lm, batch_size=1, sequence_length=8)
    assert state["transformer.layers.0.self_attn"]["cache"].shape == (2, 1, 8, 2, 8)
    assert sample_latents(flow_lm).isfinite().all()


def test_low_rank_text_table():
    flow_lm = build_flow_lm(**{"lookup_table.rank": 8})
    conditioner = flow_lm.conditioner
    assert conditioner.embed.weight.shape == (4001, 8)
    assert conditioner.proj is not None and conditioner.proj.weight.shape == (64, 8)
    tokens = conditioner.prepare("Hello world.")
    expected = conditioner.embed.weight[tokens] @ conditioner.proj.weight.T
    torch.testing.assert_close(conditioner(tokens), expected)
    assert sample_latents(flow_lm).isfinite().all()


def test_voice_table_adds_the_selected_voice_to_audio_frames():
    flow_lm = build_flow_lm(voices=["first", "second"])
    assert flow_lm.voice_table is not None
    assert flow_lm.voice_table.embed.weight.shape == (2, 64)
    first, second = sample_latents(flow_lm, voice=0), sample_latents(flow_lm, voice=1)
    assert not torch.allclose(first, second)
    # A zero embedding is no voice at all.
    flow_lm.voice_table.embed.weight.data[0] = 0
    without_table = build_flow_lm()
    without_table.load_state_dict(
        {k: v for k, v in flow_lm.state_dict().items() if not k.startswith("voice_table.")}
    )
    torch.testing.assert_close(sample_latents(flow_lm, voice=0), sample_latents(without_table))


def test_builtin_voice_by_name():
    config = load_config(CONFIGS_DIR / "english.yaml")
    flow_lm = build_flow_lm(voices=["first", "second"])
    config.flow_lm.voices = ["first", "second"]
    tts_model = TTSModel(flow_lm, 0.3, 1, None, -4.0, config)
    tts_model.mimi = build_mimi(config.mimi)
    stamp_state_names(tts_model)
    assert tts_model.voice_names == ["first", "second"]
    state = tts_model.get_state_for_audio_prompt("second")
    assert flow_lm.voice_table is not None
    torch.testing.assert_close(
        state["voice_table"]["embedding"][0, 0], flow_lm.voice_table.embed.weight[1]
    )
    # The prompt is just the BOS frame.
    assert int(state["transformer.layers.0.self_attn"]["offset"]) == 1


@torch.no_grad()
def test_mimi_decoder_with_its_own_dims(tmp_path: Path):
    config = load_config(CONFIGS_DIR / "english.yaml").mimi
    small = config.model_copy(
        update={
            "outer_dim": 64,
            "quantizer": config.quantizer.model_copy(update={"output_dimension": 64}),
            "decoder_seanet": config.seanet.model_copy(
                update={"dimension": 64, "n_filters": 8, "kernel_size": 3}
            ),
            "decoder_transformer": config.transformer.model_copy(
                update={
                    "d_model": 64,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 128,
                    "input_dimension": 64,
                    "output_dimensions": (64,),
                }
            ),
        }
    )
    mimi = stamp(build_mimi(small)).eval()
    assert mimi.decoder.dimension == 64 and mimi.encoder.dimension == 512
    latents = torch.randn(1, 3, config.quantizer.dimension)
    audio = mimi.decode_from_latent(latents, init_states(mimi, 1, 100))
    assert audio.shape == (1, 1, 3 * mimi.frame_size)

    # A Mimi in this package's layout loads as mimi.weights_path.
    path = tmp_path / "mimi.safetensors"
    safetensors.torch.save_file({f"mimi.{k}": v for k, v in mimi.state_dict().items()}, path)
    build_mimi(small).load_state_dict(get_mimi_state_dict(path), strict=True)
