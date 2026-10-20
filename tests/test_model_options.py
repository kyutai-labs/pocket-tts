"""Architecture options, on small randomly initialized models."""

from pathlib import Path
from typing import TypeVar

import safetensors.torch
import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import build_mimi
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
def sample_latents(flow_lm: FlowLMModel, num_frames: int = 3) -> torch.Tensor:
    """[num_frames, ldim] latents generated after a short text, streaming one frame at a time."""
    torch.manual_seed(1)
    state = init_states(flow_lm, batch_size=1, sequence_length=32)
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


def test_explicit_feed_forward_width():
    flow_lm = build_flow_lm(**{"transformer.dim_feedforward": 48})
    weights = flow_lm.state_dict()
    assert weights["transformer.layers.0.linear1.weight"].shape == (48, 64)
    assert weights["transformer.layers.0.linear2.weight"].shape == (64, 48)
    default = build_flow_lm().state_dict()["transformer.layers.0.linear1.weight"]
    assert default.shape == (64 * 4, 64)  # d_model * hidden_scale
    assert sample_latents(flow_lm).isfinite().all()


def test_pointwise_head_is_lsd_one_step_without_time_conditions():
    pointwise = build_flow_lm(**{"flow.type": "pointwise"})
    assert pointwise.flow_net.num_time_conds == 0
    # The same head as an LSD head whose time embeddings are zero: one LSD step is then
    # the noise plus the head's output for it.
    lsd = build_flow_lm()
    state = lsd.state_dict()
    for k, v in state.items():
        if k.startswith("flow_net.time_embed.") and not k.endswith("freqs"):
            v.zero_()
    lsd.load_state_dict({**state, **pointwise.state_dict()}, strict=True)
    assert torch.equal(sample_latents(pointwise), sample_latents(lsd))
