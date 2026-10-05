"""The shipped configs must be runnable as-is.

Every defect these catch has actually shipped: a scratch config that stopped
before the quality transition, a batch size a quarter of the floor it needs,
and a teacher path pointing at an architecture the distill step cannot load.
"""

from pathlib import Path
from typing import Any

import pytest

from training.args import TrainArgs, _from_dict, load_args
from training.modules.builders import load_model_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRATCH = CONFIGS / "scratch.yaml"
DISTILL = CONFIGS / "depth_distill.yaml"

# Below 64 rows per optimizer step the acoustic-quality transition arrives late
# or not at all, and 400k steps is where expressivity settles (see README).
MIN_EFFECTIVE_BATCH = 64
MIN_SCRATCH_STEPS = 400_000


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_config_parses(path: Path):
    load_args(path)


def test_scratch_reaches_the_effective_batch_floor():
    args = load_args(SCRATCH)
    assert args.batch_size * args.grad_accum_steps >= MIN_EFFECTIVE_BATCH, (
        "scratch must reach 64 rows per step on a single GPU: "
        f"{args.batch_size} x {args.grad_accum_steps}"
    )


def test_scratch_runs_past_the_quality_transition():
    assert load_args(SCRATCH).max_steps >= MIN_SCRATCH_STEPS


def test_scratch_builds_the_reference_teacher_depth():
    args = load_args(SCRATCH)
    config = load_model_config(args.model_config, args.model_overrides)
    assert config.flow_lm.transformer.num_layers == 24


def test_distill_teacher_is_deeper_than_its_student():
    args = load_args(DISTILL)
    student = load_model_config(args.model_config, args.model_overrides)
    teacher = load_model_config(args.distill_teacher_config, args.distill_teacher_overrides)
    assert teacher.flow_lm.transformer.num_layers > student.flow_lm.transformer.num_layers
    # Depth distillation copies every non-backbone tensor, so the rest must match.
    assert teacher.flow_lm.transformer.d_model == student.flow_lm.transformer.d_model


def test_distill_teacher_weights_point_at_the_scratch_run():
    args = load_args(DISTILL)
    assert args.distill_teacher_weights, "the distill config must name a teacher checkpoint"
    assert str(load_args(SCRATCH).run_dir) in args.distill_teacher_weights, (
        "the documented path is scratch -> distill; the teacher checkpoint should come from "
        f"{load_args(SCRATCH).run_dir}"
    )


class TestArgValidation:
    """Misconfigurations that used to run and quietly do the wrong thing."""

    def test_num_ckpt_keep_zero_is_rejected(self):
        with pytest.raises(ValueError, match="num_ckpt_keep"):
            TrainArgs(num_ckpt_keep=0)

    def test_zero_frequencies_are_rejected(self):
        for field in ("valid_freq", "ckpt_freq", "log_freq"):
            kwargs: dict[str, Any] = {field: 0}
            with pytest.raises(ValueError, match=field):
                TrainArgs(**kwargs)

    def test_distillation_without_a_teacher_is_rejected(self):
        with pytest.raises(ValueError, match="teacher"):
            TrainArgs(distill_cfg_coef=1.5, start_from_pretrained=False)

    def test_teacher_config_without_weights_is_rejected(self):
        with pytest.raises(ValueError, match="distill_teacher_weights"):
            TrainArgs(distill_teacher_config="x.yaml")

    def test_unknown_keys_are_rejected(self):
        """A key the parser doesn't recognize is a setting the user thinks is applied."""
        with pytest.raises(ValueError, match="distill_seed_layers"):
            _from_dict(TrainArgs, {"distill_seed_layers": "first"})


def test_xn_ptts_export_config_matches_the_stock_architecture():
    """xn-ptts's TTSConfig::v202601 is the released 6-layer English model, run the pocket-tts
    way (alex/pocket-tts-models)."""
    from training.scripts.export_xn_ptts import xn_ptts_config

    config = load_model_config(str(CONFIGS.parents[1] / "pocket_tts/config/english.yaml"), {})
    xn = xn_ptts_config(config, eos_threshold=-4.0)
    assert xn["flow_lm"] == {
        "d_model": 1024,
        "num_heads": 16,
        "num_layers": 6,
        "dim_feedforward": 4096,
        "max_period": 10000.0,
        "n_bins": 4000,
        "lut_dim": 1024,
        "flow_dim": 512,
        "flow_depth": 6,
        "ldim": 32,
        "gelu": "tanh",
        "time_rms_norm": "var",
        "insert_bos_before_voice": True,
    }
    assert xn["mimi"]["dimension"] == 512 and xn["mimi"]["quantizer_dimension"] == 32
    assert xn["cfg_null_audio_empty"], "the CFG null is the bare bos_before_voice"
    assert xn["mimi"]["ratios"] == [6, 5, 4] and xn["mimi"]["transformer_context"] == 250


def test_small_voices_configs_share_their_voice_table():
    finetune = load_args(CONFIGS / "small_voices_finetune.yaml")
    distill = load_args(CONFIGS / "small_voices_distill.yaml")
    assert finetune.voices and finetune.voices == distill.voices
    assert finetune.voice_lut_dim == distill.voice_lut_dim
    assert distill.voice_lut_dropout == distill.voice_dropout == distill.text_dropout == 0
