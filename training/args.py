import dataclasses
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataArgs:
    train_jsonl: str = ""
    valid_jsonl: str = ""
    max_duration_sec: float = 30.0  # utterances are cropped to this length
    # Voice prompt window: eligible cuts are word
    # boundaries ending inside the first max_voice_prompt_sec seconds, drawn
    # uniformly, and the prompt is the whole utterance start up to the cut --
    # so prompts vary in length and the target keeps most of the utterance.
    # <= 0 removes the window (any word boundary; full-prefix prompt).
    max_voice_prompt_sec: float = 5.0
    shuffle: bool = True


@dataclass
class FlowArgs:
    # "lsd" | "flow_matching", kwargs forwarded to the objective class.
    type: str = "lsd"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimArgs:
    lr: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_norm: float = 1.0
    warmup_steps: int = 500
    # "constant" or "cosine" (decay to lr * lr_min_ratio at max_steps). The best
    # two-phase recipe: train constant past the quality flip, then resume with
    # a cosine fine-tune.
    schedule: str = "constant"
    lr_min_ratio: float = 0.0


@dataclass
class TrainArgs:
    data: DataArgs = field(default_factory=DataArgs)
    flow: FlowArgs = field(default_factory=FlowArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)

    # A pocket-tts model config (e.g. pocket_tts/config/english.yaml or a local
    # variant). Defines the FlowLM/Mimi architecture, the tokenizer, and the
    # weights used for Mimi (and for the FlowLM too when start_from_pretrained).
    model_config: str = ""
    # Dotted-path edits applied to model_config before it is validated, e.g.
    # {"flow_lm.transformer.num_layers": 24} to train the 24-layer teacher from
    # a released model's config. Avoids copying a whole config to change a
    # field, and keeps the language choice in one place.
    model_overrides: dict[str, tp.Any] = field(default_factory=dict)
    # If true, warm-start the FlowLM from the config's weights_path (fine-tuning).
    # If false, only Mimi/tokenizer weights are used and the FlowLM is
    # re-initialized (training from scratch).
    start_from_pretrained: bool = True
    # Load the pretrained weights but start the text embedding from scratch.
    # Needed when the tokenizer differs from the one the weights were trained
    # with, e.g. when training for a new language.
    reset_text_embedding: bool = False

    run_dir: Path = Path("runs/debug")
    batch_size: int = 8  # per GPU
    # Micro-batches accumulated per optimizer step. The quality flip needs an
    # effective batch (batch_size * grad_accum_steps * world_size) >= 64;
    # small GPUs get there by accumulating instead of growing batch_size.
    grad_accum_steps: int = 1
    max_steps: int = 100_000
    seed: int = 42

    flow_batch_multiplier: int = 1  # extra flow-loss samples per backbone position
    eos_loss_weight: float = 0.1
    text_dropout: float = 0.2  # CFG dropout of the text prefix
    voice_dropout: float = 0.2  # CFG dropout of the voice prefix
    # Update the emb_mean/emb_std latent-normalization buffers by EMA for this
    # many first steps (0 keeps whatever the loaded weights contain).
    stats_ema_steps: int = 0
    stats_ema_decay: float = 0.999

    log_freq: int = 20
    valid_freq: int = 2000
    # torch.compile the backbone layers + flow head (and the distill teacher's
    # backbone) in place. +24% throughput on one GPU, +7% steady-state under
    # DDP (~9 min one-time warmup -- a wash on runs under ~2h). Dynamo cannot
    # trace the beartype wrappers, so train.py disables the claw at import
    # time; POCKET_TTS_NO_BEARTYPE=0 forces it back on and requires
    # compile: false.
    compile: bool = True
    num_valid_batches: int = 50
    # Rank 0 synthesizes these sentences every sample_freq steps into
    # run_dir/samples/ (raw weights, first batch's voice prompt) so a run can
    # be judged by ear while it trains. Empty list disables.
    sample_sentences: list[str] = field(
        default_factory=lambda: [
            "The quick brown fox jumps over the lazy dog.",
            "Training is still running; this voice is a work in progress.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        ]
    )
    sample_freq: int = 10000
    sample_temp: float = 0.3
    sample_cfg_coef: float = 1.0
    ckpt_freq: int = 2000
    num_ckpt_keep: int = 3
    ema_decay: float = 0.999  # EMA of the FlowLM weights; 0 disables
    # Latent CFG distillation: > 0 turns training into distilling a frozen copy
    # of the pretrained model. The teacher's backbone output is computed twice
    # (full conditioning and null conditioning) and combined with this guidance
    # coefficient; the student's backbone regresses onto it, with the flow head
    # and EOS head frozen. The result generates WITHOUT CFG (use --cfg 1) at the
    # quality of guided sampling, i.e. one backbone pass per step instead of two.
    # Requires start_from_pretrained: true.
    distill_cfg_coef: float = 0.0
    # Depth distillation: point these at a bigger trained model and the student
    # (built from `model_config`, typically fewer layers) learns to reproduce its
    # backbone activations. Leave empty to distill a frozen copy of the student's
    # own architecture (guidance baking only).
    distill_teacher_config: str = ""
    distill_teacher_overrides: dict[str, tp.Any] = field(default_factory=dict)
    distill_teacher_weights: str = ""
    # Use the teacher checkpoint's EMA shadow as the regression target (falls
    # back to raw weights when the checkpoint carries no EMA).
    distill_teacher_use_ema: bool = True
    # Which teacher layers seed the student backbone: "spaced" (evenly across
    # the depth) or "first" (the bottom N). No evidence either way -- "first"
    # keeps the early feature extractors contiguous.

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.num_ckpt_keep < 1:
            raise ValueError(
                f"num_ckpt_keep must be >= 1, got {self.num_ckpt_keep} "
                "(0 would keep every checkpoint, not none)"
            )
        for name in ("valid_freq", "ckpt_freq", "log_freq"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.distill_cfg_coef > 0 and not (
            self.start_from_pretrained or self.distill_teacher_config
        ):
            raise ValueError(
                "distillation needs a teacher: set distill_teacher_config, or "
                "start_from_pretrained: true to distil a frozen copy of the model itself"
            )
        if self.distill_teacher_config and not self.distill_teacher_weights:
            raise ValueError("distill_teacher_config is set but distill_teacher_weights is not")


def _from_dict(cls, data: dict[str, Any]):
    sub = {"data": DataArgs, "flow": FlowArgs, "optim": OptimArgs}
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"Unknown config key {key!r} for {cls.__name__}")
        if key in sub:
            value = _from_dict(sub[key], value)
        elif key == "betas":
            value = tuple(float(v) for v in value)
        elif fields[key].type is float:
            value = float(value)  # yaml parses "1e-5" as a string
        elif fields[key].type is int:
            value = int(value)
        elif fields[key].type is Path:
            value = Path(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_args(path: str | Path) -> TrainArgs:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _from_dict(TrainArgs, raw)


def dump_args(args: TrainArgs) -> str:
    """The resolved config (defaults included) as yaml."""

    def plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    return yaml.safe_dump(plain(dataclasses.asdict(args)), sort_keys=False)


def save_args(args: TrainArgs, path: str | Path) -> None:
    with open(path, "w") as f:
        f.write(dump_args(args))
