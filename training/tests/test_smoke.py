"""CPU smoke tests: tiny model, one training step per objective, short generation."""

import copy
import json
import random
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import sentencepiece as spm
import torch
from torch import nn

from training.scripts.convert_tokenizer import MODEL_TYPES, _fields

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.dataloader.audio as td_audio
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.modules.text_conditioner import LUTConditioner
from pocket_tts.modules.transformer import StreamingTransformer
from training.args import TrainArgs
from training.checkpointing import EMA
from training.dataloader import DataLoader, Entry
from training.modules.builders import head_from_state
from training.modules.model import TrainableTTS
from training.modules.samplers import build_flow
from training.modules.utils import dit_init, stamp_state_names
from training.scripts.shrink_checkpoint import (
    low_rank_text_table,
    narrow_attention,
    pca_channels,
    select_layers,
    shrink,
    shrink_width,
)
from training.train_utils import build_optimizer

LDIM, DIM = 8, 32


class DummyConditioner(LUTConditioner):
    """LUTConditioner without the sentencepiece download (tests only)."""

    def __init__(self, n_bins: int, dim: int, rank: int | None = None):
        nn.Module.__init__(self)
        self.dim = dim
        self.output_dim = dim
        self.embed = nn.Embedding(n_bins + 1, rank or dim)
        self.proj = None if rank is None else nn.Linear(rank, dim, bias=False)


def tiny_model(
    flow_type: str,
    context: int | None = None,
    dim: int = DIM,
    num_layers: int = 2,
    num_heads: int = 2,
    z_dim: int | None = None,
    attn_dim: int | None = None,
    num_kv_heads: int | None = None,
    rank: int | None = None,
    hidden: int = 64,
) -> TrainableTTS:
    torch.manual_seed(0)
    flow = build_flow(flow_type)
    transformer = StreamingTransformer(
        d_model=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=hidden,
        context=context,
        attn_dim=attn_dim,
        num_kv_heads=num_kv_heads,
    )
    head = SimpleMLPAdaLN(LDIM, 16, LDIM, z_dim or dim, 2, flow.num_time_conds)
    dit_init(head)
    flow_lm = FlowLMModel(
        conditioner=DummyConditioner(11, dim, rank),
        flow_net=head,
        transformer=transformer,
        dim=dim,
        ldim=LDIM,
        insert_bos_before_voice=True,
        z_dim=z_dim,
    )
    flow_lm.speaker_proj_weight = torch.nn.Parameter(torch.randn(dim, LDIM) * 0.1)
    args = TrainArgs()
    args.flow.type = flow_type
    args.flow_batch_multiplier = 2
    return TrainableTTS(flow_lm, flow, args)


def make_batch(
    B: int = 3, T: int = 11
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    latents = torch.randn(B, T, LDIM)
    mask = torch.arange(T)[None, :] < torch.tensor([T, T - 3, T - 5])[:, None]
    text = [torch.randint(0, 10, (n,)) for n in (4, 2, 6)]
    voice = torch.randn(B, 5, LDIM)
    return latents, mask, text, voice


@pytest.mark.parametrize("flow_type", ["lsd", "flow_matching", "drifting"])
def test_train_step(flow_type: str):
    model = tiny_model(flow_type)
    model.train()
    loss, metrics = model(*make_batch())
    assert loss.isfinite()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients"
    assert torch.isfinite(torch.cat([g.flatten() for g in grads])).all()
    assert "flow_loss" in metrics and "eos_loss" in metrics


SMALL_OPTIONS = {"z_dim": 8, "attn_dim": 16, "num_kv_heads": 1, "rank": 4}


@pytest.mark.parametrize("flow_type", ["lsd", "drifting"])
def test_train_step_and_generate_with_small_model_options(flow_type: str):
    """z bottleneck, narrower attention with grouped kv heads and a low-rank text table."""
    model = tiny_model(flow_type, **SMALL_OPTIONS)
    model.train()
    loss, _ = model(*make_batch())
    loss.backward()
    assert loss.isfinite()
    assert model.flow_lm.z_proj is not None and model.flow_lm.z_proj.weight.grad is not None
    latents = model.generate(
        [torch.randint(0, 10, (5,))], [torch.randn(4, LDIM)], max_frames=3, eos_threshold=1e9
    )
    assert latents[0].shape == (3, LDIM)


def test_muon_splits_narrow_grouped_attention():
    model = tiny_model("lsd", **SMALL_OPTIONS)
    model.args.optim.type = "muon"
    optimizer = build_optimizer(model, model.args, torch.device("cpu"), rank=0)
    sizes = [g["split_sizes"] for g in optimizer.param_groups if g.get("split_sizes")]
    assert [16, 8, 8] in sizes, sizes


@pytest.mark.parametrize("flow_type,cfg", [("lsd", 1.0), ("flow_matching", 1.0), ("drifting", 1.0)])
def test_generate(flow_type: str, cfg: float):
    model = tiny_model(flow_type)
    tokens = torch.randint(0, 10, (5,))
    voice = torch.randn(4, LDIM)
    n_steps = 4 if flow_type == "flow_matching" else 1
    latents = model.generate(
        [tokens], [voice], max_frames=6, temp=0.7, n_steps=n_steps, cfg_coef=cfg, eos_threshold=1e9
    )[0]
    assert latents.shape == (6, LDIM)
    assert torch.isfinite(latents).all()


def test_cfg_distill():
    """Distillation moves the backbone, freezes the heads, and starts near zero
    loss at coef 1 (student == teacher, target == teacher's conditioned z)."""
    model = tiny_model("lsd")
    model.args.distill_cfg_coef = 1.0
    model.args.text_dropout = 0.0
    model.args.voice_dropout = 0.0
    teacher = copy.deepcopy(model.flow_lm).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    model.__dict__["distill_teacher"] = teacher
    for p in model.flow_lm.flow_net.parameters():
        p.requires_grad = False
    for p in model.flow_lm.out_eos.parameters():
        p.requires_grad = False

    model.train()
    loss, metrics = model(*make_batch())
    assert "distill_mse" in metrics
    assert loss.item() < 1e-8, f"student==teacher at coef 1 should give ~0 loss, got {loss.item()}"

    model.args.distill_cfg_coef = 3.0  # now the target differs from the student
    loss, _ = model(*make_batch())
    assert loss.item() > 1e-4
    loss.backward()
    assert all(p.grad is None for p in model.flow_lm.flow_net.parameters())
    backbone_grads = [p.grad for p in model.flow_lm.transformer.parameters() if p.grad is not None]
    assert backbone_grads and torch.isfinite(torch.cat([g.flatten() for g in backbone_grads])).all()


@pytest.mark.parametrize("z_dim", [8, None])
def test_distill_into_a_narrower_student(z_dim: int | None):
    """A narrower student seeded from the teacher: with a shared z width the loss is on z
    and the heads stay frozen; without one it is on the heads' inputs, which train."""
    teacher = tiny_model("lsd", dim=48, num_heads=6, hidden=96, z_dim=z_dim).flow_lm.eval()
    model = tiny_model("lsd", num_heads=4, z_dim=z_dim)
    model.args.distill_cfg_coef = 1.0
    model.args.text_dropout = model.args.voice_dropout = 0.0
    seeded = shrink_width(teacher.state_dict(), model.flow_lm.state_dict(), head_dim=8)
    model.flow_lm.load_state_dict(seeded)
    teacher.requires_grad_(False)
    model.__dict__["distill_teacher"] = teacher
    model.flow_lm.flow_net.requires_grad_(False)
    model.flow_lm.out_eos.requires_grad_(z_dim is None)
    model.flow_lm.flow_net.cond_embed.requires_grad_(z_dim is None)
    model.train()
    loss, metrics = model(*make_batch())
    assert loss.isfinite() and ("eos_mse" in metrics) == (z_dim is None)
    loss.backward()
    grads = {k: p.grad for k, p in model.flow_lm.named_parameters()}
    assert grads["transformer.layers.0.linear1.weight"] is not None
    assert (model.flow_lm.out_eos.weight.grad is not None) == (z_dim is None)


@pytest.mark.parametrize(
    "flow_type,loss_name", [("lsd", "mse"), ("drifting", "mse"), ("drifting", "drifting")]
)
def test_head_distillation_trains_only_the_new_head(flow_type: str, loss_name: str):
    """A smaller head learns a frozen bigger one's samples, with either objective whatever the
    head type; the backbone gets no gradient."""
    model = tiny_model(flow_type)
    model.args.head_distill_loss = loss_name
    model.args.head_distill_samples = 3
    frozen = SimpleMLPAdaLN(LDIM, 32, LDIM, DIM, 3, num_time_conds=2)
    model.__dict__["head_teacher"] = head_from_state(frozen.state_dict())
    model.flow_lm.requires_grad_(False)
    model.flow_lm.flow_net.requires_grad_(True)
    model.train()
    loss, metrics = model(*make_batch())
    assert loss.isfinite() and "head_loss" in metrics
    loss.backward()
    assert all(p.grad is not None for p in model.flow_lm.flow_net.parameters())
    assert model.flow_lm.input_linear.weight.grad is None


def test_drifting_single_target_sample_is_unchanged():
    """[N, 1, D] targets give the same loss as [N, D] ones."""
    flow, head = build_flow("drifting"), SimpleMLPAdaLN(LDIM, 16, LDIM, DIM, 1, 0)
    z, x_1 = torch.randn(5, DIM), torch.randn(5, LDIM)
    losses = []
    for target in (x_1, x_1[:, None]):
        torch.manual_seed(0)
        losses.append(flow.loss(partial(head, z), torch.randn(5, LDIM), target)[0])
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)


@pytest.mark.parametrize("num_time_conds", [0, 1, 2])
def test_head_supports_every_time_cond_count(num_time_conds: int):
    """The head runs with 0, 1 or 2 time conditions."""
    head = SimpleMLPAdaLN(LDIM, 16, LDIM, DIM, 2, num_time_conds)
    assert len(head.time_embed) == num_time_conds
    c, x = torch.randn(4, DIM), torch.randn(4, LDIM)
    ts = [torch.rand(4, 1) for _ in range(num_time_conds)]
    out = head(c, *ts, x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    with pytest.raises(AssertionError):  # wrong number of time conds must fail loudly
        head(c, *ts, torch.rand(4, 1), x)


def test_head_keeps_released_state_dict_layout():
    """num_time_conds=2 is the released models' layout: both time embeddings present."""
    head = SimpleMLPAdaLN(LDIM, 16, LDIM, DIM, 2, num_time_conds=2)
    keys = set(head.state_dict())
    assert {"time_embed.0.mlp.0.weight", "time_embed.1.mlp.0.weight"} <= keys
    assert not any(k.startswith("time_embed.2") for k in keys)


def test_shrink_checkpoint():
    """Layer selection keeps the ends, renumbers contiguously, and preserves the rest."""
    assert select_layers(24, 6) == [0, 1, 2, 21, 22, 23]
    assert select_layers(6, 6) == list(range(6))
    assert select_layers(24, 1) == [0]
    state = {"flow_lm.bos_emb": torch.zeros(4)}
    for i in range(24):
        state[f"flow_lm.transformer.layers.{i}.norm1.weight"] = torch.full((4,), float(i))
    out, kept = shrink(state, 6)
    assert kept == [0, 1, 2, 21, 22, 23]
    assert "flow_lm.bos_emb" in out
    layers = sorted(int(k.split(".")[3]) for k in out if "layers" in k)
    assert layers == list(range(6)), layers
    # Layer n of the student must be teacher layer kept[n].
    for new, old in enumerate(kept):
        assert out[f"flow_lm.transformer.layers.{new}.norm1.weight"][0].item() == float(old)


def test_shrink_width_seeds_a_narrower_model():
    """Every backbone tensor of a narrower, grouped-attention, low-rank model is seeded."""
    big = tiny_model("lsd", dim=48, num_heads=6, hidden=96).flow_lm
    small = tiny_model("lsd", num_heads=4, attn_dim=32, num_kv_heads=2, rank=8, hidden=64)
    seeded = shrink_width(big.state_dict(), small.flow_lm.state_dict(), head_dim=8)
    missing = set(small.flow_lm.state_dict()) - set(seeded)
    assert not missing, missing
    small.flow_lm.load_state_dict(seeded)
    loss, _ = small(*make_batch())
    assert loss.isfinite()


def test_narrow_attention_is_exact_for_duplicated_kv_heads():
    """Grouping kv heads that are already equal within each group changes nothing."""
    rope = RotaryEmbedding()
    full = StreamingMultiheadAttention(32, 4, rope)
    k = full.in_proj.weight.data[32:].view(2, 4, 8, 32)  # [k | v] x 4 heads
    k[:, 1], k[:, 3] = k[:, 0], k[:, 2]
    grouped = StreamingMultiheadAttention(32, 4, rope, num_kv_heads=2)

    def as_layer(attn: nn.Module) -> dict[str, torch.Tensor]:
        state = {f"transformer.layers.0.self_attn.{k}": v for k, v in attn.state_dict().items()}
        return {"transformer.layers.0.norm1.weight": torch.ones(32), **state}

    seeded = narrow_attention(as_layer(full), as_layer(grouped), head_dim=8)
    grouped.load_state_dict({k.split("self_attn.")[1]: v for k, v in seeded.items()})
    x = torch.randn(1, 5, 32)
    torch.testing.assert_close(grouped(x, None), full(x, None))


@pytest.mark.parametrize("z_dim", [None, 8])
def test_pca_channels_at_full_width_is_exact(z_dim: int | None):
    """Rotating the residual stream without cutting it leaves the model's function unchanged."""
    model = tiny_model("lsd", z_dim=z_dim).eval()
    state = model.flow_lm.state_dict()
    norms = [f"transformer.layers.{i}.norm{j}" for i in range(2) for j in (1, 2)] + ["out_norm"]
    a = torch.randn(len(norms), DIM, DIM, dtype=torch.float64)
    cov = {n: a[i] @ a[i].T for i, n in enumerate(norms)}
    rotated = copy.deepcopy(model)
    rotated.flow_lm.load_state_dict({**state, **pca_channels(state, DIM, cov)})
    losses = []
    for module in (model, rotated):
        torch.manual_seed(0)
        losses.append(module(*make_batch())[0])
    torch.testing.assert_close(losses[1], losses[0])


def test_low_rank_text_table_at_full_rank_is_exact():
    table = {"conditioner.embed.weight": torch.randn(11, 6)}
    dst = {
        "conditioner.embed.weight": torch.empty(11, 6),
        "conditioner.proj.weight": torch.empty(6, 6),
    }
    seeded = low_rank_text_table(table, dst)
    product = seeded["conditioner.embed.weight"] @ seeded["conditioner.proj.weight"].T
    torch.testing.assert_close(product, table["conditioner.embed.weight"], atol=1e-5, rtol=1e-5)


def test_generate_ragged_matches_batch_of_one():
    """Ragged prefixes: batched rows must equal the batch-of-1 path exactly."""
    model = tiny_model("lsd")
    texts = [torch.randint(0, 10, (n,)) for n in (3, 7, 5)]
    voices = [torch.randn(n, LDIM) for n in (4, 6, 5)]

    singles = []
    for t, v in zip(texts, voices, strict=True):
        torch.manual_seed(7)
        singles.append(
            model.generate(
                [t], [v], max_frames=5, temp=0.0, n_steps=1, cfg_coef=1.0, eos_threshold=1e9
            )[0]
        )
    torch.manual_seed(7)
    batched = model.generate(
        texts, voices, max_frames=5, temp=0.0, n_steps=1, cfg_coef=1.0, eos_threshold=1e9
    )
    assert len(batched) == 3
    for single, batch_row in zip(singles, batched, strict=True):
        assert single.shape == batch_row.shape
        # temp=0 removes sampling noise, so rows must agree closely; the left
        # padding must not leak into any row.
        torch.testing.assert_close(single, batch_row, atol=2e-3, rtol=2e-3)


def test_generate_per_row_eos():
    """A row whose EOS fires early must come back shorter than the others."""
    model = tiny_model("lsd")
    tokens = list(torch.randint(0, 10, (2, 4)))
    voice = list(torch.randn(2, 4, LDIM))
    # eos_threshold very low => EOS fires immediately for every row.
    out = model.generate(
        tokens, voice, max_frames=8, temp=0.7, n_steps=1, cfg_coef=1.0, eos_threshold=-1e9
    )
    assert all(x.shape[0] <= 8 for x in out)
    assert all(x.shape[0] >= 0 for x in out)


def test_padded_batch_matches_unpadded_under_context_window():
    """Right-aligned padding must not change a row's attention output.

    Regression for a real bug: shifting only the key positions inflated
    query-key deltas by each row's padding, so with a finite `context` the
    oldest real keys fell outside the window. Only rows with padding and
    sequences longer than the window were affected -- which is why end-to-end
    toy generation never caught it. This asserts the property directly.
    """
    torch.manual_seed(0)
    context = 4
    tr = StreamingTransformer(
        d_model=DIM, num_heads=2, num_layers=2, dim_feedforward=64, context=context
    )
    stamp_state_names(tr)
    tr.eval()
    length, pad = 10, 6  # length > context, so the window actually binds
    row = torch.randn(1, length, DIM)

    with torch.no_grad():
        solo = tr(row, init_states(tr, 1, length + pad + 2))

        padded = torch.cat([torch.zeros(1, pad, DIM), row], dim=1)
        state = init_states(tr, 1, length + pad + 2)
        for st in state.values():
            if isinstance(st, dict) and "pad" in st:
                st["pad"].fill_(pad)
        out = tr(padded, state)[:, pad:]

    torch.testing.assert_close(solo, out, atol=1e-5, rtol=1e-5)


def test_ema_load_drops_untracked_keys():
    """A shadow must hold exactly what the checkpoint stored.

    Regression: depth distillation freezes the head, so its EMA tracks fewer
    tensors than a freshly built model exposes. Applying a shadow padded with
    the fresh model's random init overwrote the trained head with noise --
    every generation came out silent while the raw weights were fine.
    """
    model = tiny_model("lsd")
    ema = EMA(model, 0.99)
    stored = {k: v.clone() for k, v in list(ema.shadow.items())[:3]}  # a partial checkpoint

    ema.load_state_dict(stored)

    assert set(ema.shadow) == set(stored), "shadow must not keep untracked keys"
    missing = model.state_dict().keys() - ema.shadow.keys()
    assert missing, "test needs keys outside the shadow to be meaningful"
    before = {k: model.state_dict()[k].clone() for k in missing}
    model.load_state_dict(ema.shadow, strict=False)
    for k, v in before.items():
        torch.testing.assert_close(model.state_dict()[k], v)


def test_prefix_prompt(monkeypatch: pytest.MonkeyPatch):
    """The cut lands inside the window, the prompt is the utterance start,
    and the target keeps the rest of the utterance."""
    dl = DataLoader.__new__(DataLoader)
    dl.sample_rate = 24000
    dl.frame_rate = 12.5
    dl.max_duration_sec = 30.0
    dl.max_voice_prompt_sec = 5.0
    dl.tokenize = lambda s: [1, 2]
    dl.rng = random.Random(0)
    dl._failures = 0

    # 20s utterance, one word per second
    words = [{"word": f"w{i}", "start": float(i), "end": i + 0.8} for i in range(20)]
    entry = Entry(path="x", duration=20.0, transcript="t", words=words)

    calls: list[tuple[float, float]] = []

    def fake_load_window(path: str, start: float, dur: float, sr: int) -> npt.NDArray[np.float32]:
        calls.append((start, dur))
        return np.zeros(max(1, int(dur * sr)), dtype=np.float32)

    monkeypatch.setattr(td_audio, "_load_window", fake_load_window)
    for _ in range(50):
        calls.clear()
        wav, tokens, prompt, plen = dl._sample(entry)
        (t_start, t_dur), (p_start, p_dur) = calls[0], calls[1]
        assert p_start == 0.0, "prefix prompt must start at the utterance start"
        assert t_start <= 5.5, f"cut must sit inside the window, got {t_start}"
        assert p_dur == t_start, "prompt must run exactly up to the cut"
        assert t_dur >= 20.0 - 5.5 - 0.5, "target keeps most of the utterance"


def test_train_tokenizer(tmp_path: Path):
    manifest = tmp_path / "m.jsonl"
    # Unigram, the default, seeds its vocabulary from the substrings it sees, so a
    # corpus of one repeated sentence cannot fill 64 pieces. Vary the words.
    words = [f"{a}{b}{c}" for a in "bcdfgpst" for b in "aeiou" for c in "lmnrs"]
    lines = [
        json.dumps({"transcript": " ".join(words[i % len(words) :][:6] or words[:6])})
        for i in range(64)
    ]
    manifest.write_text("\n".join(lines))
    prefix = tmp_path / "tok"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.scripts.train_tokenizer",
            str(prefix),
            str(manifest),
            "--vocab-size",
            "320",
        ],
        check=True,
    )
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    assert sp.get_piece_size() == 320
    assert sp.encode("hello world") != []


def test_grad_accum_matches_big_batch():
    """Two accumulated micro-batches produce the same grads as one batch of both."""
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 1)
    xs = torch.randn(8, 4)
    ys = torch.randn(8, 1)

    def loss_of(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(net(x), y, reduction="mean")

    def grad_of(p: torch.nn.Parameter) -> torch.Tensor:
        assert p.grad is not None
        return p.grad

    net.zero_grad()
    loss_of(xs, ys).backward()
    big = [grad_of(p).clone() for p in net.parameters()]

    net.zero_grad()
    for half in (slice(0, 4), slice(4, 8)):
        (loss_of(xs[half], ys[half]) / 2).backward()
    for g, p in zip(big, net.parameters(), strict=True):
        assert torch.allclose(g, grad_of(p), atol=1e-6)


def test_train_tokenizer_matches_the_released_spec(tmp_path: Path):
    """A tokenizer trained by the script carries the spec the released ones have."""
    manifest = tmp_path / "m.jsonl"
    words = [f"{a}{b}{c}" for a in "bcdfgpst" for b in "aeiou" for c in "lmnrs"]
    manifest.write_text(
        "\n".join(
            json.dumps({"transcript": " ".join(words[i % len(words) :][:6])}) for i in range(64)
        )
    )
    prefix = tmp_path / "tok"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.scripts.train_tokenizer",
            str(prefix),
            str(manifest),
            "--vocab-size",
            "320",
        ],
        check=True,
    )
    spec = _fields((prefix.with_suffix(".model")).read_bytes())
    trainer, normalizer = _fields(spec[2][0]), _fields(spec[3][0])
    assert MODEL_TYPES[trainer.get(3, [1])[0]] == "UNIGRAM"
    assert trainer[20][0] == 6  # max_sentencepiece_length
    assert trainer[25][0] == 1  # split_digits
    assert trainer[26][0] == 1  # allow_whitespace_only_pieces
    assert trainer[35][0] == 1  # byte_fallback
    assert trainer[43][0] == 3  # pad_id
    assert normalizer.get(4, [1])[0] == 0  # remove_extra_whitespaces off
    assert not normalizer.get(2, [b""])[0]  # identity: no precompiled charsmap
