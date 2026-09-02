"""CPU smoke tests: tiny model, one training step per objective, short generation."""

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import sentencepiece as spm
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.dataloader as td
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.modules.text_conditioner import LUTConditioner
from pocket_tts.modules.transformer import StreamingTransformer
from training.args import TrainArgs
from training.checkpointing import EMA
from training.dataloader import DataLoader, Entry
from training.modules.model import TrainableTTS
from training.modules.samplers import build_flow
from training.modules.utils import dit_init, stamp_state_names
from training.scripts.shrink_checkpoint import select_layers, shrink

LDIM, DIM = 8, 32


class DummyConditioner(LUTConditioner):
    """LUTConditioner without the sentencepiece download (tests only)."""

    def __init__(self, n_bins: int, dim: int):
        nn.Module.__init__(self)
        self.dim = dim
        self.output_dim = dim
        self.embed = nn.Embedding(n_bins + 1, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


def tiny_model(flow_type: str, context: int | None = None) -> TrainableTTS:
    torch.manual_seed(0)
    flow = build_flow(flow_type)
    transformer = StreamingTransformer(
        d_model=DIM, num_heads=2, num_layers=2, dim_feedforward=64, context=context
    )
    head = SimpleMLPAdaLN(LDIM, 16, LDIM, DIM, 2, flow.num_time_conds)
    dit_init(head)
    flow_lm = FlowLMModel(
        conditioner=DummyConditioner(11, DIM),
        flow_net=head,
        transformer=transformer,
        dim=DIM,
        ldim=LDIM,
        insert_bos_before_voice=True,
    )
    flow_lm.speaker_proj_weight = torch.nn.Parameter(torch.randn(DIM, LDIM) * 0.1)
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


@pytest.mark.parametrize("flow_type", ["lsd", "flow_matching"])
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


@pytest.mark.parametrize("flow_type,cfg", [("lsd", 1.0), ("flow_matching", 1.0)])
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

    monkeypatch.setattr(td, "_load_window", fake_load_window)
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
    lines = [
        json.dumps({"transcript": f"hello world number {i} testing tokenizers"}) for i in range(64)
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
            "64",
        ],
        check=True,
    )
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    assert sp.get_piece_size() == 64
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
