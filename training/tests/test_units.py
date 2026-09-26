"""Unit tests for the pure functions the training loop leans on."""

from typing import Any

import pytest
import torch
from torch import nn

from training.args import OptimArgs, TrainArgs
from training.checkpointing import EMA
from training.modules.samplers import LSD, Drifting
from training.scripts.shrink_checkpoint import select_layers, shrink
from training.train_utils import lr_at


def _args(**kw: Any) -> TrainArgs:  # noqa: ANN401 -- TrainArgs/OptimArgs passthrough
    optim = OptimArgs(**kw.pop("optim", {}))
    return TrainArgs(optim=optim, **kw)


class TestLrSchedule:
    def test_warmup_ends_at_the_configured_lr(self):
        args = _args(optim={"lr": 2e-4, "warmup_steps": 100, "schedule": "constant"})
        assert lr_at(0, args) == pytest.approx(2e-6)
        assert lr_at(99, args) == pytest.approx(2e-4)
        assert lr_at(100, args) == pytest.approx(2e-4)

    def test_cosine_decays_to_the_floor_at_max_steps(self):
        args = _args(max_steps=1000, optim={"lr": 1e-3, "warmup_steps": 0, "schedule": "cosine"})
        assert lr_at(0, args) == pytest.approx(1e-3, rel=1e-3)
        assert lr_at(1000, args) == pytest.approx(0.0, abs=1e-9)
        assert lr_at(500, args) == pytest.approx(5e-4, rel=1e-2)

    def test_cosine_respects_lr_min_ratio(self):
        args = _args(
            max_steps=1000,
            optim={"lr": 1e-3, "warmup_steps": 0, "schedule": "cosine", "lr_min_ratio": 0.1},
        )
        assert lr_at(1000, args) == pytest.approx(1e-4, rel=1e-6)

    def test_constant_stays_flat(self):
        args = _args(max_steps=1000, optim={"lr": 2e-4, "warmup_steps": 0, "schedule": "constant"})
        assert lr_at(10, args) == lr_at(999, args) == pytest.approx(2e-4)


class TestSelectLayers:
    def test_keeps_the_ends_of_the_stack(self):
        assert select_layers(24, 6) == [0, 1, 2, 21, 22, 23]

    def test_identity_when_depths_match(self):
        assert select_layers(6, 6) == list(range(6))

    def test_rejects_growing_the_stack(self):
        with pytest.raises(AssertionError):
            select_layers(6, 24)

    def test_shrink_renumbers_kept_layers_contiguously(self):
        state = {f"transformer.layers.{i}.w": torch.tensor([float(i)]) for i in range(24)}
        state["out_eos.weight"] = torch.zeros(1)
        out, keep = shrink(state, 6)
        assert keep == [0, 1, 2, 21, 22, 23]
        assert [k for k in out if k.startswith("transformer")] == [
            f"transformer.layers.{i}.w" for i in range(6)
        ]
        # renumbering must preserve which teacher layer each slot came from
        assert out["transformer.layers.3.w"].item() == 21.0
        assert "out_eos.weight" in out, "non-backbone tensors are copied unchanged"


class TestEMA:
    def _model(self) -> nn.Linear:
        m = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)
        return m

    def test_tracks_only_trainable_parameters(self):
        m = self._model()
        m.bias.requires_grad_(False)
        ema = EMA(m, 0.9)
        assert set(ema.shadow) == {"weight"}

    def test_update_moves_towards_the_live_weights(self):
        m = self._model()
        ema = EMA(m, 0.5)
        with torch.no_grad():
            m.weight.fill_(3.0)
        ema.update(m)
        assert ema.shadow["weight"][0, 0].item() == pytest.approx(2.0)

    def test_loading_a_partial_shadow_keeps_untracked_weights(self):
        """A distilled student freezes its heads, so its shadow covers only part
        of the model; the rest must survive loading rather than be overwritten."""
        m = self._model()
        ema = EMA(m, 0.9)
        ema.load_state_dict({"weight": torch.full((2, 2), 7.0)})
        assert ema.shadow["weight"][0, 0].item() == pytest.approx(7.0)
        assert "bias" not in ema.shadow or ema.shadow["bias"][0].item() == pytest.approx(1.0)


class TestDistillProb:
    def _v(self, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * (s + t)

    def test_skip_rate_matches_probability(self):
        m = LSD(distill_prob=0.25)
        m.train()
        outs = [m.loss(self._v, torch.randn(2, 4), torch.randn(2, 4)) for _ in range(400)]
        rate = sum("flow_distill" in met for _, met, _ in outs) / 400
        assert 0.15 < rate < 0.35

    def test_ranks_agree_on_every_draw(self):
        """The skip draw is rank-synchronized: two instances (= two DDP ranks)
        must take the same branch on the same step."""
        a, b = LSD(distill_prob=0.25), LSD(distill_prob=0.25)
        a.train(), b.train()
        sa = [
            "flow_distill" in a.loss(self._v, torch.randn(2, 4), torch.randn(2, 4))[1]
            for _ in range(100)
        ]
        sb = [
            "flow_distill" in b.loss(self._v, torch.randn(2, 4), torch.randn(2, 4))[1]
            for _ in range(100)
        ]
        assert sa == sb

    def test_eval_mode_always_computes_the_full_loss(self):
        """Valid losses must stay comparable across distill_prob settings."""
        m = LSD(distill_prob=0.25)
        m.eval()
        for _ in range(30):
            _, met, _ = m.loss(self._v, torch.randn(2, 4), torch.randn(2, 4))
            assert "flow_distill" in met


class TestDrifting:
    def _head(self) -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 4))

    def test_learned_temperature_gets_a_gradient(self):
        m, head = Drifting(temp=1.0, num_neg=8), self._head()
        loss, _, _ = m.loss(head, torch.randn(6, 4), torch.randn(6, 4))
        loss.mean().backward()
        assert m.temp.grad is not None and m.temp.grad.abs().item() > 0

    def test_fixed_temperature_keeps_the_resumed_value(self):
        """temp_loss_weight 0 (the fine-tune) freezes the temperature but keeps the
        parameter, so a stage-1 checkpoint that learned it loads as-is."""
        learned = Drifting(temp=10.0).state_dict()
        learned["temp"] = torch.tensor([0.05])
        m = Drifting(temp_loss_weight=0.0)
        m.load_state_dict(learned)
        assert not m.temp.requires_grad and m.temp.item() == pytest.approx(0.05)

    def test_decode_is_the_head_output(self):
        head, x_0 = self._head(), torch.randn(3, 4)
        assert torch.equal(Drifting().decode(head, x_0), head(x_0))
