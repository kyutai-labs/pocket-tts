"""CALM training objectives on the per-frame latent head.

All objectives share one interface:

    loss(v_t, x_0, x_1) -> (loss [N], metrics, t)
    decode(v_t, x_0, num_steps) -> x_1_hat

where `v_t = partial(head, z)` is the AdaLN-MLP head conditioned on the
backbone output for one frame, `x_0` is noise and `x_1` the target latent.

- FlowMatching: standard OT conditional flow matching (arXiv:2210.02747),
  1 time cond, multi-step decode.
- LSD: Lagrangian Self Distillation (arXiv:2505.18825), 2 time conds (s, t),
  1-step (or few-step) decode. This is the objective of the released
  pocket-tts models.
"""

from typing import Literal

import torch
from torch import nn

from pocket_tts.models.flow_lm import FlowNet, lsd_decode, ot_decode

from .utils import MLP, f_grad_x_only, zero_init


class FlowType(nn.Module):
    num_time_conds: int = 1

    def loss(
        self, v_t: FlowNet, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def decode(self, v_t: FlowNet, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        raise NotImplementedError


class FlowMatching(FlowType):
    """Optimal Transport conditional Flow Matching (https://arxiv.org/abs/2210.02747)."""

    num_time_conds = 1

    def __init__(self, sig_min: float = 0.001):
        super().__init__()
        self.sig_min = sig_min

    def loss(
        self, v_t: FlowNet, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        t = torch.rand_like(x_1[..., :1])
        x_t = (1 - (1 - self.sig_min) * t) * x_0 + t * x_1
        v_psi = v_t(t, x_t)
        d_psi = x_1 - (1 - self.sig_min) * x_0
        loss = ((v_psi - d_psi) ** 2).mean(dim=-1)
        return loss, {"loss": loss.mean()}, t

    def decode(self, v_t: FlowNet, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        return ot_decode(v_t, x_0, num_steps)


class LSD(FlowType):
    """Lagrangian Self Distillation (https://arxiv.org/pdf/2505.18825).

    Defaults are the champion recipe. stopgrad_type "minimal" keeps the
    self-distillation target's direct x_t dependence (f_grad_x_only)
    differentiable; "classic" detaches it entirely. They match in quality
    and training speed.
    """

    num_time_conds = 2

    def __init__(
        self,
        p_equal: float = 0.75,
        lognorm_mean: float = 0.4,
        lognorm_std: float = 1.0,
        w_t_dims: int = 32,
        w_t_depth: int = 3,
        normalize: bool = True,
        stopgrad_type: Literal["classic", "minimal"] = "minimal",
        distill_prob: float = 0.25,
    ):
        super().__init__()
        assert 0.0 < distill_prob <= 1.0, distill_prob
        # The self-distillation term costs ~2 extra flow-net forwards (jvp +
        # endpoint target) plus their backward, and is oversampled at every
        # step: computing it on a quarter of steps leaves its loss unchanged
        # and trains ~9% faster. Skipped-step losses divide the term by the
        # probability, keeping the expected gradient identical. Lower values
        # (e.g. 0.175) train faster still and leave the distill loss ~10%
        # higher; human evals hear no difference down to 0.175. The draw uses
        # a fixed-seed generator so every DDP rank takes the same branch.
        self.distill_prob = distill_prob
        self._skip_rng = torch.Generator().manual_seed(0)
        self.p_equal = p_equal
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.normalize = normalize
        self.stopgrad_type = stopgrad_type
        if normalize:
            # Learned per-(s, t) uncertainty weighting of the two loss terms.
            self.w_s_t = MLP(2, [w_t_dims] * w_t_depth + [1])
            self.w_s_t.apply(zero_init)

    def sample_t(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.randn_like(x[..., :1]) * self.lognorm_std + self.lognorm_mean)

    def sample_s_t(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.randn_like(x[..., :2]) * self.lognorm_std + self.lognorm_mean
        return torch.sigmoid(logits.min(-1)[0])[..., None], torch.sigmoid(logits.max(-1)[0])[
            ..., None
        ]

    def loss(
        self, v_t: FlowNet, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        x, e = x_1, x_0
        metrics: dict[str, torch.Tensor] = {}

        # Diagonal (instantaneous flow-matching) term at s == t.
        t = self.sample_t(x)
        x_t = t * x + (1 - t) * e
        v = x - e
        flow_diag = (v_t(t, t, x_t) - v).square().sum(dim=-1)
        metrics["flow_diag"] = flow_diag.mean()
        if self.normalize:
            diag_logvar = self.w_s_t(t, t).squeeze(-1)
            flow_diag = flow_diag * diag_logvar.exp() / x_0.shape[-1] - diag_logvar

        # Self-distillation term: the (s -> t) jump must match the instantaneous
        # flow at its own endpoint. This is LSD's own objective -- unrelated to
        # the CFG/depth distillation against a teacher, which logs distill_loss.
        # Skipped probabilistically during training only, so valid losses stay
        # comparable across settings.
        skip = (
            self.training
            and self.distill_prob < 1.0
            and float(torch.rand((), generator=self._skip_rng)) >= self.distill_prob
        )
        if skip:
            return self.p_equal * flow_diag, metrics, t
        s, t = self.sample_s_t(x)
        x_s = s * x + (1 - s) * e
        # The stub's has_aux=True 3-tuple leaks into the return type.
        vt, dvdt = torch.func.jvp(  # ty: ignore[invalid-assignment]
            v_t, (s, t, x_s), (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(x_s))
        )
        x_t = x_s + (t - s) * vt
        dxdt = vt + (t - s) * dvdt
        if self.stopgrad_type == "minimal":
            u_tgt = f_grad_x_only(lambda y: v_t(t, t, y), x_t)
        else:
            with torch.no_grad():
                u_tgt = v_t(t, t, x_t)
        flow_distill = (dxdt - u_tgt).square().sum(dim=-1)
        metrics["flow_distill"] = flow_distill.mean()
        if self.normalize:
            distill_logvar = self.w_s_t(s, t).squeeze(-1)
            flow_distill = flow_distill * distill_logvar.exp() / x_0.shape[-1] - distill_logvar

        distill_w = (1 - self.p_equal) / (self.distill_prob if self.training else 1.0)
        loss = self.p_equal * flow_diag + distill_w * flow_distill
        return loss, metrics, t

    def decode(self, v_t: FlowNet, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        return lsd_decode(v_t, x_0, num_steps)


FLOW_TYPES: dict[str, type[FlowType]] = {"flow_matching": FlowMatching, "lsd": LSD}


def build_flow(name: str, **kwargs: object) -> FlowType:
    return FLOW_TYPES[name](**kwargs)
