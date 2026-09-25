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
- Drifting: one-step noise-to-data generator trained with a kernel drift
  field (arXiv:2602.04770), no time cond.
"""

import math
from typing import Literal

import torch
from torch import nn

from pocket_tts.models.flow_lm import FlowNet, drifting_decode, lsd_decode, ot_decode

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


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise Euclidean distances, [B, N, D] x [B, M, D] -> [B, N, M]."""
    sq = (
        x.square().sum(-1)[:, :, None]
        + y.square().sum(-1)[:, None, :]
        - 2 * torch.einsum("bnd,bmd->bnm", x, y)
    )
    return sq.clamp(min=eps).sqrt()


class Drifting(FlowType):
    """Drifting generative model (https://arxiv.org/abs/2602.04770).

    A one-step noise-to-data generator. Each step draws `num_neg` generations per
    position (the negatives) and moves the generator output along a kernel drift
    field V: attraction to the data sample minus repulsion from the other
    generations, via the fixed-point loss ||x - stopgrad(x + V)||^2. The kernel is
    Laplacian, on distances normalized by their per-position mean.

    The kernel temperature starts at `temp` and, with temp_loss_weight > 0, is
    learned by maximizing the data sample's share of the kernel mass; with 0 it
    stays fixed (at `temp`, or at the learned value when resuming a run that
    learned it).
    """

    num_time_conds = 0
    # The backward through num_neg vmapped head draws per position overflows in
    # bf16; the head is small, so the loss runs in fp32.
    fp32_loss = True

    def __init__(
        self,
        temp: float = 10.0,
        temp_loss_weight: float = 1.0,
        num_neg: int = 64,
        normalize_force: bool | Literal["batch"] = "batch",
        min_scale: float = 1e-3,
        min_temp: float = 1e-3,
    ):
        super().__init__()
        assert normalize_force in (True, False, "batch"), normalize_force
        self.num_neg = num_neg
        self.normalize_force = normalize_force
        self.min_scale = min_scale
        self.min_temp = min_temp
        self.temp_loss_weight = temp_loss_weight
        self.temp = nn.Parameter(torch.full((1,), temp), requires_grad=temp_loss_weight > 0)

    def _drift(
        self,
        dists: torch.Tensor,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        temp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logit = -dists / temp  # [N, num_neg, 1 + num_neg]: each generation vs data + generations
        # Log-mass of the data sample relative to a flat kernel (0 = chance).
        pos_log_mass = logit.log_softmax(dim=-1)[..., 0] + math.log(self.num_neg)
        A = (logit.softmax(dim=-1) * logit.softmax(dim=-2)).clamp(min=1e-6).sqrt().detach()
        A_pos, A_neg = A[..., :1], A[..., 1:]
        W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)
        force = W_pos @ y_pos - W_neg @ y_neg
        force = force - (W_pos.sum(dim=-1, keepdim=True) - W_neg.sum(dim=-1, keepdim=True)) * x
        rms = force.square().mean((-1, -2), keepdim=True).clamp(min=1e-8).sqrt()
        if self.normalize_force == "batch":
            force = force / rms.mean(dim=0, keepdim=True)
        elif self.normalize_force:
            force = force / rms
        return force, pos_log_mass

    def loss(
        self, v_t: FlowNet, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        e = torch.randn(self.num_neg, *x_0.shape, device=x_0.device, dtype=x_0.dtype)
        x = torch.vmap(v_t, out_dims=1)(e)  # [N, num_neg, D]
        y_neg = x.detach()
        y_pos = x_1[:, None]  # [N, 1, D]
        with torch.no_grad():
            dists = torch.cat([cdist(y_neg, y_pos), cdist(y_neg, y_neg)], dim=-1)
            n = self.num_neg + 1
            scale = (dists * n / (n - 1)).mean(dim=(-1, -2), keepdim=True).clamp(min=self.min_scale)
            # A generation never attracts or repels itself.
            self_mask = torch.eye(self.num_neg, device=x.device) * 100.0
            dists = dists / scale + torch.cat(
                [torch.zeros_like(dists[..., :1]), self_mask.expand_as(dists[..., 1:])], dim=-1
            )
            scale_input = scale / x.shape[-1] ** 0.5
        x, y_pos, y_neg = (z / scale_input for z in (x, y_pos, y_neg))
        temp = self.temp.clamp(min=self.min_temp)
        V, pos_log_mass = self._drift(dists, x.detach(), y_pos, y_neg, temp)
        loss = (x - (x + V).detach()).square().mean(dim=(-1, -2))
        # Per position, the generation that puts the most mass on the data sample sets the temperature.
        temp_loss = -pos_log_mass.amax(dim=-1).mean()
        metrics = {
            "drifting_loss": loss.mean(),
            "temp": self.temp.detach().squeeze(),
            "std_mean": y_neg.std(dim=1).norm(dim=-1).mean(),
        }
        return loss + self.temp_loss_weight * temp_loss, metrics, torch.zeros_like(x_0[..., :1])

    def decode(self, v_t: FlowNet, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        return drifting_decode(v_t, x_0)


FLOW_TYPES: dict[str, type[FlowType]] = {
    "flow_matching": FlowMatching,
    "lsd": LSD,
    "drifting": Drifting,
}


def build_flow(name: str, **kwargs: object) -> FlowType:
    return FLOW_TYPES[name](**kwargs)
