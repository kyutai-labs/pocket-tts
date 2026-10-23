"""Muon optimizer (orthogonalized momentum) with an aux AdamW.

Muon applies to 2D hidden weights only; embeddings, heads, gains and 1D
params need AdamW, so a single optimizer instance carries both update rules
selected per param group via `use_muon` (keeps checkpointing's single
`optimizer.state_dict()` contract).

The orthogonalization is the Polar Express polynomial iteration (Amsel et al.,
arXiv 2505.16932): a composition of degree-5 polynomials whose coefficients are
optimal per step, evaluated in bfloat16 with matrix products only. Fused
weights (q/k/v projections, adaLN modulations) are orthogonalized per logical
block through the group's `split_sizes`.

References: Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (github.com/KellerJordan/Muon); Liu et al., "Muon is scalable for
LLM training" (arXiv 2502.16982) for the RMS-matched update scale.
"""

from collections.abc import Callable
from itertools import repeat
from typing import Any

import torch
from torch.optim.adamw import adamw as functional_adamw

# Optimal degree-5 coefficients per iteration (l=1e-3, safety 1e-2, cushion 0.02),
# from github.com/NoahAmsel/PolarExpress; the last one is the fixed point.
_POLAR_EXPRESS_RAW = [
    (8.287212018145630, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.947849916737911, 0.544843108292660),
    (3.948690853482295, -2.908902115962949, 0.551819139437014),
    (3.318419657370602, -2.488488024314874, 0.510048940123720),
    (2.300652019954817, -1.668903984574749, 0.418807311952567),
    (1.891301407787398, -1.267995827194587, 0.376804089485248),
    (1.875001480853448, -1.250001645399949, 0.375000164547425),
    (1.875, -1.25, 0.375),
]
_POLAR_EXPRESS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_EXPRESS_RAW[:-1]] + [
    _POLAR_EXPRESS_RAW[-1]
]


def polar_express(G: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """Approximate polar factor of G (semi-orthogonal, same shape), batched over leading dims.

    Polar Express applies `steps` degree-5 polynomials X <- aX + bX^3 + cX^5 to the
    normalized input; bf16 throughout, matrix products only.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    coeffs = _POLAR_EXPRESS[:steps] + list(
        repeat(_POLAR_EXPRESS[-1], max(0, steps - len(_POLAR_EXPRESS)))
    )
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


_polar_express_compiled = torch.compile(polar_express, dynamic=False)


def _orthogonalize(G: torch.Tensor, steps: int) -> torch.Tensor:
    if G.is_cuda:
        return _polar_express_compiled(G, steps)
    return polar_express(G, steps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Param groups with use_muon=True get Muon; the rest get AdamW.

    Muon group keys: lr, momentum, polar_steps, weight_decay, split_sizes
    (row block sizes of a fused weight; default one block), rms_match (scale
    the orthogonal update by 0.2 * sqrt(max(rows, cols)) so its RMS matches an
    AdamW update, instead of sqrt(max(1, rows / cols))). AdamW group keys: lr,
    betas, eps, weight_decay. The training loop rescales each group's lr through
    group["lr_scale"].
    """

    def __init__(self, param_groups: list[dict[str, Any]]):
        for g in param_groups:
            if g.get("use_muon"):
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("polar_steps", 6)
                g.setdefault("weight_decay", 0.01)
                g.setdefault("split_sizes", None)
                g.setdefault("rms_match", False)
            else:
                g.setdefault("lr", 2e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.1)
            g["initial_lr"] = g["lr"]
        super().__init__(param_groups, {})
        self._adamw_steps: dict[int, torch.Tensor] = {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._adamw_steps.clear()
        for state in self.state.values():
            if "step" in state:
                state["step"] = int(state["step"])

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # ty: ignore[invalid-method-override]
        assert closure is None
        for group in self.param_groups:
            if group.get("use_muon"):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return None

    @staticmethod
    def _blocks(p: torch.Tensor, g: torch.Tensor, sizes: list[int] | None) -> list[torch.Tensor]:
        """Logical matrices of a (possibly fused) weight, as views of its lookahead grad."""
        g = g.view(p.size(0), -1)
        if not sizes:
            return [g]
        assert sum(sizes) == g.size(0), (sizes, g.shape)
        return list(torch.split(g, sizes, dim=0))

    def _muon_step(self, group: dict[str, Any]) -> None:
        # Nesterov lookahead per param (multi-tensor ops), then one batched polar
        # iteration per distinct block shape: sequential small iterations are
        # launch-bound, stacked they are a handful of large bmm calls.
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        bufs = []
        for p, g in zip(params, grads):
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            bufs.append(state["momentum_buffer"])
        torch._foreach_lerp_(bufs, grads, 1 - group["momentum"])
        looks = torch._foreach_lerp(grads, bufs, group["momentum"])
        # (param index, block index) -> block; batched per block shape
        by_shape: dict[tuple[int, ...], list[tuple[int, int, torch.Tensor]]] = {}
        blocks_of: list[list[torch.Tensor]] = []
        for i, (p, g) in enumerate(zip(params, looks)):
            blocks = self._blocks(p, g, group["split_sizes"])
            blocks_of.append(blocks)
            for j, b in enumerate(blocks):
                by_shape.setdefault(tuple(b.shape), []).append((i, j, b))
        # Under DDP every rank holds identical grads and momentum, so the orthogonalization
        # is sharded: rank r computes a contiguous slice of each shape's blocks, then one
        # all_gather per shape puts every orthogonalized block on every rank.
        dist = torch.distributed
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if world > 1 else 0
        out: dict[tuple[int, int], torch.Tensor] = {}
        for shape, items in by_shape.items():
            rows, cols = int(shape[0]), int(shape[1])
            n = len(items)
            per = -(-n // world)
            mine = [b for k, (_, _, b) in enumerate(items) if k // per == rank]
            if mine:
                ortho = _orthogonalize(torch.stack(mine), group["polar_steps"])
            else:
                ortho = torch.empty(
                    (0, rows, cols), dtype=torch.bfloat16, device=items[0][2].device
                )
            if world > 1:
                pad = per - ortho.size(0)
                if pad:
                    ortho = torch.cat(
                        [
                            ortho,
                            torch.zeros((pad, rows, cols), dtype=ortho.dtype, device=ortho.device),
                        ]
                    )
                gathered = torch.empty(
                    (per * world, rows, cols), dtype=ortho.dtype, device=ortho.device
                )
                dist.all_gather_into_tensor(gathered, ortho.contiguous())
                ortho = gathered[:n]
            if group["rms_match"]:
                scale = 0.2 * max(rows, cols) ** 0.5
            else:
                scale = (
                    max(1.0, rows / cols) ** 0.5
                )  # keeps update RMS ~constant across aspect ratios
            for (i, j, _), o in zip(items, ortho):
                out[(i, j)] = o * scale
        lr, wd = group["lr"], group["weight_decay"]
        torch._foreach_mul_(params, 1 - lr * wd)
        updates = [
            torch.cat([out[(i, j)] for j in range(len(blocks_of[i]))], dim=0)
            .reshape(p.shape)
            .to(p.dtype)
            for i, p in enumerate(params)
        ]
        torch._foreach_add_(params, updates, alpha=-lr)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        fused = all(p.is_cuda for p in params)
        grads, avgs, squares, counters = [], [], [], []
        for p in params:
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            device = p.device if fused else torch.device("cpu")
            counter = self._adamw_steps.get(id(p))
            if counter is None or counter.device != device:
                counter = torch.tensor(float(state["step"]), dtype=torch.float32, device=device)
                self._adamw_steps[id(p)] = counter
            state["step"] += 1
            grads.append(p.grad)
            avgs.append(state["exp_avg"])
            squares.append(state["exp_avg_sq"])
            counters.append(counter)
        beta1, beta2 = group["betas"]
        functional_adamw(
            params,
            grads,
            avgs,
            squares,
            [],
            counters,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
            foreach=not fused,
            fused=fused,
        )
