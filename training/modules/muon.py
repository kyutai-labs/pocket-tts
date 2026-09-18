"""Muon optimizer (Newton-Schulz orthogonalized momentum) with an aux AdamW.

Muon applies to 2D hidden weights only; embeddings, heads, gains and 1D
params need AdamW, so a single optimizer instance carries both update rules
selected per param group via `use_muon` (keeps checkpointing's single
`optimizer.state_dict()` contract).

Reference: Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (github.com/KellerJordan/Muon).
"""

from collections.abc import Callable

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate orthogonalization of G (semi-orthogonal factor of its SVD), batched over
    a leading dim when G is 3D: each [rows, cols] slice is orthogonalized on its own.

    Quintic Newton-Schulz iteration in bf16; coefficients from the reference
    implementation, tuned for fast convergence at slight loss of precision.
    """
    assert G.ndim in (2, 3)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Param groups with use_muon=True get Muon; the rest get AdamW.

    Group keys: muon groups use (lr, momentum, ns_steps, weight_decay);
    adamw groups use (lr, betas, eps, weight_decay). The training loop's
    scheduler rescales each group's lr through group["initial_lr"].
    """

    def __init__(self, param_groups: list[dict]):
        for g in param_groups:
            if g.get("use_muon"):
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("ns_steps", 5)
                g.setdefault("weight_decay", 0.01)
                g.setdefault("split", 1)
            else:
                g.setdefault("lr", 2e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.1)
            g["initial_lr"] = g["lr"]
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        assert closure is None
        for group in self.param_groups:
            if group.get("use_muon"):
                self._muon_step(group)
            else:
                self._adamw_step(group)

    def _muon_step(self, group: dict) -> None:
        # Nesterov lookahead per param (multi-tensor ops), then one batched Newton-Schulz per
        # distinct shape: 24 layers x a few matrices as sequential small NS iterations were
        # launch-bound (~45% of the step), stacked they are a handful of large bmm calls.
        split = group.get("split", 1)
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
        by_shape: dict[tuple, list] = {}
        for p, g in zip(params, looks):
            g = g.view(p.size(0), -1)
            if split > 1:
                g = g.view(split, g.size(0) // split, g.size(1))
            by_shape.setdefault(tuple(g.shape), []).append((p, g))
        # Under DDP every rank holds identical grads and momentum, so the orthogonalization is
        # sharded: rank r computes the matrices with index % world == r, then one all_gather
        # per shape puts every orthogonalized matrix on every rank.
        dist = torch.distributed
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if world > 1 else 0
        lr, wd = group["lr"], group["weight_decay"]
        for shape, items in by_shape.items():
            rows, cols = shape[-2], shape[-1]
            scale = max(1.0, rows / cols) ** 0.5  # keeps update RMS ~constant across aspect ratios
            n = len(items)
            per = -(-n // world)  # matrices per rank, padded so all_gather shapes agree
            mine = [g for i, (_, g) in enumerate(items) if i // per == rank]
            if mine:
                G = torch.stack(mine)
                flat = G.reshape(-1, rows, cols) if G.ndim == 4 else G
                ortho = _zeropower_via_newtonschulz5(flat, group["ns_steps"]).reshape(G.shape)
            else:
                ortho = torch.empty((0,) + shape, dtype=torch.bfloat16, device=items[0][1].device)
            if world > 1:
                pad = per - ortho.size(0)
                if pad:
                    ortho = torch.cat(
                        [ortho, torch.zeros((pad,) + shape, dtype=ortho.dtype, device=ortho.device)]
                    )
                out = torch.empty((per * world,) + shape, dtype=ortho.dtype, device=ortho.device)
                dist.all_gather_into_tensor(out, ortho.contiguous())
                ortho = out[:n]
            ps = [p for p, _ in items]
            torch._foreach_mul_(ps, 1 - lr * wd)
            torch._foreach_add_(
                ps, [o.reshape(p.shape).to(p.dtype) for p, o in zip(ps, ortho)], alpha=-lr * scale
            )

    def _adamw_step(self, group: dict) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        m, v = [], []
        for p, g in zip(params, grads):
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
            state["step"] += 1
            m.append(state["exp_avg"])
            v.append(state["exp_avg_sq"])
        step = self.state[params[0]]["step"]
        beta1, beta2 = group["betas"]
        torch._foreach_lerp_(m, grads, 1 - beta1)
        torch._foreach_mul_(v, beta2)
        torch._foreach_addcmul_(v, grads, grads, value=1 - beta2)
        bias1 = 1 - beta1**step
        bias2 = 1 - beta2**step
        denom = torch._foreach_div(v, bias2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group["eps"])
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_addcdiv_(params, m, denom, value=-group["lr"] / bias1)
