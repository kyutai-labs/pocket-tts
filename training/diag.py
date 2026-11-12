"""Log-only training diagnostics (TrainArgs.diag, off by default).

MFU: model FLOPs of one optimizer step, counted once with torch.utils.flop_counter.FlopCounterMode
on a real step (forward + backward of every micro-batch, all ranks summed), divided by
(step time x world size x PEAK_FLOPS). FlopCounterMode counts SDPA at the full T x T cost (the
masked kernels do compute it all), optimizer and all-reduce at 0.

Gradient noise scale (McCandlish et al. 2018, arXiv 1812.06162, appendix A): B_simple = tr(S)/|G|^2
from the unbiased two-batch-size estimator. |G_small|^2 = squared norm of a gradient over B_small
rows (each rank's own step gradient before the all-reduce, or one micro-batch's), |G_big|^2 that of
the all-reduced gradient over B_big rows:
    |G|^2 = (B_big |G_big|^2 - B_small |G_small|^2) / (B_big - B_small)
    tr(S) = (|G_small|^2 - |G_big|^2) / (1 / B_small - 1 / B_big)
Both are EMA'd separately before the ratio (single estimates are very noisy).
"""

import time
from collections import defaultdict
from collections.abc import Iterator
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")
PEAK_FLOPS = 989e12  # H100 SXM bf16 dense


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def two_batch_estimates(
    small_sq: float, big_sq: float, b_small: float, b_big: float
) -> tuple[float, float]:
    """Unbiased (|G|^2, tr(S)) from the mean squared norms of gradients over b_small and b_big rows."""
    g2 = (b_big * big_sq - b_small * small_sq) / (b_big - b_small)
    s = (small_sq - big_sq) / (1 / b_small - 1 / b_big)
    return g2, s


class NoiseScale:
    """Call start() before a step's micro-batches; per micro-batch, the forward on the bare module
    (not DDP), backward(scaled_loss), then micro(task, rows, scale) (scale = the factor its loss
    was multiplied by); end() instead of the DDP all-reduce: it all-reduces (mean) p.grad itself.
    Per task: |G_small| from single micro-batches, |G_big| from all of that task's micro-batches
    on all ranks. Total: |G_small| from each rank's step gradient (its rows), |G_big| all ranks'.
    p.grad accumulates exactly as without the diagnostic; the copies are flat fp32 buffers."""

    def __init__(self, params: list[torch.nn.Parameter], ema_decay: float = 0.9):
        self.params = [p for p in params if p.requires_grad]
        self.decay = ema_decay
        self.ema: dict[str, list[float]] = {}  # key -> [g2, s, weight] (debiased EMA)

    def _flat(self) -> torch.Tensor:
        return torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float()
                for p in self.params
            ]
        )

    def start(self) -> None:
        self.prev: torch.Tensor | None = None
        self.sums: dict[str, torch.Tensor] = {}
        self.sq: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self.rows: dict[str, int] = {}
        self.step_rows = 0

    def backward(self, loss: torch.Tensor) -> None:
        """loss.backward() that DDP does not see: torch.autograd.grad fires no gradient-accumulator
        hooks, so the reducer stays as the last DDP step left it (a backward through the bare
        module, or DDP.no_sync, breaks the next find_unused_parameters step). Call the forward on
        the bare module too."""
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        for p, g in zip(self.params, grads):
            if g is not None:
                p.grad = g if p.grad is None else p.grad.add_(g)

    @torch.no_grad()
    def micro(self, task: str, rows: int, scale: float) -> None:
        cur = self._flat()
        g = (cur - self.prev if self.prev is not None else cur) / scale
        self.prev = cur
        self.sums[task] = self.sums[task] + g if task in self.sums else g
        self.sq[task] += g.square().sum().item()
        self.n[task] += 1
        self.rows[task] = rows
        self.step_rows += rows

    def _update(self, key: str, small_sq: float, big_sq: float, b_small: float, b_big: float):
        g2, s = two_batch_estimates(small_sq, big_sq, b_small, b_big)
        e = self.ema.setdefault(key, [0.0, 0.0, 0.0])
        e[0] = self.decay * e[0] + (1 - self.decay) * g2
        e[1] = self.decay * e[1] + (1 - self.decay) * s
        e[2] = self.decay * e[2] + (1 - self.decay)

    @torch.no_grad()
    def end(self) -> None:
        world = _world()
        tasks = sorted(self.sums)  # same micro schedule on every rank
        local = self._flat()
        means = [self.sums[t] / self.n[t] for t in tasks]
        small = torch.tensor(
            [local.square().sum().item()] + [self.sq[t] / self.n[t] for t in tasks],
            device=local.device,
        )
        if world > 1:
            dist.all_reduce(small)
            small /= world
            for t in [local, *means]:
                dist.all_reduce(t)
                t /= world
            offset = 0
            for p in self.params:  # the all-reduced gradient back into p.grad (DDP's job)
                if p.grad is not None:
                    p.grad.copy_(local[offset : offset + p.numel()].view_as(p))
                offset += p.numel()
        if world > 1:
            self._update(
                "total",
                small[0].item(),
                local.square().sum().item(),
                self.step_rows,
                self.step_rows * world,
            )
        for i, t in enumerate(tasks):
            if world * self.n[t] > 1 and not (world > 1 and len(tasks) == 1 and self.n[t] == 1):
                self._update(
                    t,
                    small[i + 1].item(),
                    means[i].square().sum().item(),
                    self.rows[t],
                    self.rows[t] * self.n[t] * world,
                )
        self.prev, self.sums = None, {}

    def metrics(self) -> dict[str, float]:
        out = {}
        for key, (g2, s, w) in self.ema.items():
            if w > 0:
                out[f"gns_{key}_g2"] = g2 / w
                out[f"gns_{key}_trS"] = s / w
                out[f"gns_{key}_B"] = s / g2 if g2 > 0 else float("nan")
        return out


class StepTimer:
    """Synchronized wall-clock breakdown of a few steps: mark(name) charges the time since the
    previous mark to name (after a cuda synchronize). wrap() a loader to charge its blocking
    next() to "data". Inactive (no syncs) outside [first, first + n) steps."""

    def __init__(self, first: int, n: int = 20):
        self.first, self.n = first, n
        self.on = False
        self.t: dict[str, float] = defaultdict(float)
        self.last = 0.0

    def mark(self, name: str) -> None:
        if self.on:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.t[name] += now - self.last
            self.last = now

    def wrap(self, it: Iterator[T]) -> Iterator[T]:
        while True:
            self.mark("fwd")
            x = next(it)
            self.mark("data")
            yield x

    def step(self, step: int) -> str | None:
        """Call at the top of each step; returns the breakdown line when the window ends."""
        if step == self.first:
            torch.cuda.synchronize()
            self.on, self.last = True, time.perf_counter()
        elif self.on:
            self.mark("other")
        if step == self.first + self.n and self.on:
            self.on = False
            total = sum(self.t.values())
            return (
                " | ".join(
                    f"{k} {1e3 * v / self.n:.0f} ms ({100 * v / total:.0f}%)"
                    for k, v in self.t.items()
                )
                + f" | step {1e3 * total / self.n:.0f} ms"
            )
        return None
