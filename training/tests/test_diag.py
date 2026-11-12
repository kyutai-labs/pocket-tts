import torch

from training.diag import NoiseScale


def test_noise_scale_recovers_known_b_simple():
    """Per-row gradients G + noise with tr(S) known: B_simple = tr(S) / |G|^2."""
    torch.manual_seed(0)
    d, rows, micro = 64, 4, 3
    G = torch.randn(d)
    sigma = 2.0  # per-row noise std per coordinate: tr(S) = d sigma^2
    p = torch.nn.Parameter(torch.zeros(d))
    ns = NoiseScale([p], ema_decay=0.999)
    for _ in range(3000):
        p.grad = None
        ns.start()
        for _ in range(micro):
            g = G + sigma * torch.randn(rows, d).mean(0)  # mean over the micro-batch's rows
            p.grad = g / micro if p.grad is None else p.grad + g / micro
            ns.micro("t", rows, 1 / micro)
        ns.end()
    m = ns.metrics()
    want = d * sigma**2 / G.square().sum().item()
    assert abs(m["gns_t_B"] / want - 1) < 0.1, (m["gns_t_B"], want)
    assert abs(m["gns_t_trS"] / (d * sigma**2) - 1) < 0.05


class _TwoHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk, self.a, self.b = (
            torch.nn.Linear(16, 16),
            torch.nn.Linear(16, 4),
            torch.nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        return (self.a if task == "a" else self.b)(self.trunk(x)).square().mean()


def _ddp_worker(rank: int, world: int, port: int, out: dict[str, object]):
    import os

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    module = _TwoHeads()
    ddp = DDP(module, find_unused_parameters=True)  # each micro-batch leaves a head unused
    ns = NoiseScale(list(module.parameters()))
    grads = []
    # Plain DDP accumulation (as training/train.py), then a diagnostic step on the same data (same
    # gradient), then plain again (DDP's reducer must not be disturbed by the diagnostic step).
    for diag in (False, True, False):
        torch.manual_seed(1 + rank)
        module.zero_grad()
        if diag:
            ns.start()
        for micro, task in enumerate("abb"):
            loss = (module if diag else ddp)(torch.randn(8, 16), task) / 3
            if diag:
                ns.backward(loss)
                ns.micro(task, 8, 1 / 3)
            elif micro < 2:
                with ddp.no_sync():
                    loss.backward()
            else:
                loss.backward()
        if diag:
            ns.end()
        grads.append(
            torch.cat([p.grad.flatten() for p in module.parameters() if p.grad is not None])
        )
    if rank == 0:
        out["diff"] = max((grads[0] - g).abs().max().item() for g in grads[1:])
        out["metrics"] = ns.metrics()
    dist.destroy_process_group()


def test_noise_scale_under_ddp_matches_plain_allreduce():
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_ddp_worker, args=(2, port, out), nprocs=2)
    assert out["diff"] < 1e-6, out["diff"]
    assert {"gns_total_B", "gns_b_B"} <= set(out["metrics"]), out["metrics"]
