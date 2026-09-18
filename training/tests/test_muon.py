import torch

from training.modules.muon import MuonWithAuxAdam, _zeropower_via_newtonschulz5


def test_newton_schulz_orthogonalizes():
    g = torch.randn(4, 64, 128)
    o = _zeropower_via_newtonschulz5(g, 5).float()
    # the quintic iteration lands every singular value near 1 (reference range ~0.7-1.2)
    sv = torch.linalg.svdvals(o)
    assert sv.min() > 0.5 and sv.max() < 1.5


def test_muon_with_aux_adam_updates_every_group():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(48, 16))  # 3 stacked q/k/v blocks
    h = torch.nn.Parameter(torch.randn(32, 16))
    gain = torch.nn.Parameter(torch.ones(16))
    opt = MuonWithAuxAdam(
        [
            {"params": [h], "use_muon": True, "lr": 0.005, "weight_decay": 0.0},
            {"params": [w], "use_muon": True, "lr": 0.005, "weight_decay": 0.0, "split": 3},
            {"params": [gain], "lr": 1e-3, "weight_decay": 0.0},
        ]
    )
    before = [p.detach().clone() for p in (w, h, gain)]
    for _ in range(2):
        loss = (w.sum() + h.sum() + gain.sum()) * 0.1
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p, b in zip((w, h, gain), before):
        assert not torch.allclose(p, b)
    # a Muon step moves a matrix by lr * scale * (a near-orthogonal matrix): bounded, non-zero
    rms = (h.detach() - before[1]).pow(2).mean().sqrt().item()
    assert 1e-4 < rms < 0.05
