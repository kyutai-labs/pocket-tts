import torch

from training.modules.muon import MuonWithAuxAdam, polar_express


def test_polar_express_is_near_orthogonal():
    g = torch.randn(4, 64, 128)
    sv = torch.linalg.svdvals(polar_express(g, 6).float())
    assert sv.min() > 0.9 and sv.max() < 1.1


def test_split_sizes_orthogonalize_each_block():
    # a fused q/k/v weight with GQA-sized k and v blocks: each block is a polar factor
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(64 + 16 + 16, 64))
    opt = MuonWithAuxAdam(
        [
            {
                "params": [w],
                "use_muon": True,
                "lr": 1.0,
                "weight_decay": 0.0,
                "split_sizes": [64, 16, 16],
            }
        ]
    )
    w.grad = torch.randn_like(w)
    before = w.detach().clone()
    opt.step()
    update = (before - w.detach()).float()  # lr 1: update = scale * polar(block) per block
    for block, scale in zip(torch.split(update, [64, 16, 16], dim=0), (1.0, 1.0, 1.0)):
        sv = torch.linalg.svdvals(block / scale)
        assert sv.min() > 0.8 and sv.max() < 1.2


def test_muon_with_aux_adam_updates_every_group():
    torch.manual_seed(0)
    h = torch.nn.Parameter(torch.randn(32, 16))
    gain = torch.nn.Parameter(torch.ones(16))
    opt = MuonWithAuxAdam(
        [
            {"params": [h], "use_muon": True, "lr": 0.005, "weight_decay": 0.0},
            {"params": [gain], "lr": 1e-3, "weight_decay": 0.0},
        ]
    )
    before = [p.detach().clone() for p in (h, gain)]
    for _ in range(2):
        loss = (h.sum() + gain.sum()) * 0.1
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p, b in zip((h, gain), before):
        assert not torch.allclose(p, b)
    state = opt.state_dict()
    opt2 = MuonWithAuxAdam(
        [
            {"params": [h], "use_muon": True, "lr": 0.005, "weight_decay": 0.0},
            {"params": [gain], "lr": 1e-3, "weight_decay": 0.0},
        ]
    )
    opt2.load_state_dict(state)
    assert isinstance(opt2.state[gain]["step"], int)
