import torch
import torch.nn as nn

def _rms_norm(x: torch.Tensor, alpha: torch.Tensor, eps: float):
    assert x.dim() >= alpha.dim()
    x_dtype = x.dtype
    # PyTorch var uses n-1 denominator by default if unbiased=True
    # In pocket_tts it seems they use the default var which is unbiased=True (correction=1)
    var = eps + x.var(dim=-1, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        alpha_shape = (dim,)
        self.alpha = nn.Parameter(torch.full(alpha_shape, 1.0, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.eps)

# Test data
dim = 4
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
norm = RMSNorm(dim)
y = norm(x)

print(f"Input: {x}")
print(f"Output: {y}")

# Manual calc with var()
# mean = (1+2+3+4)/4 = 2.5
# var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 3  (Bessel's correction)
#     = (2.25 + 0.25 + 0.25 + 2.25) / 3 = 5 / 3 = 1.666...
# std = sqrt(1.666... + 1e-5) = 1.2909...
# y = [1/std, 2/std, 3/std, 4/std]
# y = [0.7746, 1.5492, 2.3238, 3.0984]

print(f"Manual check: {x / torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-5)}")

# Standard RMSNorm check (mean(x^2))
# r2 = (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1+4+9+16)/4 = 30/4 = 7.5
# std_rms = sqrt(7.5 + 1e-5) = 2.7386...
# y_rms = [1/std_rms, 2/std_rms, 3/std_rms, 4/std_rms]
# y_rms = [0.3651, 0.7303, 1.0954, 1.4606]
print(f"Standard RMSNorm: {x / torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-5)}")
