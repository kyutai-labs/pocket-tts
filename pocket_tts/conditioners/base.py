from typing import NamedTuple

import torch


class TokenizedText(NamedTuple):
    tokens: torch.Tensor  # should be long tensor.
