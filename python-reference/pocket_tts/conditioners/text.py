import logging

import sentencepiece
import torch
from torch import nn

from pocket_tts.conditioners.base import BaseConditioner, TokenizedText
from pocket_tts.utils.utils import download_if_necessary

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77, PAD, PAD, PAD, PAD, PAD, PAD]]

    Args:
        n_bins (int): should be equal to the number of elements in the sentencepiece tokenizer.
        tokenizer_path (str): path to the sentencepiece tokenizer model.

    """

    def __init__(self, nbins: int, tokenizer_path: str) -> None:
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        assert nbins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()} but nbins={nbins} was specified"
        )

    def __call__(self, text: str) -> TokenizedText:
        return TokenizedText(torch.tensor(self.sp.encode(text, out_type=int))[None, :])


class LUTConditioner(BaseConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        possible_values (list[str] or None): list of possible values for the tokenizer.
    """

    def __init__(self, n_bins: int, tokenizer_path: str, dim: int, output_dim: int):
        super().__init__(dim=dim, output_dim=output_dim)
        self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
        self.embed = nn.Embedding(n_bins + 1, self.dim)  # n_bins + 1 for padding.

    def prepare(self, x: str) -> TokenizedText:
        tokens = self.tokenizer(x)
        tokens = tokens[0].to(self.embed.weight.device)
        return TokenizedText(tokens)

    def _get_condition(self, inputs: TokenizedText) -> torch.Tensor:
        embeds = self.embed(inputs[0])
        return embeds
