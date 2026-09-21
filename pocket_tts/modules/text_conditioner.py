import logging

import sentencepiece
import tokenizers
import torch
from torch import nn

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

    def __init__(self, nbins: int, tokenizer_path: str):
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        local_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(local_path))
        assert nbins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()} but nbins={nbins} was specified"
        )

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text))[None, :]


class JsonTokenizer:
    """A `tokenizer.json` (HuggingFace tokenizers) holding the same vocabulary.

    Converted from the SentencePiece model, so it produces identical token ids;
    it loads without the sentencepiece runtime and is the format the released
    configs point at. `.model` files stay supported for models trained elsewhere.
    """

    def __init__(self, nbins: int, tokenizer_path: str):
        logger.info("Loading tokenizers json from %s", tokenizer_path)
        local_path = download_if_necessary(tokenizer_path)
        self.tokenizer = tokenizers.Tokenizer.from_file(str(local_path))
        vocab_size = self.tokenizer.get_vocab_size()
        assert nbins == vocab_size, (
            f"tokenizer has vocab size={vocab_size} but nbins={nbins} was specified"
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text))[None, :]


Tokenizer = SentencePieceTokenizer | JsonTokenizer


def build_tokenizer(nbins: int, tokenizer_path: str, kind: str = "sentencepiece") -> Tokenizer:
    """Pick the backend from the config's `tokenizer` field, else from the suffix."""
    if kind in ("tokenizers", "json") or str(tokenizer_path).split("@")[0].endswith(".json"):
        return JsonTokenizer(nbins, tokenizer_path)
    return SentencePieceTokenizer(nbins, tokenizer_path)


DEFAULT_TOKENIZER_N_BINS = 4000
DEFAULT_TOKENIZER_PATH = (
    "hf://kyutai/pocket-tts-without-voice-cloning/"
    "tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214"
)


def get_default_tokenizer() -> Tokenizer:
    """Return a SentencePieceTokenizer with the default model path and vocab size.

    Downloads the tokenizer model from HuggingFace on first use.
    """
    return build_tokenizer(DEFAULT_TOKENIZER_N_BINS, DEFAULT_TOKENIZER_PATH)


class LUTConditioner(nn.Module):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        possible_values (list[str] or None): list of possible values for the tokenizer.
    """

    def __init__(
        self,
        n_bins: int,
        tokenizer_path: str,
        dim: int,
        output_dim: int,
        tokenizer: str = "sentencepiece",
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.tokenizer = build_tokenizer(n_bins, tokenizer_path, tokenizer)
        self.embed = nn.Embedding(n_bins + 1, self.dim)  # n_bins + 1 for padding.

    def prepare(self, x: str) -> torch.Tensor:
        return self.tokenizer(x).to(self.embed.weight.device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)
