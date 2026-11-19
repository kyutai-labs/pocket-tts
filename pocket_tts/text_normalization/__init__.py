"""Optional rewriting of numbers as words before tokenization, one module per language.

Selected by `normalize_numbers` in the model config; off by default. Numbers a normalizer cannot
read reliably (decimals, ambiguous separators, very large values) are left as they are.
"""

from collections.abc import Callable

from pocket_tts.text_normalization import english, hindi

NORMALIZERS: dict[str, Callable[[str], str]] = {
    "english": english.normalize_numbers,
    "hindi": hindi.normalize_numbers,
}


def normalize_numbers(text: str, language: str) -> str:
    return NORMALIZERS[language](text)
