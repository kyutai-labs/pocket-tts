"""User-defined pronunciation overrides.

The text conditioner tokenizes graphemes with SentencePiece; there is no phoneme input.
Mispronounced words (names, technical terms, loanwords, homographs) can still be fixed by
respelling them the way the model reads them, e.g. "live" -> "lyve" or, for the German
model, "Meeting" -> "Mieting". This module applies such respellings from a word-to-text
lexicon before the text is chunked and tokenized.
"""

import json
import re
from pathlib import Path


def apply_pronunciations(text: str, pronunciations: dict[str, str] | None) -> str:
    """Replace every whole-word occurrence of a lexicon entry with its respelling.

    Matching ignores case, and entries may span several words or contain punctuation
    ("E-Mail", "New York"). Longer entries win over shorter ones that they contain.
    When the matched text starts with an uppercase letter and the respelling does not,
    the respelling is capitalized, so "Live." and "live" both map through "live".
    """
    if not pronunciations:
        return text
    keys = sorted((key for key in pronunciations if key.strip()), key=len, reverse=True)
    if not keys:
        return text
    lookup = {key.casefold(): pronunciations[key] for key in keys}
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(key) for key in keys) + r")(?!\w)", re.IGNORECASE
    )

    def substitute(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = lookup.get(word.casefold(), word)
        if word[0].isupper() and replacement[:1].islower():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return pattern.sub(substitute, text)


def load_pronunciations(path: str | Path) -> dict[str, str]:
    """Load a lexicon from a JSON file mapping words to respellings."""
    with open(path, encoding="utf-8") as f:
        pronunciations = json.load(f)
    if not isinstance(pronunciations, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in pronunciations.items()
    ):
        raise ValueError(f"{path} must contain a JSON object mapping words to respellings")
    return pronunciations
