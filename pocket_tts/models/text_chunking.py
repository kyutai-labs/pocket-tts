"""Turning a text prompt into the chunks the model generates one at a time.

The model is trained on single sentences, so long inputs are split on sentence
boundaries and regrouped into chunks that fit `max_tokens`.
"""

import logging
import re

from pocket_tts.modules.text_conditioner import Tokenizer

logger = logging.getLogger(__name__)

_HINDI_NUMBERS_0_99 = {
    0: "शून्य",
    1: "एक",
    2: "दो",
    3: "तीन",
    4: "चार",
    5: "पाँच",
    6: "छह",
    7: "सात",
    8: "आठ",
    9: "नौ",
    10: "दस",
    11: "ग्यारह",
    12: "बारह",
    13: "तेरह",
    14: "चौदह",
    15: "पंद्रह",
    16: "सोलह",
    17: "सत्रह",
    18: "अठारह",
    19: "उन्नीस",
    20: "बीस",
    21: "इक्कीस",
    22: "बाईस",
    23: "तेईस",
    24: "चौबीस",
    25: "पच्चीस",
    26: "छब्बीस",
    27: "सत्ताईस",
    28: "अट्ठाईस",
    29: "उनतीस",
    30: "तीस",
    31: "इकतीस",
    32: "बत्तीस",
    33: "तैंतीस",
    34: "चौंतीस",
    35: "पैंतीस",
    36: "छत्तीस",
    37: "सैंतीस",
    38: "अड़तीस",
    39: "उनतालीस",
    40: "चालीस",
    41: "इकतालीस",
    42: "बयालीस",
    43: "तैंतालीस",
    44: "चौवालीस",
    45: "पैंतालीस",
    46: "छियालीस",
    47: "सैंतालीस",
    48: "अड़तालीस",
    49: "उनचास",
    50: "पचास",
    51: "इक्यावन",
    52: "बावन",
    53: "तिरेपन",
    54: "चौवन",
    55: "पचपन",
    56: "छप्पन",
    57: "सत्तावन",
    58: "अट्ठावन",
    59: "उनसठ",
    60: "साठ",
    61: "इकसठ",
    62: "बासठ",
    63: "तिरेसठ",
    64: "चौंसठ",
    65: "पैंसठ",
    66: "छियासठ",
    67: "सरसठ",
    68: "अड़सठ",
    69: "उनहत्तर",
    70: "सत्तर",
    71: "इकहत्तर",
    72: "बहत्तर",
    73: "तिहत्तर",
    74: "चौहत्तर",
    75: "पचहत्तर",
    76: "छिहत्तर",
    77: "सतहत्तर",
    78: "अठहत्तर",
    79: "उन्नासी",
    80: "अस्सी",
    81: "इक्यासी",
    82: "बयासी",
    83: "तिरासी",
    84: "चौरासी",
    85: "पचासी",
    86: "छियासी",
    87: "सत्तासी",
    88: "अठासी",
    89: "नवासी",
    90: "नब्बे",
    91: "इक्यानवे",
    92: "बानवे",
    93: "तिरानवे",
    94: "चौरानवे",
    95: "पंचानवे",
    96: "छियानवे",
    97: "सत्तानवे",
    98: "अट्ठानवे",
    99: "निन्यानवे",
    100: "सौ",
}

_ENGLISH_NUMBERS_0_19 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_ENGLISH_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]

_DEVANAGARI_DIGITS_TRANS = str.maketrans("०१२३४५६७८९", "0123456789")


def _number_to_hindi(num: int) -> str:
    if num in _HINDI_NUMBERS_0_99:
        return _HINDI_NUMBERS_0_99[num]
    if num < 1000:
        h, rem = divmod(num, 100)
        res = f"{_HINDI_NUMBERS_0_99.get(h, str(h))} सौ"
        return f"{res} {_number_to_hindi(rem)}" if rem else res
    if num < 100000:
        th, rem = divmod(num, 1000)
        res = f"{_number_to_hindi(th)} हज़ार"
        return f"{res} {_number_to_hindi(rem)}" if rem else res
    if num < 10000000:
        lakh, rem = divmod(num, 100000)
        res = f"{_number_to_hindi(lakh)} लाख"
        return f"{res} {_number_to_hindi(rem)}" if rem else res
    cr, rem = divmod(num, 10000000)
    res = f"{_number_to_hindi(cr)} करोड़"
    return f"{res} {_number_to_hindi(rem)}" if rem else res


def _number_to_english(num: int) -> str:
    if 0 <= num < 20:
        return _ENGLISH_NUMBERS_0_19[num]
    if 20 <= num < 100:
        tens, rem = divmod(num, 10)
        return f"{_ENGLISH_TENS[tens]}-{_ENGLISH_NUMBERS_0_19[rem]}" if rem else _ENGLISH_TENS[tens]
    if 100 <= num < 1000:
        h, rem = divmod(num, 100)
        res = f"{_ENGLISH_NUMBERS_0_19[h]} hundred"
        return f"{res} {_number_to_english(rem)}" if rem else res
    if 1000 <= num < 1000000:
        th, rem = divmod(num, 1000)
        res = f"{_number_to_english(th)} thousand"
        return f"{res} {_number_to_english(rem)}" if rem else res
    return str(num)


_HINDI_ORDINALS = {
    1: "पहला",
    2: "दूसरा",
    3: "तीसरा",
    4: "चौथा",
    5: "पाँचवाँ",
    6: "छठा",
    7: "सातवाँ",
    8: "आठवाँ",
    9: "नौवाँ",
    10: "दसवाँ",
}

_ENG_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}

_HINDI_COMMON_ABBR = {
    "AI": "एआई",
    "ML": "एमएल",
    "CEO": "सीईओ",
    "CTO": "सीटीओ",
    "HR": "एचआर",
    "Dr.": "डॉक्टर",
    "Mr.": "मिस्टर",
    "Mrs.": "मिसेज",
    "PM": "पीएम",
    "AM": "एएम",
    "OTP": "ओटीपी",
    "KYC": "केवाईसी",
    "OK": "ओके",
    "vs": "बनाम",
    "vs.": "बनाम",
}


def normalize_text(text: str, mode: str = "auto") -> tuple[str, list[dict[str, str]]]:
    """Normalize numeric digits, ordinals, currencies, and abbreviations into spoken words."""
    changes: list[dict[str, str]] = []
    text = text.translate(_DEVANAGARI_DIGITS_TRANS)

    is_hi = (mode in ("hindi", "hinglish")) or (
        mode == "auto" and bool(re.search(r"[\u0900-\u097F]", text))
    )

    if is_hi:
        # Currency: ₹ and $
        def _repl_inr(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = f"{_number_to_hindi(val)} रुपये"
            changes.append({"from": m.group(0), "to": rep, "type": "currency"})
            return rep

        text = re.sub(r"₹\s*(\d+)", _repl_inr, text)

        def _repl_usd(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = f"{_number_to_hindi(val)} डॉलर"
            changes.append({"from": m.group(0), "to": rep, "type": "currency"})
            return rep

        text = re.sub(r"\$\s*(\d+)", _repl_usd, text)

        # Percentage: %
        def _repl_pct(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = f"{_number_to_hindi(val)} प्रतिशत"
            changes.append({"from": m.group(0), "to": rep, "type": "percentage"})
            return rep

        text = re.sub(r"(\d+)\s*%", _repl_pct, text)

        # Ordinals: 1st, 2nd, 3rd, 10th
        def _repl_ord(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = _HINDI_ORDINALS.get(val, f"{_number_to_hindi(val)}वाँ")
            changes.append({"from": m.group(0), "to": rep, "type": "ordinal"})
            return rep

        text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _repl_ord, text, flags=re.IGNORECASE)

        # Time: HH:MM
        def _repl_time(m: re.Match[str]) -> str:
            hh, mm = int(m.group(1)), int(m.group(2))
            if mm == 0:
                rep = f"{_number_to_hindi(hh)} बजे"
            elif mm == 30:
                rep = f"साढ़े {_number_to_hindi(hh)} बजे"
            elif mm == 15:
                rep = f"सवा {_number_to_hindi(hh)} बजे"
            elif mm == 45:
                rep = f"पौने {_number_to_hindi((hh % 12) + 1)} बजे"
            else:
                rep = f"{_number_to_hindi(hh)} बजकर {_number_to_hindi(mm)} मिनट"
            changes.append({"from": m.group(0).strip(), "to": rep, "type": "time"})
            return rep

        text = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*बजे)?", _repl_time, text)

        # Common abbreviations
        for abbr, word in _HINDI_COMMON_ABBR.items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            if re.search(pattern, text):
                text = re.sub(pattern, word, text)
                changes.append({"from": abbr, "to": word, "type": "abbreviation"})

        # Standalone numbers
        def _repl_num(m: re.Match[str]) -> str:
            val = int(m.group(0))
            rep = _number_to_hindi(val)
            changes.append({"from": m.group(0), "to": rep, "type": "number"})
            return rep

        text = re.sub(r"\b\d+\b", _repl_num, text)

    else:
        # English normalization
        def _repl_usd_en(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = f"{_number_to_english(val)} dollars"
            changes.append({"from": m.group(0), "to": rep, "type": "currency"})
            return rep

        text = re.sub(r"\$\s*(\d+)", _repl_usd_en, text)

        def _repl_pct_en(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = f"{_number_to_english(val)} percent"
            changes.append({"from": m.group(0), "to": rep, "type": "percentage"})
            return rep

        text = re.sub(r"(\d+)\s*%", _repl_pct_en, text)

        def _repl_ord_en(m: re.Match[str]) -> str:
            val = int(m.group(1))
            rep = _ENG_ORDINALS.get(val, f"{_number_to_english(val)}th")
            changes.append({"from": m.group(0), "to": rep, "type": "ordinal"})
            return rep

        text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _repl_ord_en, text, flags=re.IGNORECASE)

        def _repl_num_en(m: re.Match[str]) -> str:
            val = int(m.group(0))
            rep = _number_to_english(val)
            changes.append({"from": m.group(0), "to": rep, "type": "number"})
            return rep

        text = re.sub(r"\b\d+\b", _repl_num_en, text)

    return text, changes


def normalize_numbers(text: str, mode: str = "auto") -> str:
    """Normalize numeric digits into spoken words in Hindi or English."""
    normalized, _ = normalize_text(text, mode=mode)
    return normalized


def prepare_text_prompt(
    text: str,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
    append_terminal_punctuation: bool = True,
    capitalize_first_letter: bool = True,
    replace_characters: dict[str, str] | None = None,
    normalize_numbers_to_words: bool = False,
) -> tuple[str, int]:
    text = text.strip()
    if replace_characters:
        text = " ".join(text.translate(str.maketrans(replace_characters)).split())
        # Deleted quotes leave '"Hi?", she said' as 'Hi?, she said', which reads as a sentence end
        # followed by a stray comma; keep the sentence mark only.
        text = re.sub(r"([.!?\u2026])\s*[,;:]", r"\1", text)
    if normalize_numbers_to_words:
        text = normalize_numbers(text)
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    if not normalize_numbers_to_words and re.search(r"\d", text):
        logger.warning(
            "Digits found in input text. Digits are not normalized: "
            "consider writing numbers as words (e.g. 'तीन बजे', 'three thirty')."
        )
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    # Make sure it starts with an uppercase letter. Only meaningful for
    # orthographies that have case; see Config.capitalize_first_letter.
    if capitalize_first_letter and not text[0].isupper():
        text = text[0].upper() + text[1:]

    if append_terminal_punctuation:
        text = _ensure_terminal_punctuation(text)

    # The model does not perform well when there are very few tokens, so
    # we can add empty spaces at the beginning to increase the token count.
    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


_TERMINAL_PUNCTUATION = ".!?\u2026\u0964\u0965"
# A segment ending on a decimal number, as in "... a fee of $822.95," or "... is 3.14."
_ENDS_WITH_DECIMAL_NUMBER = re.compile(r"\d\.\d+\W*$")
_WEAK_PUNCTUATION = ",;:-\u2013\u2014"
_CLOSERS = "\"'\u201d\u2019)]\u00bb"
_TERMINAL_PUNCTUATION_CHARS = set(".!?\u2026\u0964\u0965")
_WEAK_PUNCTUATION_CHARS = set(",;:-\u2013\u2014")


def get_punctuation_tokens(tokenizer: Tokenizer) -> tuple[list[int], list[int]]:
    """Scan the tokenizer vocabulary to find all token IDs that represent terminal and weak punctuation."""
    eos_tokens: set[int] = set()
    fallback_tokens: set[int] = set()

    if hasattr(tokenizer, "get_vocab_size") and hasattr(tokenizer, "id_to_piece"):
        vocab_size = tokenizer.get_vocab_size()
        for idx in range(vocab_size):
            piece = tokenizer.id_to_piece(idx)
            clean = piece.replace("\u2581", "").replace("Ġ", "").strip()
            if clean and all(c in _TERMINAL_PUNCTUATION_CHARS for c in clean):
                eos_tokens.add(idx)
            elif clean and all(c in _WEAK_PUNCTUATION_CHARS for c in clean):
                fallback_tokens.add(idx)

    if not eos_tokens:
        for p in [".", "!", "?", "...", "।", "॥"]:
            eos_tokens.update(tokenizer(p)[0].tolist())
    if not fallback_tokens:
        for p in [",", ";", ":", "-", "–"]:
            fallback_tokens.update(tokenizer(p)[0].tolist())

    return list(eos_tokens), list(fallback_tokens)


def _ensure_terminal_punctuation(text: str) -> str:
    """Make sure the prompt ends with sentence-final punctuation.

    The model is trained on sentences that end with a period, a question mark
    or an exclamation mark. Without one, the last word is often mispronounced
    or repeated. Text that already ends with one, possibly followed by closing
    quotes or brackets, is left alone. A trailing comma, colon or dash is
    replaced by a period. Anything else gets a period appended after the
    closing quote or bracket: the sentence splitter only recognizes a bare
    period token, and a quote left alone after a period would become a chunk
    of its own.
    """
    core = text.rstrip(_CLOSERS + " ")
    closers = text[len(core) :].strip()
    if not core or core[-1] in _TERMINAL_PUNCTUATION:
        return text
    if core[-1] in _WEAK_PUNCTUATION:
        return core.rstrip(_WEAK_PUNCTUATION + " ") + "." + closers
    return text + "."


def _is_decimal_period_boundary(
    list_of_tokens: list[int], segment_start_idx: int, tokenizer: Tokenizer
) -> bool:
    """Return True when segment_start_idx begins right after a decimal period, as in "3.14".

    Only the tokens next to the boundary are decoded, so checking every sentence boundary
    stays linear in the text length. Decoding the period together with the next token keeps
    the space between them: in "I counted to 3. 14 people were there." the period ends a
    sentence.
    """
    before = tokenizer.decode(list_of_tokens[max(0, segment_start_idx - 2) : segment_start_idx])
    after = tokenizer.decode(list_of_tokens[segment_start_idx - 1 : segment_start_idx + 1])
    return (
        len(before) >= 2
        and before[-1] == "."
        and before[-2].isdigit()
        and len(after) >= 2
        and after[1].isdigit()
    )


def _find_boundary_indices(
    list_of_tokens: list[int],
    boundary_tokens: list[int],
    tokenizer: Tokenizer | None = None,
    skip_decimal_periods: bool = False,
) -> list[int]:
    """Find token indices where text should be split based on boundary tokens.

    Returns a list of boundary positions used to slice segments. Each consecutive
    pair (indices[i], indices[i+1]) defines one segment. The first element is
    always 0 and the last is always len(list_of_tokens).
    """
    boundary_set = set(boundary_tokens)
    indices = [0]
    previous_was_boundary = False
    for idx, token in enumerate(list_of_tokens):
        if token in boundary_set:
            previous_was_boundary = True
        else:
            if previous_was_boundary:
                if (
                    skip_decimal_periods
                    and tokenizer is not None
                    and _is_decimal_period_boundary(list_of_tokens, idx, tokenizer)
                ):
                    previous_was_boundary = False
                    continue
                indices.append(idx)
            previous_was_boundary = False
    indices.append(len(list_of_tokens))
    return indices


def _segments_from_boundaries(
    list_of_tokens: list[int], boundary_indices: list[int], tokenizer: Tokenizer
) -> list[tuple[int, str]]:
    """Decode token segments between boundary indices into (token_count, text) pairs."""
    segments = []
    for i in range(len(boundary_indices) - 1):
        start = boundary_indices[i]
        end = boundary_indices[i + 1]
        text = tokenizer.decode(list_of_tokens[start:end])
        segments.append((end - start, text))
    return segments


def split_into_best_sentences(
    tokenizer: Tokenizer,
    text_to_generate: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
    append_terminal_punctuation: bool = True,
    capitalize_first_letter: bool = True,
    replace_characters: dict[str, str] | None = None,
) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(
        text_to_generate,
        pad_with_spaces_for_short_inputs,
        remove_semicolons,
        append_terminal_punctuation,
        capitalize_first_letter,
        replace_characters,
    )
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens[0].tolist()

    end_of_sentence_tokens, fallback_tokens = get_punctuation_tokens(tokenizer)
    sentence_boundaries = _find_boundary_indices(
        list_of_tokens, end_of_sentence_tokens, tokenizer, skip_decimal_periods=True
    )
    nb_tokens_and_sentences = _segments_from_boundaries(
        list_of_tokens, sentence_boundaries, tokenizer
    )

    # Sub-split oversized sentences on commas, semicolons, and colons to prevent skipped words
    refined_segments = []
    for nb_tokens, text in nb_tokens_and_sentences:
        if nb_tokens <= max_tokens:
            refined_segments.append((nb_tokens, text))
        else:
            sub_tokens = tokenizer(text.strip())[0].tolist()
            sub_boundaries = _find_boundary_indices(sub_tokens, fallback_tokens)
            sub_segments = _segments_from_boundaries(sub_tokens, sub_boundaries, tokenizer)
            if len(sub_segments) > 1:
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append((nb_tokens, text))

    # Never end a chunk on a decimal number: "... $822.95," would become the chunk "... $822.95."
    # and the model takes the decimal point for the sentence end, stopping before the decimals.
    # Such a segment stays with the next one.
    glued_segments: list[tuple[int, str]] = []
    for nb_tokens, text in refined_segments:
        if glued_segments and _ENDS_WITH_DECIMAL_NUMBER.search(glued_segments[-1][1]):
            prev_tokens, prev_text = glued_segments[-1]
            glued_segments[-1] = (prev_tokens + nb_tokens, prev_text.rstrip() + " " + text.strip())
        else:
            glued_segments.append((nb_tokens, text))
    refined_segments = glued_segments

    max_nb_tokens_in_a_chunk = max_tokens
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in refined_segments:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    for chunk in chunks:
        chunk_tokens = tokenizer(chunk.strip())[0].tolist()
        if len(chunk_tokens) > max_tokens:
            logger.warning(
                "Chunk has %d tokens (max %d), generation may skip words: '%.50s...'",
                len(chunk_tokens),
                max_tokens,
                chunk,
            )

    return chunks
