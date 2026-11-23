"""English numbers as words: "$1,250" -> "one thousand two hundred fifty dollars"."""

import re

_ONES_TEXT = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
)
_ONES = _ONES_TEXT.split()
_TENS_TEXT = "_ _ twenty thirty forty fifty sixty seventy eighty ninety"
_TENS = _TENS_TEXT.split()
_IRREGULAR_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

_TIME = re.compile(r"(?<![\w.,:])(\d{1,2}):(\d\d)(?![\w:]|[.,]\d)")
# A whole number token ("1,250", "822.95", "3,5"), with an optional currency sign, scale word or
# suffix. Commas and periods are part of the token, so a decimal is never read in two halves.
_NUMBER = re.compile(
    r"(?<![\w.,:])(\$ ?)?(\d[\d,.]*\d|\d)"
    r"( (?:thousand|million|billion|trillion)(?!\w))?(%|st|nd|rd|th)?(?!\w|:\d)"
)


def cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {cardinal(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    return f"{cardinal(thousands)} thousand" + (f" {cardinal(rest)}" if rest else "")


def ordinal(n: int) -> str:
    def last_word(m: re.Match[str]) -> str:
        word = m.group(0)
        if word in _IRREGULAR_ORDINALS:
            return _IRREGULAR_ORDINALS[word]
        return word[:-1] + "ieth" if word.endswith("y") else word + "th"

    return re.sub(r"[a-z]+$", last_word, cardinal(n))


def _time(m: re.Match[str]) -> str:
    hours, minutes = int(m.group(1)), int(m.group(2))
    if hours > 23 or minutes > 59:
        return m.group(0)
    if minutes == 0:
        return cardinal(hours) + (" o'clock" if hours <= 12 else " hundred")
    return f"{cardinal(hours)} {'oh ' if minutes < 10 else ''}{cardinal(minutes)}"


def _number(m: re.Match[str]) -> str:
    currency, digits, scale, suffix = m.groups()
    if "." in digits or ("," in digits and not re.fullmatch(r"\d{1,3}(,\d{3})+", digits)):
        return m.group(0)
    n = int(digits.replace(",", ""))
    if n >= 1_000_000:
        return m.group(0)
    if suffix in ("st", "nd", "rd", "th"):
        return ordinal(n) + (scale or "")
    words = cardinal(n) + (scale or "")
    if currency:
        words += " dollar" if n == 1 and not scale else " dollars"
    return words + (" percent" if suffix == "%" else "")


def normalize_numbers(text: str) -> str:
    return _NUMBER.sub(_number, _TIME.sub(_time, text))
