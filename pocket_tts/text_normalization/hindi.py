"""Hindi numbers as words: "₹1,25,000" -> "एक लाख पच्चीस हज़ार रुपये"."""

import re

# 0 to 99: Hindi number names are irregular below a hundred.
_BELOW_100_TEXT = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस "
    "पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस "
    "सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ "
    "इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सरसठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर "
    "पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्नासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी "
    "सत्तासी अठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे "
    "निन्यानवे"
)
_BELOW_100 = _BELOW_100_TEXT.split()
_SCALES = ((10_000_000, "करोड़"), (100_000, "लाख"), (1000, "हज़ार"), (100, "सौ"))
_IRREGULAR_ORDINALS = {1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 6: "छठा"}

_TIME = re.compile(r"(?<![\w.,:])(\d{1,2}):(\d\d)(?![\w:]|[.,]\d)(?:\s*बजे)?")
# A whole number token, with an optional currency sign, scale word or suffix. Commas (Indian
# "1,25,000" or Western "125,000" grouping) and periods are part of the token, so a decimal is
# never read in two halves.
_NUMBER = re.compile(
    r"(?<![\w.,:])(₹ ?|\$ ?)?(\d[\d,.]*\d|\d)( (?:हज़ार|हजार|लाख|करोड़)(?!\w))?"
    r"(%|st|nd|rd|th)?(?!\w|:\d)"
)
_CURRENCIES = {"₹": "रुपये", "$": "डॉलर"}


def cardinal(n: int) -> str:
    if n < 100:
        return _BELOW_100[n]
    for value, name in _SCALES:
        if n >= value:
            count, rest = divmod(n, value)
            return f"{cardinal(count)} {name}" + (f" {cardinal(rest)}" if rest else "")
    raise AssertionError(n)


def ordinal(n: int) -> str:
    return _IRREGULAR_ORDINALS.get(n) or cardinal(n) + "वाँ"


def _time(m: re.Match[str]) -> str:
    hours, minutes = int(m.group(1)), int(m.group(2))
    if hours > 23 or minutes > 59:
        return m.group(0)
    hour = hours % 12 or 12
    if minutes == 0:
        return f"{cardinal(hour)} बजे"
    if minutes == 15:
        return f"सवा {cardinal(hour)} बजे"
    if minutes == 30:
        return {1: "डेढ़ बजे", 2: "ढाई बजे"}.get(hour, f"साढ़े {cardinal(hour)} बजे")
    if minutes == 45:
        return f"पौने {cardinal(hour % 12 + 1)} बजे"
    return f"{cardinal(hour)} बजकर {cardinal(minutes)} मिनट"


def _number(m: re.Match[str]) -> str:
    currency, digits, scale, suffix = m.groups()
    if "." in digits or ("," in digits and not re.fullmatch(r"\d{1,3}(,\d\d)*,\d{3}", digits)):
        return m.group(0)
    n = int(digits.replace(",", ""))
    if suffix in ("st", "nd", "rd", "th"):
        return ordinal(n) + (scale or "")
    words = cardinal(n) + (scale or "")
    if currency:
        words += " " + _CURRENCIES[currency.strip()]
    return words + (" प्रतिशत" if suffix == "%" else "")


def normalize_numbers(text: str) -> str:
    return _NUMBER.sub(_number, _TIME.sub(_time, text))
