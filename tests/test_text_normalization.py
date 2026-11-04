import pytest

from pocket_tts.text_normalization import normalize_numbers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10", "ten"),
        (
            "I have 3 cats, 12 dogs and 105 fish.",
            "I have three cats, twelve dogs and one hundred five fish.",
        ),
        ("$1", "one dollar"),
        ("$50", "fifty dollars"),
        ("$5 million", "five million dollars"),
        ("1,250 people", "one thousand two hundred fifty people"),
        ("$999,999", "nine hundred ninety-nine thousand nine hundred ninety-nine dollars"),
        ("50%", "fifty percent"),
        (
            "1st, 2nd, 3rd, 12th, 20th, 21st, 100th",
            "first, second, third, twelfth, twentieth, twenty-first, one hundredth",
        ),
        (
            "at 3:30, 7:05, 9:00 or 15:00",
            "at three thirty, seven oh five, nine o'clock or fifteen hundred",
        ),
        # Left as they are: decimals, decimal commas, malformed groups, >= 1M, invalid times.
        ("$822.95", "$822.95"),
        ("3.14 and 12.5%", "3.14 and 12.5%"),
        ("822,95 €", "822,95 €"),
        ("1,25,000", "1,25,000"),
        ("1000000 and 1,000,000", "1000000 and 1,000,000"),
        ("25:99", "25:99"),
        ("COVID19 and 5kg", "COVID19 and 5kg"),
    ],
)
def test_english(text: str, expected: str):
    assert normalize_numbers(text, "english") == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10", "दस"),
        ("१०", "दस"),
        ("10 लोग और ₹50", "दस लोग और पचास रुपये"),
        ("₹500", "पाँच सौ रुपये"),
        ("₹1,25,000", "एक लाख पच्चीस हज़ार रुपये"),
        ("125,000", "एक लाख पच्चीस हज़ार"),
        ("₹5 लाख", "पाँच लाख रुपये"),
        ("$20", "बीस डॉलर"),
        ("50%", "पचास प्रतिशत"),
        ("1st 5th 11th", "पहला पाँचवाँ ग्यारहवाँ"),
        ("3:30", "साढ़े तीन बजे"),
        ("1:30 बजे", "डेढ़ बजे"),
        ("14:30", "ढाई बजे"),
        ("3:15", "सवा तीन बजे"),
        ("12:45", "पौने एक बजे"),
        ("4:00 बजे", "चार बजे"),
        ("4:20", "चार बजकर बीस मिनट"),
        ("2,50,00,000", "दो करोड़ पचास लाख"),
        ("₹822.95", "₹822.95"),
        ("1,000,000", "1,000,000"),
    ],
)
def test_hindi(text: str, expected: str):
    assert normalize_numbers(text, "hindi") == expected
