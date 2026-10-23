"""The release-check detectors flag the failures they exist for and nothing else."""

import numpy as np

from training.eval.release_checks import (
    extra_digit,
    has_burst,
    pacing,
    repeats_ending,
    stutters,
    with_ending,
)

TEXT = (
    "You don't just suddenly go from making twenty dollars to making one thousand dollars an hour."
)


def test_end_repeats_are_flagged():
    assert repeats_ending(TEXT, "... one thousand dollars an hours an hour")
    assert repeats_ending(TEXT, "... one thousand dollars an hour an hour")
    assert repeats_ending(
        "The final count was eighty-eight.", "the final count was eighty eighty eighty"
    )


def test_clean_or_truncated_endings_are_not():
    assert not repeats_ending(TEXT, TEXT.lower())
    assert not repeats_ending("The final count was eighty-eight.", "the final count was eighty")
    assert not repeats_ending("a b c d", "a b c d e")


def test_stutters_ignore_repeats_the_text_has():
    assert stutters(
        "It carries two hundred and eighty-eight SCU.", "it carries two hundred eighty eighty a c u"
    )
    assert not stutters("It was very very cold.", "it was very very cold")
    assert not stutters(TEXT, TEXT)


def test_extra_digit_only_flags_a_digit_read_too_often():
    assert extra_digit("It carries 288 SCU.", "it carries two eight eight eight c u")
    assert not extra_digit("It carries 288 SCU.", "it carries two eight eight c u")
    assert not extra_digit("It carries 288 SCU.", "it carries two hundred eighty eight c u")
    assert not extra_digit("Call 911 now.", "call nine one one now")
    assert extra_digit("Room 316.", "room three one one six")


def test_endings():
    assert with_ending("Call me back.", "") == "Call me back"
    assert with_ending("Call me back.", "...") == "Call me back..."
    assert with_ending("Call me back.", ".") == "Call me back."


def test_burst_needs_a_loud_start_then_silence():
    sr = 24000
    wav = np.zeros(sr, dtype=np.float32)
    assert not has_burst(wav, sr)
    wav[100:600] = 0.1
    assert has_burst(wav, sr)
    wav[2000:4000] = 0.1  # speech right after: an onset, not a burst
    assert not has_burst(wav, sr)


def test_pacing_finds_inner_pauses_only():
    sr = 24000
    tone = np.sin(np.arange(sr // 2) * 0.1).astype(np.float32)
    gap = np.zeros(sr // 4, dtype=np.float32)
    span, pauses = pacing(np.concatenate([gap, tone, gap, tone, gap]), sr)
    assert len(pauses) == 1 and abs(pauses[0] - 0.25) < 0.02
    assert abs(span - 1.25) < 0.02
