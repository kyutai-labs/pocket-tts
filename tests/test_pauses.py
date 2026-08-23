import pytest

from pocket_tts.default_parameters import DEFAULT_PAUSE_SECONDS, MAX_PAUSE_SECONDS
from pocket_tts.models.tts_model import SSML_BREAK_STRENGTHS, split_on_pauses

# --------------------------------------------------------------- no markers


def test_text_without_markers_is_returned_unchanged():
    assert split_on_pauses("Hello world.") == ["Hello world."]


def test_empty_text_returns_nothing_to_say():
    assert split_on_pauses("") == []
    assert split_on_pauses("   ") == []


# --------------------------------------------------------- [pause] shorthand


def test_bare_marker_uses_the_default_duration():
    assert split_on_pauses("Ready. [pause] Go.") == ["Ready. ", DEFAULT_PAUSE_SECONDS, " Go."]


def test_explicit_duration_is_used():
    assert split_on_pauses("Ready. [pause:1.5] Go.") == ["Ready. ", 1.5, " Go."]


def test_shorthand_is_case_insensitive_and_tolerates_spaces():
    assert split_on_pauses("A [PAUSE: 2] B") == ["A ", 2.0, " B"]
    assert split_on_pauses("A [ pause ] B") == ["A ", DEFAULT_PAUSE_SECONDS, " B"]


def test_decimal_without_leading_zero():
    assert split_on_pauses("A [pause:.25] B") == ["A ", 0.25, " B"]


def test_malformed_shorthand_is_left_as_text():
    # Must stay in the spoken text rather than silently disappearing.
    assert split_on_pauses("A [pause:] B") == ["A [pause:] B"]
    assert split_on_pauses("A [paws:1] B") == ["A [paws:1] B"]
    assert split_on_pauses("A [pause 1] B") == ["A [pause 1] B"]


# ------------------------------------------------------------- SSML <break>


def test_break_with_milliseconds():
    assert split_on_pauses('Ready. <break time="500ms"/> Go.') == ["Ready. ", 0.5, " Go."]


def test_break_with_seconds():
    assert split_on_pauses('Ready. <break time="1.5s"/> Go.') == ["Ready. ", 1.5, " Go."]


def test_bare_break_uses_the_default_duration():
    assert split_on_pauses("A <break/> B") == ["A ", DEFAULT_PAUSE_SECONDS, " B"]
    assert split_on_pauses("A <break /> B") == ["A ", DEFAULT_PAUSE_SECONDS, " B"]
    assert split_on_pauses("A <break> B") == ["A ", DEFAULT_PAUSE_SECONDS, " B"]


def test_break_open_close_pair_is_one_pause():
    assert split_on_pauses("A <break></break> B") == ["A ", DEFAULT_PAUSE_SECONDS, " B"]


@pytest.mark.parametrize("name,seconds", sorted(SSML_BREAK_STRENGTHS.items()))
def test_break_strength_values(name, seconds):
    assert split_on_pauses(f'A <break strength="{name}"/> B') == ["A ", seconds, " B"]


def test_break_single_quotes_and_loose_spacing():
    assert split_on_pauses("A <break  time = '250ms' /> B") == ["A ", 0.25, " B"]


def test_break_is_case_insensitive():
    assert split_on_pauses('A <BREAK TIME="1s"/> B') == ["A ", 1.0, " B"]


def test_time_wins_over_strength_when_both_are_given():
    assert split_on_pauses('A <break time="2s" strength="weak"/> B') == ["A ", 2.0, " B"]


def test_unknown_strength_is_an_error():
    with pytest.raises(ValueError, match="Unknown <break strength"):
        split_on_pauses('A <break strength="enormous"/> B')


# ------------------------------------------------------------ SSML document


def test_speak_wrapper_is_stripped():
    ssml = '<speak>Ready. <break time="1s"/> Go.</speak>'
    assert split_on_pauses(ssml) == ["Ready. ", 1.0, " Go."]


def test_speak_wrapper_with_attributes_is_stripped():
    ssml = '<speak version="1.1" xml:lang="en-US">Hi <break/> there</speak>'
    assert split_on_pauses(ssml) == ["Hi ", DEFAULT_PAUSE_SECONDS, " there"]


def test_other_ssml_tags_raise_rather_than_being_ignored():
    # Silently dropping these would produce quietly wrong audio.
    with pytest.raises(ValueError, match="<prosody>"):
        split_on_pauses('<speak><prosody rate="slow">Slow</prosody></speak>')

    with pytest.raises(ValueError, match="Unsupported SSML tag"):
        split_on_pauses("Hello <emphasis>world</emphasis>")


def test_error_lists_every_unsupported_tag():
    with pytest.raises(ValueError) as excinfo:
        split_on_pauses("<emphasis>a</emphasis> <prosody>b</prosody>")
    message = str(excinfo.value)
    assert "<emphasis>" in message and "<prosody>" in message


def test_comparison_operators_are_not_mistaken_for_tags():
    assert split_on_pauses("5 < 6 and 7 > 3.") == ["5 < 6 and 7 > 3."]


# ------------------------------------------------------------------- mixing


def test_both_syntaxes_can_be_mixed():
    assert split_on_pauses('One [pause:1] two <break time="2s"/> three') == [
        "One ",
        1.0,
        " two ",
        2.0,
        " three",
    ]


def test_several_markers_in_one_string():
    assert split_on_pauses("One [pause:1] two [pause:2] three") == [
        "One ",
        1.0,
        " two ",
        2.0,
        " three",
    ]


def test_marker_at_the_start_and_end():
    assert split_on_pauses("[pause:1] hi [pause:2]") == [1.0, " hi ", 2.0]


def test_adjacent_markers_do_not_produce_empty_text():
    parts = split_on_pauses('Hi [pause:1]<break time="2s"/> there')
    assert parts == ["Hi ", 1.0, 2.0, " there"]
    assert all(not isinstance(p, str) or p.strip() for p in parts)


def test_only_a_marker_yields_only_silence():
    assert split_on_pauses("[pause:1]") == [1.0]
    assert split_on_pauses('<break time="1s"/>') == [1.0]


# ------------------------------------------------------------------ clamping


def test_duration_is_clamped_to_the_maximum():
    assert split_on_pauses("A [pause:9999] B") == ["A ", MAX_PAUSE_SECONDS, " B"]
    assert split_on_pauses('A <break time="9999s"/> B') == ["A ", MAX_PAUSE_SECONDS, " B"]


def test_zero_duration_is_allowed():
    assert split_on_pauses("A [pause:0] B") == ["A ", 0.0, " B"]
    assert split_on_pauses('A <break time="0ms"/> B') == ["A ", 0.0, " B"]


def test_limits_are_configurable_per_call():
    assert split_on_pauses("A [pause] B", default_seconds=3.0) == ["A ", 3.0, " B"]
    assert split_on_pauses("A [pause:99] B", max_seconds=20.0) == ["A ", 20.0, " B"]
