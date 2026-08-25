"""The eval output directory must encode everything that changes the numbers."""

import itertools
from argparse import Namespace

import pytest

from training.eval.librispeech import DEFAULT_ASR, eval_dir_name


def make_args(**over):
    base = dict(
        temp=0.3, cfg=2.0, use_ema=True, num_items=None, seed=0, asr=DEFAULT_ASR, prompt_root=None
    )
    base.update(over)
    return Namespace(**base)


def test_default_name_is_bare():
    assert eval_dir_name(make_args(), 400000) == "libri_eval_step400000_t0.3_cfg2.0"


@pytest.mark.parametrize(
    "over, fragment",
    [
        (dict(use_ema=False), "_raw"),
        (dict(num_items=100), "_n100"),
        (dict(seed=1), "_seed1"),
        (dict(asr="openai/whisper-large-v3"), "whisperlarge"),
        (dict(prompt_root="/data/libri_prompts_doraclean"), "doraclean"),
    ],
)
def test_each_knob_marks_the_name(over, fragment):
    assert fragment in eval_dir_name(make_args(**over), 400000)


def test_seeds_do_not_collide():
    """A reseeded rerun resamples the audio, so it must not overwrite the original."""
    names = {eval_dir_name(make_args(seed=s), 400000) for s in (0, 1, 2)}
    assert len(names) == 3


def test_every_pair_of_settings_is_distinguishable():
    knobs = [
        dict(),
        dict(use_ema=False),
        dict(num_items=100),
        dict(seed=1),
        dict(asr="openai/whisper-large-v3"),
        dict(prompt_root="/data/libri_prompts_doraclean_x3"),
    ]
    names = [eval_dir_name(make_args(**k), 400000) for k in knobs]
    for a, b in itertools.combinations(names, 2):
        assert a != b


def test_step_prefixes_do_not_collide_under_globbing():
    """The sweep globs on the name, so step 10000 must not prefix-match 100000."""
    short = eval_dir_name(make_args(), 10000)
    long = eval_dir_name(make_args(), 100000)
    assert not long.startswith(short)


def test_hf_prompt_root_tags_by_repo_name_not_snapshot_path():
    """hf:// roots resolve to a cache path whose basename is a commit hash;
    the tag must come from the user-facing name."""
    a = make_args(prompt_root="/hf-cache/snapshots/0a1b2c3d4e5f")
    a.prompt_name = "librispeech-enhanced-voice-prompts"
    assert eval_dir_name(a, 400000).endswith("voiceprompts")
    assert "0a1b2c3d" not in eval_dir_name(a, 400000)


def test_local_prompt_root_keeps_directory_tag():
    a = make_args(prompt_root="/data/libri_prompts_doraclean/")
    assert eval_dir_name(a, 400000).endswith("ptsdoraclean")
