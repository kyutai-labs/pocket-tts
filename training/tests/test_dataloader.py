"""Dataloader behaviour that silently degrades training when it breaks."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import safetensors.torch
import sphn
import torch

from training.dataloader import DataLoader, load_entries

SR = 24000


def _write_wav(path: Path, seconds: float = 6.0):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    sphn.write_wav(str(path), (0.2 * np.sin(2 * np.pi * 140 * t)).astype(np.float32), SR)


def _manifest(tmp_path: Path, n: int = 8, words: bool = True, duration: float = 6.0) -> str:
    wav = tmp_path / "a.flac"
    _write_wav(wav, duration)
    path = tmp_path / "m.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            entry = {"path": str(wav), "duration": duration, "transcript": "one two three four"}
            if words:
                entry["words"] = [
                    {"word": w, "start": 0.8 * j, "end": 0.8 * j + 0.6}
                    for j, w in enumerate(entry["transcript"].split())
                ]
            f.write(json.dumps(entry) + "\n")
    return str(path)


def _loader(manifest: str, **kw: Any) -> DataLoader:  # noqa: ANN401 -- DataLoader passthrough
    kw.setdefault("batch_size", 2)
    return DataLoader(
        manifest,
        lambda s: [1] * max(1, len(s) // 3),
        kw.pop("batch_size"),
        SR,
        12.5,
        kw.pop("max_duration_sec", 30.0),
        kw.pop("max_voice_prompt_sec", 3.0),
        0,
        1,
        seed=0,
        shuffle=False,
        **kw,
    )


def test_rank_sharding_partitions_entries_without_overlap(tmp_path: Path):
    m = _manifest(tmp_path, n=8)
    shards = [load_entries(m, rank, 4) for rank in range(4)]
    assert sum(len(s) for s in shards) == 8
    assert all(len(s) == 2 for s in shards)


def test_prompt_respects_the_configured_cap(tmp_path: Path):
    batch = next(iter(_loader(_manifest(tmp_path), max_voice_prompt_sec=1.0)))
    assert (batch.num_voice_prompt_frames.float() / 12.5).max().item() <= 1.0 + 1e-6


def test_batches_have_the_requested_size(tmp_path: Path):
    batch = next(iter(_loader(_manifest(tmp_path, n=8), batch_size=4)))
    assert batch.audio.shape[0] == 4
    assert len(batch.text_tokens) == 4


def test_target_audio_never_exceeds_max_duration(tmp_path: Path):
    batch = next(iter(_loader(_manifest(tmp_path, duration=30.0), max_duration_sec=5.0)))
    assert batch.audio.shape[-1] <= int(5.0 * SR) + 1


def test_unaligned_manifest_is_refused(tmp_path: Path):
    """Without word alignments the prompt would be a window of the target itself,
    so the loader refuses the manifest instead of training a prompt-copier."""
    loader = _loader(_manifest(tmp_path, n=8, words=False))
    with pytest.raises(ValueError, match="align_data"):
        next(iter(loader))


def test_entry_start_offsets_into_a_shared_file(tmp_path: Path):
    """Two utterances can share one audio file: each entry reads its own
    window, at `start`, out of the shared file."""
    low_hz, high_hz = 220, 880
    t = np.linspace(0, 10.0, int(10.0 * SR), endpoint=False)
    wav = np.where(
        t < 5.0, 0.2 * np.sin(2 * np.pi * low_hz * t), 0.2 * np.sin(2 * np.pi * high_hz * t)
    )
    audio_path = tmp_path / "shared.flac"
    sphn.write_wav(str(audio_path), wav.astype(np.float32), SR)

    manifest = tmp_path / "m.jsonl"
    with open(manifest, "w") as f:
        words = [{"word": "w", "start": 0.5, "end": 1.5}, {"word": "x", "start": 3.0, "end": 4.0}]
        f.write(
            json.dumps(
                {"path": str(audio_path), "duration": 5.0, "transcript": "w x", "words": words}
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "path": str(audio_path),
                    "duration": 5.0,
                    "transcript": "w x",
                    "start": 5.0,
                    "words": words,
                }
            )
            + "\n"
        )

    loader = _loader(str(manifest), batch_size=2)
    low_entry, high_entry = loader.get_entry(0), loader.get_entry(1)
    assert low_entry.start == 0.0
    assert high_entry.start == 5.0

    low_wav, *_ = loader._sample(low_entry)
    high_wav, *_ = loader._sample(high_entry)

    def dominant_freq(x: npt.NDArray[np.float32]) -> float:
        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1 / SR)
        return freqs[np.argmax(spectrum)]

    assert abs(dominant_freq(low_wav) - low_hz) < 2
    assert abs(dominant_freq(high_wav) - high_hz) < 2


def test_shard_smaller_than_the_bucket_pool_names_the_knobs(tmp_path: Path):
    loader = _loader(_manifest(tmp_path, n=8), batch_size=2, num_bucket_batches=5)
    with pytest.raises(ValueError, match="num_bucket_batches"):
        next(iter(loader))


def test_startup_check_rejects_manifests_too_small_for_the_loaders(tmp_path: Path):
    from training.args import TrainArgs
    from training.train_utils import check_manifest_sizes

    args = TrainArgs()
    args.batch_size = 2
    args.data.train_jsonl = _manifest(tmp_path, n=12)
    args.data.valid_jsonl = ""
    args.data.loader_procs, args.data.num_bucket_batches = 3, 2
    check_manifest_sizes(args, world_size=1)  # 12 // 3 = 4 entries per shard, 2 x 2 needed
    with pytest.raises(SystemExit, match="num_bucket_batches"):
        check_manifest_sizes(args, world_size=2)  # 12 // 6 = 2 < 4


def _voice_manifest(tmp_path: Path, voices: list[str | None], words: bool = False) -> str:
    """Unaligned entries (a voice-bank target needs no cut), each naming a voice."""
    wav = tmp_path / "v.flac"
    _write_wav(wav, 3.0)
    path = tmp_path / "voices.jsonl"
    with open(path, "w") as f:
        for voice in voices:
            entry: dict[str, Any] = {"path": str(wav), "duration": 3.0, "transcript": "one two"}
            if voice is not None:
                entry["voice"] = voice
            if words:
                entry["words"] = [
                    {"word": "one", "start": 0.2, "end": 0.6},
                    {"word": "two", "start": 2.0, "end": 2.6},
                ]
            f.write(json.dumps(entry) + "\n")
    return str(path)


def _bank() -> dict[str, torch.Tensor]:
    # Constant rows tell the voices apart in the batch: voice "a" is all 1, "b" all 2.
    return {"b": torch.full((125, 4), 2.0), "a": torch.full((125, 4), 1.0)}


def test_voice_bank_prompts_and_ids(tmp_path: Path):
    loader = _loader(
        _voice_manifest(tmp_path, ["a", "b", "b", "a"]), batch_size=4, voice_bank=_bank()
    )
    batch = next(iter(loader))
    assert batch.voice_ids is not None and batch.prompt_latents is not None
    assert batch.voice_ids.tolist() == [0, 1, 1, 0], "rows follow the sorted voice names"
    assert batch.num_voice_prompt_frames.tolist() == [125] * 4, "no crop by default"
    for b, voice_id in enumerate(batch.voice_ids.tolist()):
        assert (batch.prompt_latents[b] == voice_id + 1).all(), "prompt is the voice's reference"
    assert batch.voice_audio.shape[-1] == 0, "no prompt audio to encode"
    assert batch.audio.shape[-1] == int(3.0 * SR), "the target is the whole utterance"
    assert [t.numel() for t in batch.text_tokens] == [2] * 4, "with the full transcript"


def test_voice_bank_crop_lengths(tmp_path: Path):
    loader = _loader(
        _voice_manifest(tmp_path, ["a"] * 2),
        batch_size=2,
        voice_bank=_bank(),
        voice_prompt_crop_prob=0.5,
        voice_prompt_min_sec=2.5,
    )
    entry = loader.get_entry(0)
    lengths = [loader._voice_prompt(entry)[0].shape[0] for _ in range(400)]
    full = sum(n == 125 for n in lengths)
    assert 140 < full < 260, f"about half the prompts keep the full reference, got {full}/400"
    assert min(lengths) >= 31 and len(set(lengths)) > 40, "crops spread over 2.5 s .. 10 s"


def test_voice_bank_target_drops_words_past_the_max_duration(tmp_path: Path):
    loader = _loader(
        _voice_manifest(tmp_path, ["a"], words=True), max_duration_sec=1.5, voice_bank=_bank()
    )
    duration, text = loader._voice_target(loader.get_entry(0))
    assert duration == 1.5 and text == "one"


@pytest.mark.parametrize("voice", [None, "z"])
def test_voice_bank_refuses_unknown_voices(tmp_path: Path, voice: str | None):
    loader = _loader(_voice_manifest(tmp_path, [voice] * 2), voice_bank=_bank())
    with pytest.raises(ValueError, match="TrainArgs.voices"):
        next(iter(loader))


def test_lut_only_ids_and_empty_prompt(tmp_path: Path):
    """LUT-only mode (TrainArgs.voice_names): every row carries a voice id for the LUT, the whole
    utterance is the target, and there is no voice prompt at all."""
    loader = _loader(
        _voice_manifest(tmp_path, ["a", "b", "b", "a"]), batch_size=4, lut_names=["b", "a"]
    )
    batch = next(iter(loader))
    assert batch.voice_ids is not None, "LUT-only still needs voice ids"
    assert batch.voice_ids.tolist() == [0, 1, 1, 0], "rows follow the sorted voice names"
    assert batch.prompt_latents is None and batch.tail_latents is None, "no prompt latents"
    assert batch.voice_audio.shape[-1] == 0, "no voice prompt audio"
    assert batch.num_voice_prompt_frames.tolist() == [0] * 4, "zero voice-prompt frames"
    assert batch.audio.shape[-1] == int(3.0 * SR), "the target is the whole utterance"
    assert [t.numel() for t in batch.text_tokens] == [2] * 4, "with the full transcript"


@pytest.mark.parametrize("voice", [None, "z"])
def test_lut_only_refuses_unknown_voices(tmp_path: Path, voice: str | None):
    loader = _loader(_voice_manifest(tmp_path, [voice] * 2), lut_names=["a", "b"])
    with pytest.raises(ValueError, match="TrainArgs.voice_names"):
        next(iter(loader))


def test_voice_bank_with_precomputed_latents(tmp_path: Path):
    manifest = Path(_voice_manifest(tmp_path, ["b", "a"]))
    lat_dir = tmp_path / "lat"
    lat_dir.mkdir()
    rows = []
    for i, line in enumerate(manifest.read_text().splitlines()):
        entry = json.loads(line)
        safetensors.torch.save_file({"latents": torch.randn(37, 4)}, str(lat_dir / f"{i}.st"))
        entry["latents_file"] = f"lat/{i}.st"
        rows.append(json.dumps(entry))
    latents_manifest = tmp_path / "voices_latents.jsonl"
    latents_manifest.write_text("\n".join(rows) + "\n")
    latents_manifest.with_suffix(".meta.json").write_text(json.dumps({"stitch_frames": 4}))
    batch = next(iter(_loader(str(latents_manifest), voice_bank=_bank())))
    assert batch.voice_ids is not None and batch.prompt_latents is not None
    assert sorted(batch.voice_ids.tolist()) == [0, 1]
    assert batch.tail_latents is not None
    # Target: 3 s of audio minus nothing (no words) = 37 frames: 4 stitched + 33 stored.
    assert batch.num_audio_frames.tolist() == [37, 37]
    assert batch.tail_latents.shape[1] == 33
