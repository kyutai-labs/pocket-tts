"""Dataloader behaviour that silently degrades training when it breaks."""

import json

import numpy as np
import sphn

from training.dataloader import DataLoader, load_entries

SR = 24000


def _write_wav(path, seconds=6.0):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    sphn.write_wav(str(path), (0.2 * np.sin(2 * np.pi * 140 * t)).astype(np.float32), SR)


def _manifest(tmp_path, n=8, words=True, duration=6.0):
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


def _loader(manifest, **kw):
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


def test_rank_sharding_partitions_entries_without_overlap(tmp_path):
    m = _manifest(tmp_path, n=8)
    shards = [load_entries(m, rank, 4) for rank in range(4)]
    assert sum(len(s) for s in shards) == 8
    assert all(len(s) == 2 for s in shards)


def test_prompt_respects_the_configured_cap(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path), max_voice_prompt_sec=1.0)))
    assert (batch.num_voice_prompt_frames.float() / 12.5).max().item() <= 1.0 + 1e-6


def test_batches_have_the_requested_size(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path, n=8), batch_size=4)))
    assert batch.audio.shape[0] == 4
    assert len(batch.text_tokens) == 4


def test_target_audio_never_exceeds_max_duration(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path, duration=30.0), max_duration_sec=5.0)))
    assert batch.audio.shape[-1] <= int(5.0 * SR) + 1


def test_unaligned_manifest_still_yields_batches(tmp_path):
    """Manifests without word alignments are a documented input: the loader
    falls back to a random window as the prompt instead of hanging."""
    loader = _loader(_manifest(tmp_path, n=8, words=False))
    batch = next(iter(loader))
    assert batch.audio.shape[0] == 2


def test_entry_start_offsets_into_a_shared_file(tmp_path):
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
        f.write(json.dumps({"path": str(audio_path), "duration": 5.0, "transcript": "low"}) + "\n")
        f.write(
            json.dumps(
                {"path": str(audio_path), "duration": 5.0, "transcript": "high", "start": 5.0}
            )
            + "\n"
        )

    loader = _loader(str(manifest), batch_size=2)
    low_entry, high_entry = loader.get_entry(0), loader.get_entry(1)
    assert low_entry.start == 0.0
    assert high_entry.start == 5.0

    low_wav, *_ = loader._sample(low_entry)
    high_wav, *_ = loader._sample(high_entry)

    def dominant_freq(x):
        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1 / SR)
        return freqs[np.argmax(spectrum)]

    assert abs(dominant_freq(low_wav) - low_hz) < 2
    assert abs(dominant_freq(high_wav) - high_hz) < 2
