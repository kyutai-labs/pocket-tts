"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import json
import logging
import queue
import random
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from safetensors import safe_open

from . import audio
from .manifest import load_entries
from .types import Batch, Entry

logger = logging.getLogger(__name__)


def _prefetch(iterator: Iterator[Batch], depth: int = 4) -> Iterator[Batch]:
    """Run the (synchronous, IO-bound) loader in a background thread."""
    q: queue.Queue[Batch | None] = queue.Queue(maxsize=depth)

    def worker():
        for item in iterator:
            q.put(item)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


class DataLoader:
    def __init__(
        self,
        jsonl: str,
        tokenize: Callable[[str], list[int]],
        batch_size: int,
        sample_rate: int,
        frame_rate: float,
        max_duration_sec: float,
        max_voice_prompt_sec: float,
        rank: int,
        world_size: int,
        seed: int = 0,
        shuffle: bool = True,
        io_workers: int = 16,
    ):
        self.jsonl = jsonl
        self.entries = load_entries(jsonl, rank, world_size)
        self.tokenize = tokenize
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.max_duration_sec = max_duration_sec
        self.max_voice_prompt_sec = max_voice_prompt_sec
        self.shuffle = shuffle
        self.io_workers = io_workers
        self._failures = 0
        self.rng = random.Random(seed)
        self.frame_size = int(sample_rate / frame_rate)
        meta_path = Path(jsonl).with_suffix(".meta.json")
        self.stitch_frames = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.stitch_frames = int(meta["stitch_frames"])
            self.latents_root = Path(jsonl).parent
            logger.info(f"latent mode: stitch_frames={self.stitch_frames} from {meta_path.name}")

    MIN_CUT_SEC = 1.0  # keep at least this much audio on both sides of a cut
    TRAIL_SEC = 0.2  # silence kept after the last word, so EOS has a consistent target

    def _cap_prompt(self, prompt: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], int]:
        """Truncate the prompt to the configured cap; collation pads to batch max."""
        if self.max_voice_prompt_sec > 0:
            prompt_samples = int(self.max_voice_prompt_sec * self.sample_rate)
            prompt = prompt[:prompt_samples]
        return prompt, len(prompt)

    @staticmethod
    def _last_word_end(entry: Entry) -> float | None:
        """End of the last aligned word, i.e. where the speech actually stops.

        None when the entry carries no alignment, which is the documented
        manifest format without `words`.
        """
        if not entry.words:
            return None
        ends = [w["end"] for w in entry.words if w.get("end") is not None]
        return max(ends) if ends else None

    def _choose_cut(self, entry: Entry) -> tuple[float, str] | None:
        """(cut in seconds, transcript of the words after it), None without alignment."""
        # Cut the utterance at a random point between two aligned words:
        # audio before the cut = voice conditioning, audio after = target,
        # paired with the remaining words as text (see training/scripts/align_data.py).
        if not entry.words:
            return None
        cuts = []
        for i in range(1, len(entry.words)):
            prev, cur = entry.words[i - 1], entry.words[i]
            if prev["end"] is None or cur["start"] is None:
                continue
            cut = 0.5 * (prev["end"] + cur["start"])
            if cut >= self.MIN_CUT_SEC and entry.duration - cut >= self.MIN_CUT_SEC:
                cuts.append((cut, i))
        if cuts:
            # The prompt is the (contiguous) start of the utterance.
            # Eligible cuts are word boundaries whose preceding word ends
            # inside the prompt window; the draw is uniform over 1..k
            # eligible words, so prompt length varies and the target
            # keeps most of the utterance.
            window = self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else float("inf")
            eligible = [
                (c, i)
                for c, i in cuts
                if entry.words[i - 1].get("end") is not None and entry.words[i - 1]["end"] < window
            ]
            cuts = eligible or cuts[:1]  # degenerate windows: earliest valid cut
        if not cuts:
            return None
        cut, i = self.rng.choice(cuts)
        return cut, " ".join(w["word"] for w in entry.words[i:])

    def _sample(self, entry: Entry) -> tuple[Any, ...]:
        """(wav, tokens, prompt wav, prompt samples), or _sample_latent's tuple."""
        if entry.latents_file is not None:
            return self._sample_latent(entry)
        chosen = self._choose_cut(entry)
        if chosen is not None:
            cut, text = chosen
            tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
            # Trim to the last word (plus a short tail) rather than the end
            # of the file: 12% of utterances carry >1s of trailing silence,
            # and training on it teaches the model to emit silence instead of
            # EOS, so generations never terminate. dora does the same thing
            # via align_wav_on_words.
            end = entry.duration
            last = self._last_word_end(entry)
            if last is not None and last > cut:
                end = min(entry.duration, last + self.TRAIL_SEC)
            wav = audio._load_window(
                entry.path,
                entry.start + cut,
                min(end - cut, self.max_duration_sec),
                self.sample_rate,
            )
            # The prompt is everything from the utterance start to the
            # cut (bounded by the window via cut selection).
            prompt, length = self._cap_prompt(
                audio._load_window(entry.path, entry.start, cut, self.sample_rate)
            )
            return wav, tokens, prompt, length
        last = self._last_word_end(entry)
        end = min(entry.duration, last + self.TRAIL_SEC) if last is not None else entry.duration
        duration = min(end, self.max_duration_sec)
        wav = audio._load_window(entry.path, entry.start, duration, self.sample_rate)
        tokens = torch.tensor(self.tokenize(entry.transcript), dtype=torch.long)
        # No alignment: voice prompt from a random window of the same file.
        fallback_sec = self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else 3.0
        prompt_start = self.rng.uniform(0, max(0.0, entry.duration - fallback_sec))
        prompt, length = self._cap_prompt(
            audio._load_window(
                entry.path, entry.start + prompt_start, fallback_sec, self.sample_rate
            )
        )
        return wav, tokens, prompt, length

    def _load_latents(self, latents_file: str) -> torch.Tensor:
        path = self.latents_root / latents_file
        with safe_open(str(path), framework="pt") as f:
            return f.get_tensor("latents")

    def _latent_cut(self, entry: Entry, stored: int) -> tuple[int, str]:
        chosen = self._choose_cut(entry)
        if chosen is None or stored <= 1:
            return 0, entry.transcript
        cut, text = chosen
        cut_frames = min(max(round(cut * self.frame_rate), 1), stored - 1)
        return cut_frames, text

    def _latent_target_frames(self, entry: Entry, cut_frames: int, stored: int) -> int:
        cut_sec = cut_frames / self.frame_rate
        end = entry.duration
        last = self._last_word_end(entry)
        if last is not None and last > cut_sec:
            end = min(entry.duration, last + self.TRAIL_SEC)
        target_frames = int(min(end - cut_sec, self.max_duration_sec) * self.frame_rate)
        return max(1, min(target_frames, stored - cut_frames))

    def _latent_prompt(self, lat: torch.Tensor, cut_frames: int) -> torch.Tensor:
        stored = lat.shape[0]
        cap = (
            max(1, int(self.max_voice_prompt_sec * self.frame_rate))
            if self.max_voice_prompt_sec > 0
            else stored
        )
        if cut_frames > 0:
            return lat[: min(cut_frames, cap)]
        start = self.rng.randint(0, max(0, stored - cap))
        return lat[start : start + cap]

    def _sample_latent(self, entry: Entry) -> tuple[Any, ...]:
        """(stitch wav, tokens, prompt latents, tail latents, target frames)."""
        assert self.stitch_frames > 0, f"{entry.path}: latents entry but no meta file loaded"
        assert entry.latents_file is not None, f"{entry.path}: not a latents entry"
        lat = self._load_latents(entry.latents_file)
        cut_frames, text = self._latent_cut(entry, lat.shape[0])
        tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
        target_frames = self._latent_target_frames(entry, cut_frames, lat.shape[0])
        stitch_frames = min(self.stitch_frames, target_frames)
        stitch = audio._load_window(
            entry.path,
            entry.start + cut_frames / self.frame_rate,
            stitch_frames / self.frame_rate,
            self.sample_rate,
        )
        tail = lat[cut_frames + stitch_frames : cut_frames + target_frames]
        return stitch, tokens, self._latent_prompt(lat, cut_frames), tail, target_frames

    @staticmethod
    def _pad_latents(seqs: tuple[torch.Tensor, ...], min_len: int) -> torch.Tensor:
        length = max(min_len, max(s.shape[0] for s in seqs))
        out = torch.zeros(len(seqs), length, seqs[0].shape[-1])
        for b, s in enumerate(seqs):
            out[b, : s.shape[0]] = s
        return out

    def _collate_stitch_audio(self, stitches: tuple[npt.NDArray[np.float32], ...]) -> torch.Tensor:
        stitch_samples = self.stitch_frames * self.frame_size
        audio = torch.zeros(len(stitches), 1, stitch_samples)
        for b, w in enumerate(stitches):
            n = min(len(w), stitch_samples)
            audio[b, 0, :n] = torch.from_numpy(w[:n])
        return audio

    def _collate_latent(self, batch: list[tuple[Any, ...]]) -> Batch:
        stitches, tokens, prompts, tails, target_frames = zip(*batch, strict=True)
        num_prompt_frames = torch.tensor([max(1, p.shape[0]) for p in prompts], dtype=torch.long)
        return Batch(
            self._collate_stitch_audio(stitches),
            torch.tensor(target_frames, dtype=torch.long),
            list(tokens),
            torch.zeros(len(stitches), 1, 0),
            num_prompt_frames,
            tail_latents=self._pad_latents(tails, 0),
            prompt_latents=self._pad_latents(prompts, 1),
        )

    def get_entry(self, index: int) -> Entry:
        d = json.loads(self.entries[index])
        return Entry(
            d["path"],
            float(d["duration"]),
            d["transcript"],
            d.get("words"),
            float(d.get("start", 0.0)),
            d.get("latents_file"),
        )

    def _sample_or_none(self, entry: Entry) -> tuple[Any, ...] | None:
        try:
            return self._sample(entry)
        except Exception as exc:  # noqa: BLE001 — skip unreadable samples, whatever the cause
            self._failures += 1
            if self._failures % 1000 == 1:
                logger.warning(f"skipping unreadable sample ({self._failures} so far): {exc}")
            return None

    def __iter__(self) -> Iterator[Batch]:
        # Batches are produced in a background thread (the loader is IO-bound)
        # so the GPU never waits on network storage.
        return _prefetch(self._batches())

    def _batches(self) -> Iterator[Batch]:
        self._failures = 0
        # Each sample is two small reads from network storage, so the loader is
        # latency-bound rather than CPU-bound: fetching a batch's samples
        # concurrently keeps the GPUs fed. sphn releases the GIL, so threads are
        # enough. Without this a cold, wide corpus starves training (~2x).
        pool = ThreadPoolExecutor(max_workers=self.io_workers)
        if len(self.entries) < self.batch_size:
            raise ValueError(
                f"{len(self.entries)} usable entries for this rank but batch_size="
                f"{self.batch_size}: a batch can never be filled. Lower batch_size, "
                "use fewer ranks, or point at a larger manifest."
            )
        while True:
            yielded = 0
            order = list(range(len(self.entries)))
            if self.shuffle:
                self.rng.shuffle(order)
            samples = []
            for chunk_start in range(0, len(order), self.batch_size):
                chunk = order[chunk_start : chunk_start + self.batch_size]
                # Parse sequentially, outside the pool: get_entry is GIL-bound
                # (unlike sphn's audio reads), so parsing it on the worker
                # threads just contends with itself instead of overlapping.
                chunk_entries = [self.get_entry(i) for i in chunk]
                got = [s for s in pool.map(self._sample_or_none, chunk_entries) if s is not None]
                samples.extend(got)
                if len(samples) < self.batch_size:
                    continue
                batch, samples = samples[: self.batch_size], samples[self.batch_size :]
                yielded += 1
                if self.stitch_frames:
                    yield self._collate_latent(batch)
                    continue
                wavs, tokens, prompts, prompt_lens = zip(*batch, strict=True)
                max_len = max(len(w) for w in wavs)
                audio = torch.zeros(len(wavs), 1, max_len)
                frames = torch.zeros(len(wavs), dtype=torch.long)
                for b, w in enumerate(wavs):
                    audio[b, 0, : len(w)] = torch.from_numpy(w)
                    frames[b] = max(1, int(len(w) * self.frame_rate / self.sample_rate))
                max_prompt = max(len(p) for p in prompts)
                voice = torch.zeros(len(prompts), 1, max_prompt)
                for b, pr in enumerate(prompts):
                    voice[b, 0, : len(pr)] = torch.from_numpy(pr)
                num_voice_prompt_frames = torch.tensor(
                    [max(1, int(n * self.frame_rate / self.sample_rate)) for n in prompt_lens],
                    dtype=torch.long,
                )
                yield Batch(audio, frames, list(tokens), voice, num_voice_prompt_frames)
            if not yielded:
                raise ValueError(
                    f"no readable samples in {self.jsonl}: every entry failed to load "
                    f"({self._failures} failures). Check the paths in the manifest."
                )
