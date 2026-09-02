"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import json
import logging
import multiprocessing.queues
import queue
import random
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import sentencepiece
import sphn
import torch
import torch.multiprocessing as torch_mp
from safetensors import safe_open

from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.models.mimi import MimiModel

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    path: str
    duration: float
    transcript: str
    words: list[dict[str, Any]] | None = None  # [{"word", "start", "end"}] from align_data
    start: float = 0.0  # offset of the utterance inside the audio file (long recordings)
    latents_file: str | None = None


@dataclass
class Batch:
    audio: torch.Tensor  # [B, 1, samples], zero-padded
    num_audio_frames: torch.Tensor  # [B] valid codec frames per sample
    text_tokens: list[torch.Tensor]  # ragged, one [L_b] long tensor per sample
    voice_audio: torch.Tensor  # [B, 1, prompt_samples]
    num_voice_prompt_frames: torch.Tensor  # [B] valid codec frames of each voice prompt
    tail_latents: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None


def load_entries(path: str, rank: int, world_size: int) -> list[str]:
    entries = []
    with Path(path).open() as f:
        for idx, line in enumerate(f):
            if idx % world_size != rank:
                continue
            # We parse lazily, 6x less memory
            entries.append(line)
    logger.info(f"loaded {len(entries)} entries from {path} (rank {rank}/{world_size})")
    assert entries, f"no entries for rank {rank} in {path}"
    return entries


def _load_window(
    path: str, start_sec: float, duration_sec: float, sample_rate: int
) -> npt.NDArray[np.float32]:
    wav, sr = sphn.read(path, start_sec=start_sec, duration_sec=duration_sec)
    wav = wav.mean(axis=0)  # mono
    if sr != sample_rate:
        resampled = convert_audio(torch.from_numpy(wav)[None], int(sr), int(sample_rate), 1)
        wav = resampled[0].numpy()
    return wav


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
            wav = _load_window(
                entry.path,
                entry.start + cut,
                min(end - cut, self.max_duration_sec),
                self.sample_rate,
            )
            # The prompt is everything from the utterance start to the
            # cut (bounded by the window via cut selection).
            prompt, length = self._cap_prompt(
                _load_window(entry.path, entry.start, cut, self.sample_rate)
            )
            return wav, tokens, prompt, length
        last = self._last_word_end(entry)
        end = min(entry.duration, last + self.TRAIL_SEC) if last is not None else entry.duration
        duration = min(end, self.max_duration_sec)
        wav = _load_window(entry.path, entry.start, duration, self.sample_rate)
        tokens = torch.tensor(self.tokenize(entry.transcript), dtype=torch.long)
        # No alignment: voice prompt from a random window of the same file.
        fallback_sec = self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else 3.0
        prompt_start = self.rng.uniform(0, max(0.0, entry.duration - fallback_sec))
        prompt, length = self._cap_prompt(
            _load_window(entry.path, entry.start + prompt_start, fallback_sec, self.sample_rate)
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
        stitch = _load_window(
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


def _feed_queue(
    q: "multiprocessing.queues.Queue[Batch]",  # not subscriptable at runtime on 3.10
    sentence_piece_proto: bytes,
    loader_kwargs: dict[str, Any],
):
    torch_mp.set_sharing_strategy("file_system")
    sentence_piece = sentencepiece.SentencePieceProcessor()
    sentence_piece.load_from_serialized_proto(sentence_piece_proto)
    loader = DataLoader(tokenize=sentence_piece.encode, **loader_kwargs)
    for batch in loader:
        q.put(batch)


class SubprocessDataLoader:
    def __init__(
        self,
        jsonl: str,
        sentence_piece: sentencepiece.SentencePieceProcessor,
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
        num_procs: int = 3,
        depth: int = 8,
    ):
        ctx = torch_mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=depth)
        self._procs = []
        for i in range(num_procs):
            loader_kwargs = {
                "jsonl": jsonl,
                "batch_size": batch_size,
                "sample_rate": sample_rate,
                "frame_rate": frame_rate,
                "max_duration_sec": max_duration_sec,
                "max_voice_prompt_sec": max_voice_prompt_sec,
                "rank": rank * num_procs + i,
                "world_size": world_size * num_procs,
                "seed": seed + rank * num_procs + i,
                "shuffle": shuffle,
                "io_workers": io_workers,
            }
            proc = ctx.Process(
                target=_feed_queue,
                args=(self._queue, sentence_piece.serialized_model_proto(), loader_kwargs),
                daemon=True,
                name=f"dataloader-{i}",
            )
            proc.start()
            self._procs.append(proc)

    def _check_procs(self):
        dead = [p for p in self._procs if not p.is_alive()]
        if dead:
            raise RuntimeError(
                f"dataloader subprocess(es) died (exit codes "
                f"{[p.exitcode for p in dead]}), check their tracebacks above"
            )

    def __iter__(self) -> Iterator[Batch]:
        batches = 0
        while True:
            try:
                batch = self._queue.get(timeout=60)
            except queue.Empty:
                self._check_procs()
                continue
            batches += 1
            if batches % 100 == 0:
                self._check_procs()
            yield batch


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


@torch.no_grad()
def encode_batch(
    mimi: MimiModel, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.tail_latents is not None:
        stitch = mimi.encode_to_latent(batch.audio.to(device))
        latents = torch.cat([stitch, batch.tail_latents.to(device)], dim=1)
        T = latents.shape[1]
        num_audio_frames = batch.num_audio_frames.to(device).clamp(max=T)
        mask = torch.arange(T, device=device)[None, :] < num_audio_frames[:, None]
        assert batch.prompt_latents is not None  # set together with tail_latents
        voice_prompt_latents = batch.prompt_latents.to(device)
        num_voice_prompt_frames = batch.num_voice_prompt_frames.to(device).clamp(
            max=voice_prompt_latents.shape[1]
        )
        return latents.float(), mask, voice_prompt_latents.float(), num_voice_prompt_frames
    audio = batch.audio.to(device)
    latents = mimi.encode_to_latent(audio)  # [B, T, C]
    T = latents.shape[1]
    num_audio_frames = batch.num_audio_frames.to(device).clamp(max=T)
    mask = torch.arange(T, device=device)[None, :] < num_audio_frames[:, None]
    voice_prompt_latents = mimi.encode_to_latent(batch.voice_audio.to(device))
    num_voice_prompt_frames = batch.num_voice_prompt_frames.to(device).clamp(
        max=voice_prompt_latents.shape[1]
    )
    return latents.float(), mask, voice_prompt_latents.float(), num_voice_prompt_frames
