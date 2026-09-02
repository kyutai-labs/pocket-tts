"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import logging
import multiprocessing.queues
import queue
from collections.abc import Iterator
from typing import Any

import sentencepiece
import torch.multiprocessing as torch_mp

from .loader import DataLoader
from .types import Batch

logger = logging.getLogger(__name__)


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
