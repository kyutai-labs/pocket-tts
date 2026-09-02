"""Measure data-loader throughput for one training rank, independent of the GPU.

    python -m training.dataloader.bench MANIFEST.jsonl --procs 1,3,6 --io-workers 16,32

Prints items/s per (loader_procs, io_workers) so the loader can be checked against
what a rank consumes (batch_size x steps/s). Training is input-bound whenever the
loader's items/s is below that.
"""

import argparse
import logging
import os
import time

import sentencepiece

from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import download_if_necessary
from training.dataloader.subproc import SubprocessDataLoader


def measure(loader: SubprocessDataLoader, batches: int, warmup: int) -> float:
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    items = 0
    for _ in range(batches):
        batch = next(it)
        items += batch.num_audio_frames.shape[0]
    return items / (time.perf_counter() - t0)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest")
    parser.add_argument("--config", default="pocket_tts/config/english.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--world-size", type=int, default=4, help="ranks the corpus is sharded over"
    )
    parser.add_argument("--procs", default="1,3,6")
    parser.add_argument("--io-workers", default="16,32")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--max-voice-prompt-sec", type=float, default=5.0)
    args = parser.parse_args()

    config = load_config(args.config)
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(download_if_necessary(config.flow_lm.lookup_table.tokenizer_path))
    )
    sample_rate = config.mimi.sample_rate
    frame_rate = config.mimi.frame_rate
    print(
        f"{'procs':>5} {'io_workers':>10} {'items/s':>9}   (batch {args.batch_size}, one rank of {args.world_size})"
    )
    for procs in (int(p) for p in args.procs.split(",")):
        for io_workers in (int(w) for w in args.io_workers.split(",")):
            loader = SubprocessDataLoader(
                args.manifest,
                sp,
                args.batch_size,
                sample_rate,
                frame_rate,
                args.max_duration_sec,
                args.max_voice_prompt_sec,
                rank=0,
                world_size=args.world_size,
                seed=int(time.time()),
                io_workers=io_workers,
                num_procs=procs,
            )
            t0 = time.perf_counter()
            it = iter(loader)
            next(it)
            print(f"      first batch after {time.perf_counter() - t0:.1f}s", flush=True)
            rate = measure(loader, args.batches, args.warmup)
            print(f"{procs:>5} {io_workers:>10} {rate:>9.1f}", flush=True)
            for p in loader._procs:
                p.terminate()
                p.join(timeout=10)
    # Daemon loader processes can hold the interpreter at shutdown; nothing left to flush.
    os._exit(0)


if __name__ == "__main__":
    main()
