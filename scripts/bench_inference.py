"""
Reproducible inference latency / RTF benchmark for Pocket TTS.

Inspired by nanoGPT's bench.py: optional warm-up, then timed runs with sync.
RTS (real-time speed) matches scripts/evaluate_quantization.py:
    RTS = audio_duration_sec / wall_clock_sec  (>1 means faster than real-time)

Usage:
  uv run python scripts/bench_inference.py
  uv run python scripts/bench_inference.py --device cuda --runs 5 --json-out bench.json
  uv run python scripts/bench_inference.py --quantize --warmup 1 --runs 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from pocket_tts import TTSModel

# Short, fixed text so runs are comparable across machines / PRs.
DEFAULT_BENCH_TEXT = (
    "The quick brown fox jumps over the lazy dog. Benchmarking inference latency on CPU and GPU."
)
DEFAULT_VOICE = "alba"


@dataclass
class RunMetrics:
    run_index: int
    audio_duration_sec: float
    wall_clock_sec: float
    rts: float
    latency_ms_per_audio_sec: float


@dataclass
class BenchReport:
    device: str
    quantize: bool
    voice: str
    text_chars: int
    warmup_runs: int
    timed_runs: int
    load_time_sec: float
    runs: list[RunMetrics]
    mean_rts: float
    mean_wall_clock_sec: float
    mean_audio_duration_sec: float
    timestamp_utc: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs"] = [asdict(r) for r in self.runs]
        return d


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_benchmark(
    *,
    text: str = DEFAULT_BENCH_TEXT,
    voice: str = DEFAULT_VOICE,
    device: str = "cpu",
    quantize: bool = False,
    warmup_runs: int = 1,
    timed_runs: int = 3,
    language: str | None = None,
) -> BenchReport:
    t0 = time.perf_counter()
    tts_model = TTSModel.load_model(language=language, quantize=quantize)
    load_time = time.perf_counter() - t0
    tts_model.to(device)
    voice_state = tts_model.get_state_for_audio_prompt(voice)

    def one_run() -> RunMetrics:
        _sync_device(device)
        start = time.perf_counter()
        audio = tts_model.generate_audio(model_state=voice_state, text_to_generate=text)
        _sync_device(device)
        wall = time.perf_counter() - start
        audio_sec = len(audio) / tts_model.sample_rate
        rts = audio_sec / wall if wall > 0 else 0.0
        latency_ms = (wall / audio_sec * 1000) if audio_sec > 0 else 0.0
        return RunMetrics(
            run_index=0,
            audio_duration_sec=round(audio_sec, 4),
            wall_clock_sec=round(wall, 4),
            rts=round(rts, 4),
            latency_ms_per_audio_sec=round(latency_ms, 2),
        )

    for _ in range(warmup_runs):
        one_run()

    runs: list[RunMetrics] = []
    for i in range(timed_runs):
        m = one_run()
        m.run_index = i + 1
        runs.append(m)

    rts_vals = [r.rts for r in runs]
    wall_vals = [r.wall_clock_sec for r in runs]
    audio_vals = [r.audio_duration_sec for r in runs]

    return BenchReport(
        device=device,
        quantize=quantize,
        voice=voice,
        text_chars=len(text),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        load_time_sec=round(load_time, 3),
        runs=runs,
        mean_rts=round(statistics.mean(rts_vals), 4),
        mean_wall_clock_sec=round(statistics.mean(wall_vals), 4),
        mean_audio_duration_sec=round(statistics.mean(audio_vals), 4),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pocket TTS inference benchmark")
    parser.add_argument("--text", default=DEFAULT_BENCH_TEXT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs (default 1)")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs (default 3)")
    parser.add_argument("--language", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available.", file=sys.stderr)
        return 1

    report = run_benchmark(
        text=args.text,
        voice=args.voice,
        device=args.device,
        quantize=args.quantize,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        language=args.language,
    )

    print(f"device={report.device} quantize={report.quantize} load={report.load_time_sec}s")
    print(f"warmup={report.warmup_runs} timed_runs={report.timed_runs}")
    for r in report.runs:
        print(
            f"  run {r.run_index}: audio={r.audio_duration_sec}s "
            f"wall={r.wall_clock_sec}s RTS={r.rts}x "
            f"latency={r.latency_ms_per_audio_sec}ms/s_audio"
        )
    print(
        f"mean RTS={report.mean_rts}x "
        f"mean_wall={report.mean_wall_clock_sec}s "
        f"mean_audio={report.mean_audio_duration_sec}s"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
