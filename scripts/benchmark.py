#!/usr/bin/env python3
"""
Benchmark script for pocket-tts inference performance.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --runs 10 --quantize
    python scripts/benchmark.py --device cpu --text "Custom text to benchmark"
"""

import argparse
import gc
import statistics
import time
from contextlib import nullcontext

import torch

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_AUDIO_PROMPT, DEFAULT_VARIANT


BENCHMARK_TEXTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer.",
]


def move_state_to_device(state: dict, device: str) -> dict:
    """Recursively move all tensors in a nested state dict to the specified device."""
    result = {}
    for key, value in state.items():
        if isinstance(value, dict):
            result[key] = move_state_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def get_gpu_memory_mb() -> float | None:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


def get_gpu_memory_peak_mb() -> float | None:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return None


def benchmark_single(
    model: TTSModel,
    model_state: dict,
    text: str,
    warmup: bool = False,
    no_flash: bool = False,
) -> dict:
    """Run a single benchmark iteration."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Disable Flash Attention if requested (forces MATH backend like CPU uses)
    if no_flash:
        ctx = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    else:
        ctx = nullcontext()

    start_time = time.perf_counter()
    ttfb_time = None
    chunks = []

    with ctx:
        for chunk in model.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text,
            copy_state=True,
        ):
            if ttfb_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttfb_time = time.perf_counter()
            chunks.append(chunk)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    audio = torch.cat(chunks, dim=-1) if chunks else torch.tensor([])
    elapsed_ms = (end_time - start_time) * 1000
    ttfb_ms = (ttfb_time - start_time) * 1000 if ttfb_time else elapsed_ms
    audio_duration_ms = (audio.shape[-1] / model.sample_rate) * 1000
    rtf = audio_duration_ms / elapsed_ms  # real-time factor (>1 means faster than real-time)

    return {
        "elapsed_ms": elapsed_ms,
        "ttfb_ms": ttfb_ms,
        "audio_duration_ms": audio_duration_ms,
        "rtf": rtf,
        "audio_samples": audio.shape[-1],
        "gpu_memory_mb": get_gpu_memory_mb(),
        "gpu_memory_peak_mb": get_gpu_memory_peak_mb(),
    }


def run_benchmark(
    device: str = "cuda",
    quantize: bool = False,
    runs: int = 5,
    warmup_runs: int = 2,
    texts: list[str] | None = None,
    voice: str = DEFAULT_AUDIO_PROMPT,
    config: str = DEFAULT_VARIANT,
    no_flash: bool = False,
    compile_model: bool = False,
) -> dict:
    """Run the full benchmark suite."""

    if texts is None:
        texts = BENCHMARK_TEXTS

    print("=" * 60)
    print("Pocket-TTS Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Quantize: {quantize}")
    print(f"Compile: {compile_model}")
    print(f"Flash Attention: {'disabled' if no_flash else 'enabled'}")
    print(f"Runs per text: {runs}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Texts to benchmark: {len(texts)}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    load_start = time.perf_counter()
    model = TTSModel.load_model(config=config, quantize=quantize)
    model.to(device)
    model.eval()
    if compile_model:
        print("Compiling model with torch.compile()...")
        model.flow_lm = torch.compile(model.flow_lm, mode="reduce-overhead")
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Load voice
    print("Loading voice state...")
    voice_start = time.perf_counter()
    model_state = model.get_state_for_audio_prompt(voice)
    model_state = move_state_to_device(model_state, device)
    voice_time = time.perf_counter() - voice_start
    print(f"Voice state loaded in {voice_time:.2f}s")

    if torch.cuda.is_available():
        print(f"GPU memory after load: {get_gpu_memory_mb():.1f} MB")

    results = {
        "config": {
            "device": device,
            "quantize": quantize,
            "runs": runs,
            "warmup_runs": warmup_runs,
            "flash_attention": not no_flash,
            "compile": compile_model,
        },
        "load_time_s": load_time,
        "voice_load_time_s": voice_time,
        "texts": [],
    }

    for text_idx, text in enumerate(texts):
        print(f"\n--- Text {text_idx + 1}/{len(texts)} ---")
        print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"Length: {len(text)} chars, {len(text.split())} words")

        text_results = {
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
            "runs": [],
        }

        # Warmup
        if warmup_runs > 0:
            print(f"Warming up ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                benchmark_single(model, model_state, text, warmup=True, no_flash=no_flash)

        # Benchmark runs
        print(f"Benchmarking ({runs} runs)...")
        for run_idx in range(runs):
            result = benchmark_single(model, model_state, text, no_flash=no_flash)
            text_results["runs"].append(result)
            print(f"  Run {run_idx + 1}: {result['elapsed_ms']:.1f}ms (TTFB: {result['ttfb_ms']:.1f}ms), RTF: {result['rtf']:.2f}x")

        # Calculate statistics
        elapsed_times = [r["elapsed_ms"] for r in text_results["runs"]]
        ttfb_times = [r["ttfb_ms"] for r in text_results["runs"]]
        rtfs = [r["rtf"] for r in text_results["runs"]]

        text_results["stats"] = {
            "elapsed_ms_mean": statistics.mean(elapsed_times),
            "elapsed_ms_std": statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0,
            "elapsed_ms_min": min(elapsed_times),
            "elapsed_ms_max": max(elapsed_times),
            "ttfb_ms_mean": statistics.mean(ttfb_times),
            "ttfb_ms_std": statistics.stdev(ttfb_times) if len(ttfb_times) > 1 else 0,
            "rtf_mean": statistics.mean(rtfs),
            "rtf_std": statistics.stdev(rtfs) if len(rtfs) > 1 else 0,
            "audio_duration_ms": text_results["runs"][0]["audio_duration_ms"],
        }

        print(f"  Mean: {text_results['stats']['elapsed_ms_mean']:.1f}ms (TTFB: {text_results['stats']['ttfb_ms_mean']:.1f}ms)")
        print(f"  RTF: {text_results['stats']['rtf_mean']:.2f}x (+/- {text_results['stats']['rtf_std']:.2f})")

        results["texts"].append(text_results)

    # Overall statistics
    all_rtfs = [r["rtf"] for t in results["texts"] for r in t["runs"]]
    all_elapsed = [r["elapsed_ms"] for t in results["texts"] for r in t["runs"]]
    all_ttfb = [r["ttfb_ms"] for t in results["texts"] for r in t["runs"]]

    results["overall"] = {
        "rtf_mean": statistics.mean(all_rtfs),
        "rtf_std": statistics.stdev(all_rtfs) if len(all_rtfs) > 1 else 0,
        "elapsed_ms_mean": statistics.mean(all_elapsed),
        "ttfb_ms_mean": statistics.mean(all_ttfb),
        "gpu_memory_peak_mb": get_gpu_memory_peak_mb(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall RTF: {results['overall']['rtf_mean']:.2f}x (+/- {results['overall']['rtf_std']:.2f})")
    print(f"Overall latency: {results['overall']['elapsed_ms_mean']:.1f}ms")
    print(f"Overall TTFB: {results['overall']['ttfb_ms_mean']:.1f}ms")
    if results['overall']['gpu_memory_peak_mb']:
        print(f"Peak GPU memory: {results['overall']['gpu_memory_peak_mb']:.1f} MB")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark pocket-tts inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--no-flash", action="store_true", help="Disable Flash Attention (use MATH backend)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() on flow_lm")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs per text")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--text", type=str, default=None, help="Custom text to benchmark (overrides default texts)")
    parser.add_argument("--voice", type=str, default=DEFAULT_AUDIO_PROMPT, help="Voice to use")
    parser.add_argument("--config", type=str, default=DEFAULT_VARIANT, help="Model config")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    texts = [args.text] if args.text else None

    results = run_benchmark(
        device=args.device,
        quantize=args.quantize,
        runs=args.runs,
        warmup_runs=args.warmup,
        texts=texts,
        voice=args.voice,
        config=args.config,
        no_flash=args.no_flash,
        compile_model=args.compile,
    )

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
