#!/usr/bin/env python
"""
Benchmark script for PyTorch threading configuration.

This script benchmarks TTS generation performance with different
thread counts to help determine the optimal configuration for your system.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Setup path to import pocket_tts
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    num_threads: int
    text: str
    generation_time: float
    real_time_factor: float


def benchmark_generation(
    model,
    text: str,
    num_threads: int,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark TTS generation with specific thread count.

    Args:
        model: TTS model instance
        text: Text to generate
        num_threads: Number of threads to use
        iterations: Number of iterations to run

    Returns:
        BenchmarkResult with average metrics
    """
    logger.info(f"\n=== Benchmarking with {num_threads} thread(s) ===")

    # Set thread count
    import torch

    torch.set_num_threads(num_threads)
    logger.info(f"PyTorch threads: {torch.get_num_threads()}")

    generation_times = []

    for i in range(iterations):
        logger.info(f"  Iteration {i + 1}/{iterations}")
        start_time = time.time()
        audio = model.generate(text=text, batch_size=1)
        generation_time = time.time() - start_time

        audio_duration_sec = audio.shape[-1] / model.sample_rate
        rtf = generation_time / audio_duration_sec

        generation_times.append(generation_time)
        logger.info(f"    Generation time: {generation_time:.2f}s")
        logger.info(f"    Real-time factor: {rtf:.2f}x")

    avg_time = sum(generation_times) / len(generation_times)
    audio_duration_sec = audio.shape[-1] / model.sample_rate
    avg_rtf = avg_time / audio_duration_sec

    result = BenchmarkResult(
        num_threads=num_threads,
        text=text,
        generation_time=avg_time,
        real_time_factor=avg_rtf,
    )

    logger.info(f"  Average generation time: {avg_time:.2f}s")
    logger.info(f"  Average real-time factor: {avg_rtf:.2f}x")

    return result


def main():
    """Main entry point for threading benchmark."""
    logger.info("Starting PyTorch threading benchmark for Pocket TTS")

    # Import model loading after setting up sys.path
    import torch

    from pocket_tts.models.tts_model import TTSModel
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary

    # Load configuration
    config = load_config()
    logger.info(f"Loaded config with variant: {config.variant}")

    # Download model if necessary
    download_if_necessary(config)

    # Test with different thread counts
    thread_counts = [1, 2, 4]
    test_texts = [
        ("short", "This is a short test."),
        (
            "medium",
            "This is a medium length text that contains more words for testing performance across different threading configurations.",
        ),
        (
            "long",
            "This is a much longer text designed to test the performance characteristics of different threading configurations. "
            "It contains significantly more words and should take longer to process, which helps illustrate the differences "
            "between single-threaded and multi-threaded execution. " * 3,
        ),
    ]

    results = []

    for text_name, text in test_texts:
        logger.info(f"\n### Benchmarking {text_name} text ###")

        # Load model fresh for each thread count to ensure fair comparison
        for num_threads in thread_counts:
            # Reload model to ensure clean state
            os.environ["POCKET_TTS_NUM_THREADS"] = str(num_threads)

            # Import and load model (this will use the new thread count)
            model = TTSModel.from_pydantic_config_with_weights(
                config,
                temp=0.7,
                lsd_decode_steps=4,
                noise_clamp=None,
                eos_threshold=0.5,
            )
            model.eval()

            result = benchmark_generation(model, text, num_threads, iterations=3)
            results.append(result)

            # Clean up model to free memory
            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    logger.info("\n### Summary ###")
    logger.info("Generation times by thread count:")

    for result in results:
        logger.info(
            f"  {result.num_threads} thread(s): {result.generation_time:.2f}s "
            f"(RTF: {result.real_time_factor:.2f}x)"
        )

    # Recommendations
    logger.info("\n### Recommendations ###")
    logger.info("Run this script on your system to determine the optimal thread count.")
    logger.info(
        "Set the POCKET_TTS_NUM_THREADS environment variable to use your preferred count:"
    )
    logger.info("  export POCKET_TTS_NUM_THREADS=2")
    logger.info("  pocket-tts generate 'Your text here'")


if __name__ == "__main__":
    main()
