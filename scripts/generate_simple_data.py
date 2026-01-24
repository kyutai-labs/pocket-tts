#!/usr/bin/env python3
"""
Simplified test data generator for NumPy vs Rust validation.

This version focuses on core functionality with reliable cross-language compatibility.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class SimpleDataGenerator:
    """Simplified test data generator focusing on core functionality."""

    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        np.random.seed(seed)
        self.datasets = {}

    def generate_all(self) -> None:
        """Generate all essential test datasets."""
        print("🔄 Generating essential test datasets...")

        # Core array datasets
        self.generate_array_size_tests()
        self.generate_mathematical_tests()
        self.generate_edge_case_tests()
        self.generate_real_world_tests()

        # Generate summary
        self.generate_summary()
        print(f"✅ Generated {len(self.datasets)} dataset files")

    def generate_array_size_tests(self) -> None:
        """Generate arrays with different sizes and shapes."""
        print("📏 Generating array size tests...")

        # Different sizes
        sizes = [10, 100, 1000]
        for i, size in enumerate(sizes):
            arr = np.random.randn(size).astype(np.float32)
            self._save_single_array(
                f"size_{size}", arr, f"arrays/size_tests_{size}.parquet"
            )

        # Multi-dimensional arrays
        shapes = [(5, 4), (3, 3, 3), (2, 2, 2, 2)]
        for i, shape in enumerate(shapes):
            arr = np.random.randn(*shape).astype(np.float32)
            self._save_single_array(
                f"multidim_{i}_shape_{shape}", arr, f"arrays/multidim_{i}.parquet"
            )

    def generate_mathematical_tests(self) -> None:
        """Generate mathematical pattern datasets."""
        print("🧮 Generating mathematical tests...")

        # Trigonometric patterns
        x = np.linspace(0, 10, 100).astype(np.float32)

        patterns = {
            "sine_wave": np.sin(2 * np.pi * 0.5 * x),
            "cosine_wave": np.cos(2 * np.pi * 0.3 * x),
            "exponential": np.exp(x / 5),
            "logarithmic": np.log(x + 1),
            "random_normal": np.random.randn(1000).astype(np.float32),
            "random_uniform": np.random.uniform(-1, 1, 1000).astype(np.float32),
        }

        for name, arr in patterns.items():
            self._save_single_array(name, arr, f"mathematical/{name}.parquet")

    def generate_edge_case_tests(self) -> None:
        """Generate edge case and boundary condition data."""
        print("⚠️ Generating edge case tests...")

        # Edge cases
        edge_cases = {
            "with_nan": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
            "with_inf": np.array([1.0, np.inf, -np.inf, 4.0], dtype=np.float32),
            "very_large": np.array([1e10, 1e15, 1e20], dtype=np.float32),
            "very_small": np.array([1e-10, 1e-15, 1e-20], dtype=np.float32),
            "near_zero": np.array([1e-7, 1e-8, 1e-9], dtype=np.float32),
        }

        for name, arr in edge_cases.items():
            self._save_single_array(name, arr, f"edge_cases/{name}.parquet")

    def generate_real_world_tests(self) -> None:
        """Generate real-world application datasets."""
        print("🌍 Generating real-world tests...")

        # Audio-like signals
        sample_rate = 1000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Speech-like frequencies
        audio_patterns = {
            "speech_signal": (
                np.sin(2 * np.pi * 200 * t) * 0.3
                + np.sin(2 * np.pi * 500 * t) * 0.2
                + np.sin(2 * np.pi * 1000 * t) * 0.1
            ).astype(np.float32),
            "musical_note": np.sin(2 * np.pi * 440 * t).astype(np.float32),  # A4 note
            "noise_signal": np.random.randn(len(t)).astype(np.float32),
        }

        for name, arr in audio_patterns.items():
            self._save_single_array(name, arr, f"real_world/audio_{name}.parquet")

        # Time series patterns
        ts_patterns = {
            "linear_trend": np.linspace(0, 100, 252).astype(np.float32)
            + np.random.randn(252) * 2,
            "seasonal_pattern": 10
            * np.sin(2 * np.pi * np.arange(252) / 50).astype(np.float32),
            "random_walk": np.cumsum(np.random.randn(252) * 0.1).astype(np.float32),
        }

        for name, arr in ts_patterns.items():
            self._save_single_array(name, arr, f"real_world/timeseries_{name}.parquet")

    def _save_single_array(self, name: str, arr: np.ndarray, rel_path: str) -> None:
        """Save a single array dataset with metadata."""
        output_path = self.output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as Parquet (flatten for Arrow compatibility)
        flat_arr = arr.flatten()
        table = pa.table({"data": pa.array(flat_arr)})
        pq.write_table(table, output_path)

        # Save as NumPy
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, arr)

        # Save metadata
        metadata = {
            "name": name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "size": arr.size,
            "description": f"Test data for {name}",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
        }

        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.datasets[rel_path] = metadata

    def generate_summary(self) -> None:
        """Generate summary report."""
        report = {
            "generation_info": {
                "total_datasets": len(self.datasets),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": self.seed,
                "generator_version": "1.0.0",
            },
            "datasets": self.datasets,
        }

        report_path = self.output_dir / "generation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Human-readable summary
        summary_path = self.output_dir / "SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write("# Test Data Generation Summary\\n\\n")
            f.write(
                f"Generated **{len(self.datasets)} datasets** for NumPy vs Rust validation.\\n\\n"
            )
            f.write("## Generated Datasets\\n\\n")
            for path in sorted(self.datasets.keys()):
                metadata = self.datasets[path]
                f.write(f"- **{metadata['name']}** - {metadata['description']}\\n")
                f.write(f"  - Shape: {metadata['shape']}, Type: {metadata['dtype']}\\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate essential test datasets for NumPy vs Rust validation"
    )
    parser.add_argument("--output-dir", default="test_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = SimpleDataGenerator(args.output_dir, args.seed)
    generator.generate_all()


if __name__ == "__main__":
    main()
