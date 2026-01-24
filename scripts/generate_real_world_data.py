#!/usr/bin/env python3
"""
Generate comprehensive real-world test datasets for NumPy vs Rust validation.

This script creates diverse, realistic test datasets covering all NumPy use cases
to ensure comprehensive validation between Python NumPy and Rust implementations.

Usage:
    python scripts/generate_real_world_data.py [--categories all|array|mathematical|linalg|fft|statistics|edge_cases|real_world]
                                           [--sizes small|medium|large|all]
                                           [--output-dir test_data/]
                                           [--seed 42]
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal, stats, fft
import matplotlib.pyplot as plt


class TestDataGenerator:
    """Comprehensive test data generator for NumPy vs Rust validation."""

    def __init__(self, output_dir: Path, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        np.random.seed(seed)

        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)

        # Store metadata for each dataset
        self.datasets = {}

    def generate_all(self) -> None:
        """Generate all test dataset categories."""
        print("Generating comprehensive test dataset suite...")

        # Generate all categories
        self.generate_array_sizes()
        self.generate_dtype_coverage()
        self.generate_mathematical_patterns()
        self.generate_linear_algebra_data()
        self.generate_fft_data()
        self.generate_statistical_data()
        self.generate_edge_cases()
        self.generate_real_world_data()

        # Generate summary report
        self.generate_summary_report()

        print(f"✅ Generated {len(self.datasets)} datasets in {self.output_dir}")

    def generate_array_sizes(self) -> None:
        """Generate arrays with various sizes and dimensions."""
        print("📏 Generating array size coverage datasets...")

        base_dir = self.output_dir / "arrays"

        # Small arrays (1-10 elements)
        small_data = {}
        for size in [1, 3, 5, 7, 10]:
            arr = np.random.randn(size).astype(np.float32)
            small_data[f"size_{size}"] = arr

        self._save_dataset(
            small_data,
            base_dir / "small" / "float32_random.parquet",
            metadata={
                "description": "Small arrays with random values",
                "dtype": "float32",
            },
        )

        # Medium arrays (100-1000 elements)
        medium_data = {}
        for size in [100, 250, 500, 750, 1000]:
            arr = np.random.randn(size).astype(np.float32)
            medium_data[f"size_{size}"] = arr

        self._save_dataset(
            medium_data,
            base_dir / "medium" / "float32_random.parquet",
            metadata={
                "description": "Medium arrays with random values",
                "dtype": "float32",
            },
        )

        # Large arrays (10K-100K elements)
        large_data = {}
        for size in [10000, 25000, 50000, 75000, 100000]:
            arr = np.random.randn(size).astype(np.float32)
            large_data[f"size_{size}"] = arr

        self._save_dataset(
            large_data,
            base_dir / "large" / "float32_random.parquet",
            metadata={
                "description": "Large arrays with random values",
                "dtype": "float32",
            },
        )

        # Multi-dimensional arrays
        multidim_data = {}
        shapes = [
            (10,),
            (5, 4),
            (3, 3, 3),
            (2, 2, 2, 2),
            (2, 3, 4),
            (2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
        ]

        for i, shape in enumerate(shapes):
            arr = np.random.randn(*shape).astype(np.float32)
            multidim_data[f"dim_{i}_shape_{shape}"] = arr

        self._save_dataset(
            multidim_data,
            base_dir / "multidimensional" / "float32_random.parquet",
            metadata={
                "description": "Multi-dimensional arrays with various shapes",
                "dtype": "float32",
            },
        )

        # Special shapes
        special_data = {}
        special_data["empty"] = np.array([], dtype=np.float32)
        special_data["single_element"] = np.array([42.0], dtype=np.float32)
        special_data["zero_dim"] = np.array(42.0, dtype=np.float32)
        special_data["broadcast_1x5"] = np.ones((1, 5), dtype=np.float32)
        special_data["broadcast_5x1"] = np.ones((5, 1), dtype=np.float32)

        self._save_dataset(
            special_data,
            base_dir / "special_shapes" / "float32_special.parquet",
            metadata={
                "description": "Special shape arrays for edge case testing",
                "dtype": "float32",
            },
        )

    def generate_dtype_coverage(self) -> None:
        """Generate arrays with all supported NumPy dtypes."""
        print("🔢 Generating dtype coverage datasets...")

        base_dir = self.output_dir / "arrays"
        base_size = 1000

        # Generate test data for each dtype
        dtype_data = {}

        # Integer types
        for bits in [8, 16, 32, 64]:
            signed_dtype = f"int{bits}"
            unsigned_dtype = f"uint{bits}"

            # Generate appropriate range for each bit size
            if bits == 8:
                range_min, range_max = -128, 127
                urange_min, urange_max = 0, 255
            elif bits == 16:
                range_min, range_max = -32768, 32767
                urange_min, urange_max = 0, 65535
            elif bits == 32:
                range_min, range_max = -2147483648, 2147483647
                urange_min, urange_max = 0, 4294967295
            else:  # 64
                range_min, range_max = -9223372036854775808, 9223372036854775807
                urange_min, urange_max = 0, 18446744073709551615

            dtype_data[f"{signed_dtype}_random"] = np.random.randint(
                range_min, range_max + 1, base_size, dtype=signed_dtype
            )
            dtype_data[f"{unsigned_dtype}_random"] = np.random.randint(
                urange_min, urange_max + 1, base_size, dtype=unsigned_dtype
            )

        # Floating point types
        for dtype_name, dtype in [
            ("float16", np.float16),
            ("float32", np.float32),
            ("float64", np.float64),
        ]:
            dtype_data[f"{dtype_name}_random"] = np.random.randn(base_size).astype(
                dtype
            )
            dtype_data[f"{dtype_name}_uniform"] = np.random.uniform(
                -1.0, 1.0, base_size
            ).astype(dtype)

        # Complex types
        for dtype_name, dtype in [
            ("complex64", np.complex64),
            ("complex128", np.complex128),
        ]:
            real = np.random.randn(base_size).astype(
                np.float32 if dtype == np.complex64 else np.float64
            )
            imag = np.random.randn(base_size).astype(
                np.float32 if dtype == np.complex64 else np.float64
            )
            dtype_data[f"{dtype_name}_random"] = (real + 1j * imag).astype(dtype)

        # Boolean type only (skip object types to avoid Arrow issues)
        dtype_data["bool_random"] = np.random.choice([True, False], base_size)

        self._save_dataset(
            dtype_data,
            base_dir / "dtype_coverage.parquet",
            metadata={
                "description": "Coverage of all NumPy dtypes",
                "base_size": base_size,
            },
        )

    def generate_mathematical_patterns(self) -> None:
        """Generate arrays with various mathematical patterns."""
        print("🧮 Generating mathematical pattern datasets...")

        base_dir = self.output_dir / "mathematical"
        size = 1000
        x = np.linspace(-10, 10, size).astype(np.float32)

        # Basic operations patterns
        basic_ops = {}
        basic_ops["arange"] = np.arange(0, size, 1, dtype=np.float32)
        basic_ops["linspace"] = np.linspace(0, 100, size, dtype=np.float32)
        basic_ops["logspace"] = np.logspace(-3, 3, size, dtype=np.float32)
        basic_ops["geomspace"] = np.geomspace(1, 1000, size, dtype=np.float32)

        self._save_dataset(
            basic_ops,
            base_dir / "basic_ops" / "array_creation.parquet",
            metadata={"description": "Basic array creation patterns", "size": size},
        )

        # Trigonometric patterns
        trig_data = {}
        trig_data["sine_wave"] = np.sin(2 * np.pi * 0.1 * np.arange(size)).astype(
            np.float32
        )
        trig_data["cosine_wave"] = np.cos(2 * np.pi * 0.1 * np.arange(size)).astype(
            np.float32
        )
        trig_data["tangent_values"] = np.tan(x[::10]).astype(
            np.float32
        )  # Subsample to avoid asymptotes
        trig_data["arcsin_values"] = np.arcsin(
            np.linspace(-0.9, 0.9, size // 10)
        ).astype(np.float32)
        trig_data["arccos_values"] = np.arccos(
            np.linspace(-0.9, 0.9, size // 10)
        ).astype(np.float32)

        self._save_dataset(
            trig_data,
            base_dir / "trigonometric" / "waveforms.parquet",
            metadata={"description": "Trigonometric function patterns", "size": size},
        )

        # Exponential and logarithmic patterns
        exp_data = {}
        exp_data["exponential"] = np.exp(x / 10).astype(np.float32)
        exp_data["natural_log"] = np.log(np.abs(x) + 1).astype(np.float32)
        exp_data["log_base10"] = np.log10(np.abs(x) + 1).astype(np.float32)
        exp_data["power_2"] = np.power(x / 10, 2).astype(np.float32)
        exp_data["power_3"] = np.power(x / 10, 3).astype(np.float32)
        exp_data["square_root"] = np.sqrt(np.abs(x)).astype(np.float32)

        self._save_dataset(
            exp_data,
            base_dir / "exponential" / "functions.parquet",
            metadata={
                "description": "Exponential and logarithmic patterns",
                "size": size,
            },
        )

        # Statistical patterns
        stat_data = {}
        stat_data["normal_distribution"] = np.random.normal(0, 1, size).astype(
            np.float32
        )
        stat_data["uniform_distribution"] = np.random.uniform(-1, 1, size).astype(
            np.float32
        )
        stat_data["exponential_distribution"] = np.random.exponential(1, size).astype(
            np.float32
        )
        stat_data["poisson_samples"] = np.random.poisson(5, size).astype(np.float32)
        stat_data["binomial_samples"] = np.random.binomial(10, 0.5, size).astype(
            np.float32
        )

        self._save_dataset(
            stat_data,
            base_dir / "statistical" / "distributions.parquet",
            metadata={"description": "Statistical distributions", "size": size},
        )

    def generate_linear_algebra_data(self) -> None:
        """Generate linear algebra test matrices and vectors."""
        print("🔷 Generating linear algebra datasets...")

        base_dir = self.output_dir / "linalg"

        # Basic matrices
        matrix_data = {}

        # Identity matrices
        for n in [2, 3, 4, 5, 10]:
            matrix_data[f"identity_{n}x{n}"] = np.eye(n, dtype=np.float32)

        # Diagonal matrices
        for n in [3, 4, 5]:
            diagonal_values = np.arange(1, n + 1, dtype=np.float32)
            matrix_data[f"diagonal_{n}x{n}"] = np.diag(diagonal_values)

        # Symmetric matrices
        for n in [3, 4, 5]:
            A = np.random.randn(n, n).astype(np.float32)
            symmetric = (A + A.T) / 2
            matrix_data[f"symmetric_{n}x{n}"] = symmetric

        # Orthogonal matrices (using Gram-Schmidt process)
        for n in [3, 4]:
            A = np.random.randn(n, n).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            matrix_data[f"orthogonal_{n}x{n}"] = Q

        self._save_dataset(
            matrix_data,
            base_dir / "matrices" / "special_matrices.parquet",
            metadata={
                "description": "Special matrix types (identity, diagonal, symmetric, orthogonal)"
            },
        )

        # Vector operations
        vector_data = {}
        for size in [3, 4, 10, 100]:
            vector_data[f"vector_{size}d"] = np.random.randn(size).astype(np.float32)

        # Orthogonal vectors
        v1 = np.array([1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0], dtype=np.float32)
        v3 = np.array([0, 0, 1], dtype=np.float32)
        vector_data["orthogonal_basis_3d"] = np.stack([v1, v2, v3])

        self._save_dataset(
            vector_data,
            base_dir / "vectors" / "vectors.parquet",
            metadata={"description": "Vector data for dot products and operations"},
        )

    def generate_fft_data(self) -> None:
        """Generate FFT and signal processing test data."""
        print("🌊 Generating FFT signal datasets...")

        base_dir = self.output_dir / "fft"
        sample_rate = 1000
        duration = 2.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Various signal patterns
        signal_data = {}

        # Pure sine waves at different frequencies
        frequencies = [10, 50, 100, 250]  # Hz
        for freq in frequencies:
            sine_wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
            signal_data[f"sine_{freq}hz"] = sine_wave

        # Composite signals
        composite = (
            np.sin(2 * np.pi * 50 * t)
            + 0.5 * np.sin(2 * np.pi * 120 * t)
            + 0.3 * np.sin(2 * np.pi * 200 * t)
        ).astype(np.float32)
        signal_data["composite_multiple_freq"] = composite

        # Chirp signal (frequency sweep)
        chirp = signal.chirp(t, f0=10, f1=400, t1=duration, method="linear").astype(
            np.float32
        )
        signal_data["chirp_sweep"] = chirp

        # Square wave
        square = signal.square(2 * np.pi * 50 * t).astype(np.float32)
        signal_data["square_50hz"] = square

        # Sawtooth wave
        sawtooth = signal.sawtooth(2 * np.pi * 50 * t).astype(np.float32)
        signal_data["sawtooth_50hz"] = sawtooth

        # Noise patterns
        noise_data = {}
        noise_data["gaussian_noise"] = np.random.randn(len(t)).astype(np.float32)
        noise_data["uniform_noise"] = np.random.uniform(-1, 1, len(t)).astype(
            np.float32
        )
        noise_data["salt_pepper"] = np.random.choice(
            [-1, 0, 1], len(t), p=[0.05, 0.9, 0.05]
        ).astype(np.float32)

        self._save_dataset(
            signal_data,
            base_dir / "signals" / "waveforms.parquet",
            metadata={
                "description": "Various signal waveforms for FFT testing",
                "sample_rate": sample_rate,
                "duration": duration,
            },
        )

        self._save_dataset(
            noise_data,
            base_dir / "signals" / "noise_patterns.parquet",
            metadata={
                "description": "Noise patterns for signal processing",
                "sample_rate": sample_rate,
            },
        )

    def generate_statistical_data(self) -> None:
        """Generate statistical distribution and sampling data."""
        print("📊 Generating statistical datasets...")

        base_dir = self.output_dir / "statistics"

        # Distribution samples
        dist_data = {}

        # Large samples from various distributions
        sample_size = 10000

        # Normal distributions with different parameters
        for mean, std in [(0, 1), (5, 2), (-3, 0.5), (10, 5)]:
            samples = np.random.normal(mean, std, sample_size).astype(np.float32)
            dist_data[f"normal_mean{mean}_std{std}"] = samples

        # Other distributions
        dist_data["uniform_neg1_to_1"] = np.random.uniform(-1, 1, sample_size).astype(
            np.float32
        )
        dist_data["exponential_lambda1"] = np.random.exponential(1, sample_size).astype(
            np.float32
        )
        dist_data["gamma_shape2_scale2"] = np.random.gamma(2, 2, sample_size).astype(
            np.float32
        )
        dist_data["beta_2_5"] = np.random.beta(2, 5, sample_size).astype(np.float32)

        self._save_dataset(
            dist_data,
            base_dir / "distributions" / "large_samples.parquet",
            metadata={
                "description": "Large samples from statistical distributions",
                "sample_size": sample_size,
            },
        )

        # Time series with trends and seasonality
        ts_data = {}
        n_points = 1000

        # Linear trend
        trend = np.linspace(0, 100, n_points).astype(np.float32)
        noise = np.random.randn(n_points) * 2
        ts_data["linear_trend_noise"] = trend + noise

        # Seasonal pattern
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 50).astype(np.float32)
        ts_data["seasonal_pattern"] = seasonal

        # Trend + seasonal + noise
        ts_data["trend_seasonal_noise"] = trend + seasonal + noise

        # Random walk
        random_walk = np.cumsum(np.random.randn(n_points) * 0.1).astype(np.float32)
        ts_data["random_walk"] = random_walk

        self._save_dataset(
            ts_data,
            base_dir / "sampling" / "time_series.parquet",
            metadata={
                "description": "Time series patterns for statistical testing",
                "n_points": n_points,
            },
        )

    def generate_edge_cases(self) -> None:
        """Generate edge case and boundary condition data."""
        print("⚠️ Generating edge case datasets...")

        base_dir = self.output_dir / "edge_cases"

        # NaN and Inf handling
        nan_inf_data = {}

        # Arrays with NaN values
        arr_nan = np.array([1.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float32)
        nan_inf_data["mixed_nan"] = arr_nan

        # All NaN array
        nan_inf_data["all_nan"] = np.full(10, np.nan, dtype=np.float32)

        # Arrays with Inf values
        arr_inf = np.array([1.0, np.inf, -np.inf, 4.0, np.inf], dtype=np.float32)
        nan_inf_data["mixed_inf"] = arr_inf

        # Mixed NaN and Inf
        nan_inf_data["mixed_nan_inf"] = np.array(
            [1.0, np.nan, np.inf, -np.inf, np.nan], dtype=np.float32
        )

        self._save_dataset(
            nan_inf_data,
            base_dir / "nan_inf" / "special_values.parquet",
            metadata={
                "description": "Arrays with NaN and Inf values for edge case testing"
            },
        )

        # Near overflow/underflow
        overflow_data = {}

        # Very large and small values
        overflow_data["very_large"] = np.array([1e35, 1e36, 1e37], dtype=np.float32)
        overflow_data["very_small"] = np.array([1e-35, 1e-36, 1e-37], dtype=np.float32)
        overflow_data["near_overflow"] = np.finfo(np.float32).max * np.array(
            [0.9, 0.99, 1.0], dtype=np.float32
        )
        overflow_data["near_underflow"] = np.finfo(np.float32).min * np.array(
            [0.9, 0.99, 1.0], dtype=np.float32
        )

        self._save_dataset(
            overflow_data,
            base_dir / "overflow" / "extreme_values.parquet",
            metadata={"description": "Values near overflow and underflow limits"},
        )

        # Precision edge cases
        precision_data = {}

        # Machine epsilon tests
        eps = np.finfo(np.float32).eps
        precision_data["machine_epsilon"] = np.array(
            [1.0, 1.0 + eps, 1.0 + 2 * eps], dtype=np.float32
        )
        precision_data["tiny_differences"] = np.array(
            [1.0, 1.0000001, 1.0000002], dtype=np.float32
        )
        precision_data["very_small_differences"] = np.array(
            [1.0, 1.0 + 1e-7, 1.0 + 2e-7], dtype=np.float32
        )

        self._save_dataset(
            precision_data,
            base_dir / "precision" / "epsilon_tests.parquet",
            metadata={
                "description": "Precision edge cases and machine epsilon testing"
            },
        )

        # Near-singular matrices
        singular_data = {}

        # Nearly singular matrices
        for n in [2, 3, 4]:
            A = np.random.randn(n, n).astype(np.float32)
            # Make one row nearly dependent on others
            A[-1] = A[0] + 1e-6 * np.random.randn(n)
            singular_data[f"nearly_singular_{n}x{n}"] = A

        self._save_dataset(
            singular_data,
            base_dir / "singular" / "nearly_singular.parquet",
            metadata={
                "description": "Nearly singular matrices for numerical stability testing"
            },
        )

    def generate_real_world_data(self) -> None:
        """Generate application-specific real-world datasets."""
        print("🌍 Generating real-world application datasets...")

        base_dir = self.output_dir / "real_world"

        # Audio processing data
        audio_data = {}

        # Different sample rates and formats
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        duration = 1.0  # seconds

        for sr in sample_rates:
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

            # Speech-like frequency range (100-4000 Hz)
            speech_signal = np.sum(
                [
                    np.sin(2 * np.pi * freq * t) * amplitude
                    for freq, amplitude in [
                        (200, 0.3),
                        (500, 0.2),
                        (1000, 0.15),
                        (2000, 0.1),
                    ]
                ],
                axis=0,
            ).astype(np.float32)

            audio_data[f"speech_{sr}hz"] = speech_signal

        # Music-like signals
        sr_music = 44100
        t_music = np.linspace(0, 0.1, int(sr_music * 0.1), dtype=np.float32)

        # Musical notes (A4 = 440 Hz, C5 = 523 Hz, E5 = 659 Hz)
        note_a4 = np.sin(2 * np.pi * 440 * t_music) * 0.3
        note_c5 = np.sin(2 * np.pi * 523 * t_music) * 0.2
        note_e5 = np.sin(2 * np.pi * 659 * t_music) * 0.25

        audio_data["musical_chord"] = (note_a4 + note_c5 + note_e5).astype(np.float32)

        self._save_dataset(
            audio_data,
            base_dir / "audio" / "processing_samples.parquet",
            metadata={
                "description": "Audio processing samples at various sample rates",
                "duration": duration,
            },
        )

        # Image processing data
        image_data = {}

        # Different image patterns
        sizes = [(64, 64), (128, 128), (256, 256)]

        for height, width in sizes:
            # Gradient
            x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            gradient = (x + y) / 2.0
            image_data[f"gradient_{height}x{width}"] = gradient.astype(np.float32)

            # Checkerboard pattern
            checker = ((x * 8).astype(int) + (y * 8).astype(int)) % 2
            image_data[f"checkerboard_{height}x{width}"] = checker.astype(np.float32)

            # Radial gradient
            center_x, center_y = width // 2, height // 2
            xx, yy = np.meshgrid(
                np.arange(width) - center_x, np.arange(height) - center_y
            )
            radial = np.sqrt(xx**2 + yy**2)
            radial_normalized = radial / np.max(radial)
            image_data[f"radial_{height}x{width}"] = radial_normalized.astype(
                np.float32
            )

        # RGB image data
        rgb_height, rgb_width = 128, 128
        rgb_image = np.random.randint(
            0, 256, (rgb_height, rgb_width, 3), dtype=np.uint8
        )
        image_data["rgb_random_128x128x3"] = rgb_image

        # Grayscale image
        gray_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        image_data["grayscale_random_256x256"] = gray_image

        self._save_dataset(
            image_data,
            base_dir / "image" / "processing_patterns.parquet",
            metadata={"description": "Image processing patterns and samples"},
        )

        # Financial time series
        financial_data = {}

        # Stock price simulation
        n_days = 252  # Trading days in a year
        initial_price = 100.0

        # Geometric Brownian motion
        dt = 1 / 252  # Daily steps
        mu, sigma = 0.1, 0.2  # 10% annual return, 20% volatility

        returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days
        )
        price_path = initial_price * np.exp(np.cumsum(returns))

        financial_data["stock_price_simulation"] = price_path.astype(np.float32)
        financial_data["daily_returns"] = returns.astype(np.float32)

        # Volatility clustering (GARCH-like)
        volatility = np.ones(n_days)
        for i in range(10, n_days):
            volatility[i] = 0.1 * volatility[i - 1] + 0.9 * (returns[i - 1] ** 2)

        clustered_returns = np.random.normal(0, np.sqrt(volatility)) * 0.02
        financial_data["volatility_clustered_returns"] = clustered_returns.astype(
            np.float32
        )

        self._save_dataset(
            financial_data,
            base_dir / "financial" / "time_series.parquet",
            metadata={
                "description": "Financial time series with volatility patterns",
                "n_days": n_days,
            },
        )

        # Scientific measurement data
        scientific_data = {}

        # Sensor measurements with noise
        time_points = 1000
        t_scientific = np.linspace(0, 10, time_points).astype(np.float32)

        # Temperature sensor with drift and noise
        true_temperature = (
            20 + 5 * np.sin(2 * np.pi * 0.1 * t_scientific) + 0.1 * t_scientific
        )
        sensor_noise = np.random.normal(0, 0.5, time_points)
        measured_temperature = true_temperature + sensor_noise

        scientific_data["temperature_sensor"] = measured_temperature.astype(np.float32)
        scientific_data["true_temperature"] = true_temperature.astype(np.float32)

        # Chemical concentration decay
        concentration_0 = 100.0  # Initial concentration
        decay_rate = 0.1
        concentration = concentration_0 * np.exp(-decay_rate * t_scientific)

        scientific_data["chemical_decay"] = concentration.astype(np.float32)

        # Physics experiment data (pendulum motion)
        pendulum_length = 1.0  # meters
        gravity = 9.81  # m/s^2
        omega = np.sqrt(gravity / pendulum_length)

        # Small angle approximation: θ(t) = θ₀ * cos(ωt)
        initial_angle = 0.1  # radians
        pendulum_angle = initial_angle * np.cos(omega * t_scientific)

        scientific_data["pendulum_motion"] = pendulum_angle.astype(np.float32)

        self._save_dataset(
            scientific_data,
            base_dir / "scientific" / "measurement_data.parquet",
            metadata={
                "description": "Scientific measurement data from various experiments"
            },
        )

    def _save_dataset(
        self, data: Dict[str, np.ndarray], filepath: Path, metadata: Dict[str, Any]
    ) -> None:
        """Save dataset in multiple formats with metadata."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save as Parquet (primary format)
        parquet_path = filepath

        # Save as separate Parquet files for each array to handle different shapes
        # Filter out arrays that Arrow can't handle
        filtered_data = {}
        for name, arr in data.items():
            # Skip problematic dtypes for Arrow compatibility
            if arr.dtype == np.object_ or (
                hasattr(arr, "dtype") and str(arr.dtype) == "object"
            ):
                print(f"Skipping {name} due to incompatible dtype: {arr.dtype}")
                continue
            elif arr.dtype == np.bool_:
                # Convert bool to int for Arrow compatibility
                filtered_data[name] = arr.astype(np.int16)
            else:
                filtered_data[name] = arr

        if not filtered_data:
            print(f"No compatible arrays to save for {parquet_path}")
            return

        if len(filtered_data) == 1:
            # Single array - save directly
            name, arr = next(iter(filtered_data.items()))
            flat_array = arr.flatten()
            table = pa.table({name: pa.array(flat_array)})
            pq.write_table(table, parquet_path)
        else:
            # Multiple arrays - save each as separate dataset with naming
            for name, arr in filtered_data.items():
                flat_array = arr.flatten()
                table = pa.table({name: pa.array(flat_array)})

                # Create filename with array name
                single_path = (
                    parquet_path.parent
                    / f"{parquet_path.stem}_{name}{parquet_path.suffix}"
                )
                pq.write_table(table, single_path)

        # Save as NumPy files (secondary format)
        npy_dir = filepath.parent / (filepath.stem + "_npy")
        npy_dir.mkdir(exist_ok=True)
        for name, arr in data.items():
            np.save(npy_dir / f"{name}.npy", arr)

        # Save metadata
        metadata_with_shapes = {
            **metadata,
            "arrays": {
                name: {"shape": arr.shape, "dtype": str(arr.dtype), "size": arr.size}
                for name, arr in data.items()
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator_version": "1.0.0",
        }

        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_with_shapes, f, indent=2, default=str)

        # Store for summary report
        self.datasets[str(filepath.relative_to(self.output_dir))] = metadata_with_shapes

    def generate_summary_report(self) -> None:
        """Generate a summary report of all generated datasets."""
        report = {
            "generation_summary": {
                "total_datasets": len(self.datasets),
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": self.seed,
                "generator_version": "1.0.0",
            },
            "datasets": self.datasets,
        }

        report_path = self.output_dir / "generation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate human-readable summary
        summary_path = self.output_dir / "SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write("# Test Data Generation Summary\n\n")
            f.write(
                f"Generated **{len(self.datasets)} datasets** on {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
            )
            f.write(f"Random seed: {self.seed}\n\n")
            f.write("## Dataset Categories\n\n")

            categories = {}
            for dataset_path, metadata in self.datasets.items():
                category = dataset_path.split("/")[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(dataset_path)

            for category, datasets in categories.items():
                f.write(f"### {category.title()}\n")
                for dataset in datasets:
                    f.write(f"- `{dataset}`\n")
                f.write("\n")

        print(f"📊 Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test datasets for NumPy vs Rust validation"
    )
    parser.add_argument(
        "--categories",
        choices=[
            "all",
            "array",
            "mathematical",
            "linalg",
            "fft",
            "statistics",
            "edge_cases",
            "real_world",
        ],
        default="all",
        help="Dataset categories to generate (default: all)",
    )
    parser.add_argument(
        "--sizes",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Array size categories to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_data",
        help="Output directory for generated datasets (default: test_data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )

    args = parser.parse_args()

    generator = TestDataGenerator(Path(args.output_dir), args.seed)

    if args.categories == "all":
        generator.generate_all()
    elif args.categories == "array":
        generator.generate_array_sizes()
        generator.generate_dtype_coverage()
    elif args.categories == "mathematical":
        generator.generate_mathematical_patterns()
    elif args.categories == "linalg":
        generator.generate_linear_algebra_data()
    elif args.categories == "fft":
        generator.generate_fft_data()
    elif args.categories == "statistics":
        generator.generate_statistical_data()
    elif args.categories == "edge_cases":
        generator.generate_edge_cases()
    elif args.categories == "real_world":
        generator.generate_real_world_data()

    print("✅ Test data generation completed!")


if __name__ == "__main__":
    main()
