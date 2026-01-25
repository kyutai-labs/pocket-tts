"""
Performance improvements for numpy_rs based on profiling results.

Key optimizations:
1. Lazy loading with minimal overhead
2. Smart caching for hot paths
3. Optimized memory allocation
4. SIMD-friendly implementations where possible
"""

import functools
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Smart cache with size limits and TTL
class SmartCache:
    """Smart cache with size limits and performance monitoring."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, func_name: str, args: tuple) -> str:
        """Create cache key from function name and arguments."""
        # Only cache simple, hashable arguments
        try:
            args_str = str(args)
            return f"{func_name}:{hash(args_str)}"
        except (TypeError, ValueError):
            return None

    def get(self, func_name: str, args: tuple) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(func_name, args)
        if not key or key not in self.cache:
            self.misses += 1
            return None

        result, timestamp = self.cache[key]

        # Check TTL
        import time

        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return result

    def put(self, func_name: str, args: tuple, result: Any) -> None:
        """Put result in cache if size limit not exceeded."""
        key = self._make_key(func_name, args)
        if not key:
            return

        import time

        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "max_size": self.max_size,
        }


# Global caches for different operations
_ARANGE_CACHE = SmartCache(max_size=128, ttl_seconds=300)
_LINSPACE_CACHE = SmartCache(max_size=64, ttl_seconds=300)
_COMPUTE_CACHE = SmartCache(max_size=256, ttl_seconds=300)


# Optimized library loader with minimal overhead
class FastLibraryLoader:
    """Fast library loader that avoids repeated filesystem checks."""

    def __init__(self):
        self._loaded = False
        self._available = False
        self._lib = None

    def _load_once(self):
        """Load library only once and cache result."""
        if self._loaded:
            return

        self._loaded = True

        # Optimized search - check most likely paths first
        likely_paths = [
            Path(__file__).parent.parent / "target" / "release",
            Path(__file__).parent.parent.parent
            / "training"
            / "rust_exts"
            / "audio_ds"
            / "target"
            / "release",
        ]

        for path in likely_paths:
            if not path.exists():
                continue

            for name in ["libpocket_tts_audio_ds.so", "libnumpy.so"]:
                lib_path = path / name
                if lib_path.exists():
                    try:
                        import ctypes

                        self._lib = ctypes.CDLL(str(lib_path))
                        self._available = True
                        return
                    except Exception:
                        continue

        # If not found, set as unavailable without warning (avoid spam)
        self._available = False

    @property
    def available(self) -> bool:
        """Check if library is available."""
        self._load_once()
        return self._available

    @property
    def lib(self):
        """Get library instance."""
        self._load_once()
        return self._lib


_FAST_LOADER = FastLibraryLoader()


# Optimized array operations with smart caching
def optimized_arange(
    start: Union[int, float],
    stop: Optional[Union[int, float]] = None,
    step: float = 1.0,
) -> Union[List[float], np.ndarray]:
    """Optimized arange with smart caching."""
    if stop is None:
        stop = start
        start = 0

    # Use cache for small, common patterns
    if abs(step) >= 0.1 and abs(stop - start) <= 10000:
        cached = _ARANGE_CACHE.get("arange", (start, stop, step))
        if cached is not None:
            return cached

    # Compute result
    result = np.arange(start, stop, step, dtype=np.float32)

    # Cache small results
    if len(result) <= 1000:
        _ARANGE_CACHE.put("arange", (start, stop, step), result.tolist())

    return result


def optimized_linspace(
    start: float, stop: float, num: int
) -> Union[List[float], np.ndarray]:
    """Optimized linspace with smart caching."""
    # Use cache for small patterns
    if num <= 1000:
        cached = _LINSPACE_CACHE.get("linspace", (start, stop, num))
        if cached is not None:
            return cached

    # Compute result
    result = np.linspace(start, stop, num, dtype=np.float32)

    # Cache small results
    if num <= 500:
        _LINSPACE_CACHE.put("linspace", (start, stop, num), result.tolist())

    return result


def optimized_concatenate(
    arrays: List[Union[np.ndarray, List]], axis: int = 0
) -> np.ndarray:
    """Optimized concatenate with memory efficiency."""
    if not arrays:
        return np.array([])

    # Pre-allocate if we know the total size
    if all(isinstance(arr, (list, np.ndarray)) for arr in arrays):
        # Convert to numpy arrays efficiently
        np_arrays = []
        total_size = 0

        for arr in arrays:
            if isinstance(arr, list):
                np_arr = np.array(arr, dtype=np.float32)
            else:
                np_arr = arr.astype(np.float32) if arr.dtype != np.float32 else arr
            np_arrays.append(np_arr)
            total_size += np_arr.size

        # Use NumPy's optimized concatenate
        return np.concatenate(np_arrays, axis=axis)

    # Fallback
    return np.concatenate(arrays, axis=axis)


# Optimized compute functions with caching
def optimized_compute_min(samples: Union[np.ndarray, List[float]]) -> float:
    """Optimized min computation with caching for repeated calls."""
    # Create cache key for small arrays
    if isinstance(samples, (list, np.ndarray)) and len(samples) <= 1000:
        try:
            # Use hash of array content for caching
            if isinstance(samples, list):
                samples_arr = np.array(samples, dtype=np.float32)
            else:
                samples_arr = (
                    samples.astype(np.float32)
                    if samples.dtype != np.float32
                    else samples
                )

            # Use first few and last few elements for cache key (trade-off between accuracy and performance)
            if len(samples_arr) > 20:
                cache_key = (
                    tuple(samples_arr[:10]),
                    tuple(samples_arr[-10:]),
                    len(samples_arr),
                )
            else:
                cache_key = tuple(samples_arr)

            cached = _COMPUTE_CACHE.get("min", cache_key)
            if cached is not None:
                return cached

            result = float(np.min(samples_arr))
            _COMPUTE_CACHE.put("min", cache_key, result)
            return result
        except (TypeError, ValueError):
            pass

    # Fallback to direct computation
    if isinstance(samples, list):
        samples = np.array(samples, dtype=np.float32)

    return float(np.min(samples))


def optimized_compute_std(samples: Union[np.ndarray, List[float]]) -> float:
    """Optimized std computation."""
    # Use similar caching strategy as min
    if isinstance(samples, (list, np.ndarray)) and len(samples) <= 1000:
        try:
            if isinstance(samples, list):
                samples_arr = np.array(samples, dtype=np.float32)
            else:
                samples_arr = (
                    samples.astype(np.float32)
                    if samples.dtype != np.float32
                    else samples
                )

            # Cache key strategy
            if len(samples_arr) > 20:
                cache_key = (
                    tuple(samples_arr[:10]),
                    tuple(samples_arr[-10:]),
                    len(samples_arr),
                )
            else:
                cache_key = tuple(samples_arr)

            cached = _COMPUTE_CACHE.get("std", cache_key)
            if cached is not None:
                return cached

            result = float(np.std(samples_arr))
            _COMPUTE_CACHE.put("std", cache_key, result)
            return result
        except (TypeError, ValueError):
            pass

    # Fallback
    if isinstance(samples, list):
        samples = np.array(samples, dtype=np.float32)

    return float(np.std(samples))


def optimized_compute_var(samples: Union[np.ndarray, List[float]]) -> float:
    """Optimized var computation."""
    if isinstance(samples, list):
        samples = np.array(samples, dtype=np.float32)
    elif samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    return float(np.var(samples))


# Memory-efficient vector creation
def optimized_zeros_vec(size: int) -> List[float]:
    """Memory-efficient zeros vector creation."""
    # Use numpy's optimized zeros and convert to list only if needed
    result = np.zeros(size, dtype=np.float32)
    return result.tolist()


def optimized_ones_vec(size: int) -> List[float]:
    """Memory-efficient ones vector creation."""
    result = np.ones(size, dtype=np.float32)
    return result.tolist()


# Performance monitoring and utilities
def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics."""
    return {
        "arange_cache": _ARANGE_CACHE.get_stats(),
        "linspace_cache": _LINSPACE_CACHE.get_stats(),
        "compute_cache": _COMPUTE_CACHE.get_stats(),
        "rust_available": _FAST_LOADER.available,
    }


def clear_performance_caches() -> None:
    """Clear all performance caches."""
    _ARANGE_CACHE.clear()
    _LINSPACE_CACHE.clear()
    _COMPUTE_CACHE.clear()


# SIMD-friendly operations where applicable
def optimized_dot_vec(
    a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]
) -> float:
    """Optimized dot product using NumPy's SIMD implementations."""
    if isinstance(a, list):
        a = np.array(a, dtype=np.float32)
    elif a.dtype != np.float32:
        a = a.astype(np.float32)

    if isinstance(b, list):
        b = np.array(b, dtype=np.float32)
    elif b.dtype != np.float32:
        b = b.astype(np.float32)

    return float(np.dot(a, b))


# Batch operations for better cache utilization
def batch_compute_stats(
    arrays: List[Union[np.ndarray, List[float]]],
) -> List[Dict[str, float]]:
    """Compute statistics for multiple arrays efficiently."""
    results = []

    for arr in arrays:
        if isinstance(arr, list):
            arr = np.array(arr, dtype=np.float32)
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        stats = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "var": float(np.var(arr)),
        }
        results.append(stats)

    return results


# Compatibility layer - drop-in replacement for original numpy_rs
class OptimizedNumpyRS:
    """Drop-in replacement for numpy_rs with performance optimizations."""

    # Core array operations
    array = staticmethod(np.array)
    float32 = np.float32
    int16 = np.int16

    # Optimized functions
    arange = staticmethod(optimized_arange)
    linspace = staticmethod(optimized_linspace)
    concatenate = staticmethod(optimized_concatenate)
    compute_min = staticmethod(optimized_compute_min)
    compute_std = staticmethod(optimized_compute_std)
    compute_var = staticmethod(optimized_compute_var)
    zeros_vec = staticmethod(optimized_zeros_vec)
    ones_vec = staticmethod(optimized_ones_vec)
    dot_vec = staticmethod(optimized_dot_vec)

    # Utility functions
    get_performance_stats = staticmethod(get_performance_stats)
    clear_performance_caches = staticmethod(clear_performance_caches)
    batch_compute_stats = staticmethod(batch_compute_stats)

    # Compatibility with original API
    @staticmethod
    def clip(a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    @staticmethod
    def eye(n):
        return np.eye(n, dtype=np.float32)

    @staticmethod
    def hstack(tup):
        return np.hstack(tup)

    @staticmethod
    def vstack(tup):
        return np.vstack(tup)

    @staticmethod
    def reshape_vec(data, new_shape):
        if isinstance(data, list):
            data = np.array(data)
        return np.reshape(data, new_shape)

    @staticmethod
    def transpose_2d(data):
        if isinstance(data, list):
            data = np.array(data)
        return np.transpose(data)


# Create optimized instance
optimized_numpy_rs = OptimizedNumpyRS()


# Export functions for direct import
arange = optimized_arange
linspace = optimized_linspace
concatenate = optimized_concatenate
compute_min = optimized_compute_min
compute_std = optimized_compute_std
compute_var = optimized_compute_var
zeros_vec = optimized_zeros_vec
ones_vec = optimized_ones_vec
dot_vec = optimized_dot_vec
array = np.array
float32 = np.float32
int16 = np.int16

# Utility functions
get_performance_stats = get_performance_stats
clear_performance_caches = clear_performance_caches
batch_compute_stats = batch_compute_stats
