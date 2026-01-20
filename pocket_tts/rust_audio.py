"""Python bindings for Rust audio processing extensions."""

import ctypes
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class RustAudioProcessor:
    """Python wrapper for Rust audio processing functions."""

    def __init__(self, lib_path: Optional[str] = None):
        """Initialize the Rust audio processor.

        Args:
            lib_path: Path to the compiled Rust library. If None, searches standard locations.
        """
        self._lib = None
        self._lib_path = self._find_library(lib_path)
        self._load_library()

    def _find_library(self, lib_path: Optional[str]) -> str:
        """Find the Rust library file."""
        if lib_path:
            return lib_path

        # Search in standard locations
        search_paths = [
            Path(__file__).parent.parent.parent / "training" / "rust_exts" / "audio_ds",
            Path.cwd() / "training" / "rust_exts" / "audio_ds",
            Path("/usr/local/lib"),
            Path("/opt/homebrew/lib"),
        ]

        for path in search_paths:
            if path.exists():
                # Look for the library
                for name in ["libpocket_tts_audio_ds.so", "libpocket_tts_audio_ds.dylib"]:
                    lib_file = path / name
                    if lib_file.exists():
                        return str(lib_file)

        # Return default path for building
        return str(
            Path.cwd()
            / "training"
            / "rust_exts"
            / "audio_ds"
            / "target"
            / "release"
            / "libpocket_tts_audio_ds.so"
        )

    def _load_library(self):
        """Load the Rust shared library."""
        try:
            self._lib = ctypes.CDLL(self._lib_path)
            self._setup_prototypes()
        except OSError as e:
            warnings.warn(
                f"Could not load Rust library from {self._lib_path}. "
                f"Rust functions will not be available. Build with: cd training/rust_exts/audio_ds && cargo build --release\n"
                f"Error: {e}"
            )
            self._lib = None

    def _setup_prototypes(self):
        """Set up C function prototypes for type safety."""
        # normalize_audio
        self._lib.normalize_audio.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        self._lib.normalize_audio.restype = ctypes.POINTER(ctypes.c_float)

        # apply_gain
        self._lib.apply_gain.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        self._lib.apply_gain.restype = ctypes.POINTER(ctypes.c_float)

        # resample_linear
        self._lib.resample_linear.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._lib.resample_linear.restype = ctypes.POINTER(ctypes.c_float)

        # resample_sinc
        self._lib.resample_sinc.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._lib.resample_sinc.restype = ctypes.POINTER(ctypes.c_float)

        # int16_to_float32
        self._lib.int16_to_float32.argtypes = [ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t]
        self._lib.int16_to_float32.restype = ctypes.POINTER(ctypes.c_float)

        # float32_to_int16
        self._lib.float32_to_int16.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.float32_to_int16.restype = ctypes.POINTER(ctypes.c_int16)

        # apply_fade
        self._lib.apply_fade.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._lib.apply_fade.restype = ctypes.POINTER(ctypes.c_float)

        # free_audio_buffer
        self._lib.free_audio_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.free_audio_buffer.restype = None

        # log10_vec
        self._lib.log10_vec.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.log10_vec.restype = ctypes.POINTER(ctypes.c_float)

        # compute_rms
        self._lib.compute_rms.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.compute_rms.restype = ctypes.c_float

        # compute_peak
        self._lib.compute_peak.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.compute_peak.restype = ctypes.c_float

    def _to_c_array(
        self, samples: np.ndarray
    ) -> Tuple[ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]:
        """Convert numpy array to C array."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        size = samples.size
        c_array = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return c_array, size

    def _from_c_array(
        self, c_ptr: ctypes.POINTER(ctypes.c_float), size: ctypes.c_size_t
    ) -> np.ndarray:
        """Convert C array to numpy array."""
        return np.ctypeslib.as_array(c_ptr, shape=(size,)).copy()

    def normalize(self, samples: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """Normalize audio samples to [-1, 1] range.

        Args:
            samples: Input audio samples
            gain: Additional gain factor (default: 1.0)

        Returns:
            Normalized audio samples
        """
        if self._lib is None:
            # Fallback to pure Python implementation
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                return samples / max_val * 0.99 * gain
            return samples

        c_array, size = self._to_c_array(samples)
        result_ptr = self._lib.normalize_audio(c_array, size, gain)
        result = self._from_c_array(result_ptr, size)
        self._lib.free_audio_buffer(result_ptr, size)
        return result

    def apply_gain(self, samples: np.ndarray, gain: float) -> np.ndarray:
        """Apply gain to audio samples.

        Args:
            samples: Input audio samples
            gain: Gain factor to apply

        Returns:
            Audio samples with gain applied
        """
        if self._lib is None:
            return samples * gain

        c_array, size = self._to_c_array(samples)
        result_ptr = self._lib.apply_gain(c_array, size, gain)
        result = self._from_c_array(result_ptr, size)
        self._lib.free_audio_buffer(result_ptr, size)
        return result

    def resample_linear(self, samples: np.ndarray, target_length: int) -> np.ndarray:
        """Resample audio using linear interpolation.

        Args:
            samples: Input audio samples
            target_length: Desired output length

        Returns:
            Resampled audio samples
        """
        if self._lib is None:
            # Fallback to numpy interpolation
            original_length = len(samples)
            indices = np.linspace(0, original_length - 1, target_length)
            return np.interp(indices, np.arange(original_length), samples).astype(np.float32)

        c_array, size = self._to_c_array(samples)
        result_ptr = self._lib.resample_linear(c_array, size, target_length)
        result = self._from_c_array(result_ptr, target_length)
        self._lib.free_audio_buffer(result_ptr, target_length)
        return result

    def resample_sinc(
        self, samples: np.ndarray, target_length: int, quality: int = 8
    ) -> np.ndarray:
        """Resample audio using high-quality sinc interpolation.

        Args:
            samples: Input audio samples
            target_length: Desired output length
            quality: Quality parameter (8, 12, or 16)

        Returns:
            Resampled audio samples
        """
        if self._lib is None:
            # Fallback to linear interpolation
            return self.resample_linear(samples, target_length)

        c_array, size = self._to_c_array(samples)
        result_ptr = self._lib.resample_sinc(c_array, size, target_length, quality)
        result = self._from_c_array(result_ptr, target_length)
        self._lib.free_audio_buffer(result_ptr, target_length)
        return result

    def apply_fade(
        self, samples: np.ndarray, fade_in_ms: float, fade_out_ms: float, sample_rate: int = 24000
    ) -> np.ndarray:
        """Apply fade in/out to audio.

        Args:
            samples: Input audio samples
            fade_in_ms: Fade in duration in milliseconds
            fade_out_ms: Fade out duration in milliseconds
            sample_rate: Sample rate

        Returns:
            Audio with fade applied
        """
        fade_in_samples = int(fade_in_ms * sample_rate / 1000)
        fade_out_samples = int(fade_out_ms * sample_rate / 1000)

        if self._lib is None:
            # Fallback to numpy implementation
            faded = samples.copy()
            if fade_in_samples > 0:
                fade_in = np.linspace(0, 1, fade_in_samples)
                faded[:fade_in_samples] *= fade_in
            if fade_out_samples > 0:
                fade_out = np.linspace(1, 0, fade_out_samples)
                faded[-fade_out_samples:] *= fade_out
            return faded

        c_array, size = self._to_c_array(samples)
        result_ptr = self._lib.apply_fade(c_array, size, fade_in_samples, fade_out_samples)
        result = self._from_c_array(result_ptr, size)
        self._lib.free_audio_buffer(result_ptr, size)
        return result

    def compute_rms(self, samples: np.ndarray) -> float:
        """Compute RMS energy of audio.

        Args:
            samples: Input audio samples

        Returns:
            RMS energy value
        """
        if self._lib is None:
            return np.sqrt(np.mean(samples**2))

        c_array, size = self._to_c_array(samples)
        return self._lib.compute_rms(c_array, size)

    def compute_peak(self, samples: np.ndarray) -> float:
        """Compute peak amplitude of audio.

        Args:
            samples: Input audio samples

        Returns:
            Peak amplitude value
        """
        if self._lib is None:
            return np.max(np.abs(samples))

        c_array, size = self._to_c_array(samples)
        return self._lib.compute_peak(c_array, size)

    def is_available(self) -> bool:
        """Check if Rust library is available."""
        return self._lib is not None


# Global instance
_rust_processor: Optional[RustAudioProcessor] = None


def get_rust_processor() -> RustAudioProcessor:
    """Get the global Rust audio processor instance."""
    global _rust_processor
    if _rust_processor is None:
        _rust_processor = RustAudioProcessor()
    return _rust_processor


def normalize_audio(samples: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Normalize audio samples using Rust if available, else Python fallback.

    Args:
        samples: Input audio samples
        gain: Additional gain factor

    Returns:
        Normalized audio samples
    """
    processor = get_rust_processor()
    return processor.normalize(samples, gain)


def apply_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    """Apply gain to audio samples.

    Args:
        samples: Input audio samples
        gain: Gain factor

    Returns:
        Audio with gain applied
    """
    processor = get_rust_processor()
    return processor.apply_gain(samples, gain)


def resample_audio(samples: np.ndarray, target_length: int, method: str = "linear") -> np.ndarray:
    """Resample audio to target length.

    Args:
        samples: Input audio samples
        target_length: Desired output length
        method: Resampling method ("linear" or "sinc")

    Returns:
        Resampled audio
    """
    processor = get_rust_processor()
    if method == "sinc":
        return processor.resample_sinc(samples, target_length)
    else:
        return processor.resample_linear(samples, target_length)


def apply_fade(
    samples: np.ndarray, fade_in_ms: float = 10, fade_out_ms: float = 10, sample_rate: int = 24000
) -> np.ndarray:
    """Apply fade in/out to audio.

    Args:
        samples: Input audio samples
        fade_in_ms: Fade in duration in milliseconds
        fade_out_ms: Fade out duration in milliseconds
        sample_rate: Sample rate

    Returns:
        Audio with fade applied
    """
    processor = get_rust_processor()
    return processor.apply_fade(samples, fade_in_ms, fade_out_ms, sample_rate)


def compute_audio_metrics(samples: np.ndarray) -> dict:
    """Compute audio quality metrics.

    Args:
        samples: Input audio samples

    Returns:
        Dictionary with RMS, peak, and other metrics
    """
    processor = get_rust_processor()
    rms = processor.compute_rms(samples)
    peak = processor.compute_peak(samples)
    dynamic_range_db = 20 * np.log10(peak / max(rms, 1e-10))
    return {"rms": rms, "peak": peak, "dynamic_range_db": dynamic_range_db}


# def log10_array(samples: np.ndarray) -> np.ndarray:
#     """Compute base-10 logarithm element-wise using Rust if available.
#
#     Args:
#         samples: Input audio samples
#
#     Returns:
#         Array with log10 of each element
#     """
#     if self._lib is None:
#         return np.log10(samples)
#
#     c_array, size = self._to_c_array(samples)
#     result_ptr = self._lib.log10_vec(c_array, size)
#     result = self._from_c_array(result_ptr, size)
#     self._lib.free_audio_buffer(result_ptr, size)
#     return result
