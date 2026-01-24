"""
NumPy replacement using pure Rust library via C ABI.

This module provides NumPy-like functions backed by Rust implementations
from the audio_ds crate. When available, functions are called
via ctypes FFI for maximum performance. When unavailable, functions
fall back to standard NumPy.
"""

import ctypes
import warnings
from pathlib import Path

import numpy as _np
from numpy import ctypeslib

# Global library instance
_LIB = None
_AVAILABLE = False


class LibraryLoader:
    def __init__(self):
        self._load_library()

    def _load_library(self):
        """Load libpocket_tts_audio_ds.so from standard locations."""
        global _LIB, _AVAILABLE

        search_paths = [
            Path(__file__).parent.parent.parent
            / "training"
            / "rust_exts"
            / "audio_ds"
            / "target"
            / "release",
            Path(__file__).parent.parent / "target" / "release",
            Path("/usr/local/lib"),
        ]

        for path in search_paths:
            for name in [
                "libnumpy.so",
                "libpocket_tts_audio_ds.so",
                "libpocket_tts_audio_ds.dylib",
            ]:
                lib_path = path / name
                if lib_path.exists():
                    _LIB = ctypes.CDLL(str(lib_path))
                    _AVAILABLE = True
                    return

        warnings.warn(
            "Could not load libpocket_tts_audio_ds.so. "
            "Rust functions will not be available. "
            "Build with: cd training/rust_exts/audio_ds && cargo build --release"
        )
        _LIB = None
        _AVAILABLE = False


_LOADER = LibraryLoader()


def arange(start, stop, step=1.0):
    if _AVAILABLE:
        num = int((stop - start) / step)
        _LIB.arange.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        _LIB.arange.restype = ctypes.POINTER(ctypes.c_float)
        ptr = _LIB.arange(start, stop, step)
        return [ptr[i] for i in range(num)]
    else:
        import numpy as np

        return np.arange(start, stop, step)


def log_vec(samples):
    if _AVAILABLE:
        size = samples.size
        _LIB.log_vec.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        _LIB.log_vec.restype = ctypes.POINTER(ctypes.c_float)
        c_array, size = _to_c_array(samples)
        ptr = _LIB.log_vec(c_array, size)
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(ptr, size)
        return result
    else:
        import numpy as np

        return np.log(samples)


def clip_vec(samples, a_min, a_max):
    if _AVAILABLE:
        size = samples.size
        _LIB.clip_vec.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
        ]
        _LIB.clip_vec.restype = ctypes.POINTER(ctypes.c_float)
        c_array, size = _to_c_array(samples)
        ptr = _LIB.clip_vec(c_array, size, a_min, a_max)
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(ptr, size)
        return result
    else:
        import numpy as np

        return np.clip(samples, a_min, a_max)


def power_vec(samples, exponent):
    if _AVAILABLE:
        size = samples.size
        _LIB.power_vec.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        _LIB.power_vec.restype = ctypes.POINTER(ctypes.c_float)
        c_array, size = _to_c_array(samples)
        ptr = _LIB.power_vec(c_array, size, exponent)
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(ptr, size)
        return result
    else:
        import numpy as np

        return np.power(samples, exponent)


def compute_min(samples):
    if _AVAILABLE:
        size = samples.size
        _LIB.compute_min.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        _LIB.compute_min.restype = ctypes.c_float
        c_array, size = _to_c_array(samples)
        result = _LIB.compute_min(c_array, size)
        return result
    else:
        import numpy as np

        return np.min(samples)


def compute_std(samples):
    if _AVAILABLE:
        size = samples.size
        _LIB.compute_std.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        _LIB.compute_std.restype = ctypes.c_float
        c_array, size = _to_c_array(samples)
        result = _LIB.compute_std(c_array, size)
        return result
    else:
        import numpy as np

        return np.std(samples)


def compute_var(samples):
    if _AVAILABLE:
        size = samples.size
        _LIB.compute_var.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        _LIB.compute_var.restype = ctypes.c_float
        c_array, size = _to_c_array(samples)
        result = _LIB.compute_var(c_array, size)
        return result
    else:
        import numpy as np

        return np.var(samples)


def zeros_vec(size):
    if _AVAILABLE:
        _LIB.zeros_vec.argtypes = [ctypes.c_size_t]
        _LIB.zeros_vec.restype = ctypes.POINTER(ctypes.c_float)
        ptr = _LIB.zeros_vec(size)
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(ptr, size)
        return result
    else:
        import numpy as np

        return np.zeros(size)


def ones_vec(size):
    if _AVAILABLE:
        _LIB.ones_vec.argtypes = [ctypes.c_size_t]
        _LIB.ones_vec.restype = ctypes.POINTER(ctypes.c_float)
        ptr = _LIB.ones_vec(size)
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(ptr, size)
        return result
    else:
        import numpy as np

        return np.ones(size)


def eye(n):
    if _AVAILABLE:
        _LIB.eye.argtypes = [ctypes.c_size_t]
        _LIB.eye.restype = ctypes.POINTER(ctypes.c_float)
        ptr = _LIB.eye(n)
        result = _from_c_array(ptr, n * n)
        _LIB.free_float_buffer(ptr, n * n)
        return result
    else:
        import numpy as np

        return np.eye(n)


def dot_vec(a, b):
    if _AVAILABLE:
        a_size = a.size
        b_size = b.size
        _LIB.dot_vec.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        _LIB.dot_vec.restype = ctypes.c_float
        c_a, a_size = _to_c_array(a)
        c_b, b_size = _to_c_array(b)
        result = _LIB.dot_vec(c_a, a_size, c_b, b_size)
        _LIB.free_float_buffer(c_a, a_size)
        _LIB.free_float_buffer(c_b, b_size)
        return result
    else:
        import numpy as np

        return np.dot(a, b)


def matmul_2d(a, b):
    if _AVAILABLE:
        a_rows = a.shape[0]
        a_cols = a.shape[1]
        b_rows = b.shape[0]
        b_cols = b.shape[1]
        _LIB.matmul_2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        _LIB.matmul_2d.restype = ctypes.POINTER(ctypes.c_float)
        c_a, a_size = _to_c_array(a)
        c_b, b_size = _to_c_array(b)
        ptr = _LIB.matmul_2d(c_a, a_rows, a_cols, c_b, b_rows, b_cols)
        result = _from_c_array(ptr, a_rows * b_cols)
        _LIB.free_float_buffer(c_a, a_size)
        _LIB.free_float_buffer(c_b, b_size)
        return result
    else:
        import numpy as np

        return np.matmul(a, b)


def reshape_vec(data, new_shape):
    if _AVAILABLE:
        size = data.size
        new_size = int(new_shape[0]) * int(new_shape[1])
        if size != new_size:
            raise ValueError(
                f"cannot reshape array of size {size} into shape {new_shape}"
            )
        _LIB.reshape_vec.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        _LIB.reshape_vec.restype = ctypes.POINTER(ctypes.c_float)
        c_data, size = _to_c_array(data)
        c_shape = (ctypes.c_size_t * 2)(
            *(ctypes.c_size_t.from_buffer(new_shape.encode(), 8))
        )
        ptr = _LIB.reshape_vec(c_data, size, c_shape, ctypes.size_t(c_shape))
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(c_data, size)
        return result
    else:
        import numpy as np

        return np.reshape(data, new_shape)


def transpose_2d(data):
    if _AVAILABLE:
        rows = data.shape[0]
        cols = data.shape[1]
        _LIB.transpose_2d.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        _LIB.transpose_2d.restype = ctypes.POINTER(ctypes.c_float)
        c_data, size = _to_c_array(data)
        ptr = _LIB.transpose_2d(c_data, rows, cols)
        result = _from_c_array(ptr, size)
        _LIB.free_float_buffer(c_data, size)
        return result
    else:
        import numpy as np

        return np.transpose(data)


def concatenate(arrays, axis=None):
    if _AVAILABLE:
        count = len(arrays)
        sizes = (ctypes.c_size_t * count)(
            *(ctypes.c_size_t.from_buffer([len(a.shape) for a in arrays], 8))
        )
        arrays_ptrs = (ctypes.POINTER(ctypes.c_size_t) * count)(
            *(_to_c_array_ptr(a) for a in arrays)
        )
        _LIB.concatenate.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        _LIB.concatenate.restype = ctypes.POINTER(ctypes.c_float)
        total_size = sum(a.size for a in arrays)
        ptr = _LIB.concatenate(arrays_ptrs, sizes, count)
        result = _from_c_array(ptr, total_size)
        for a in arrays:
            _LIB.free_float_buffer(_to_c_array_ptr(a), a.size)
        _LIB.free_float_buffer(ptr, total_size)
        return result
    else:
        import numpy as np

        return np.concatenate(arrays, axis=axis)


def vstack(arrays):
    if _AVAILABLE:
        return concatenate(arrays, axis=0)
    else:
        import numpy as np

        return np.vstack(arrays)


def hstack(arrays):
    if _AVAILABLE:
        return concatenate(arrays, axis=1)
    else:
        import numpy as np

        return np.hstack(arrays)


def _to_c_array(arr):
    c_array, size = _to_c_array_ptr(arr)
    result = _from_c_array(c_array, size)
    return result, c_array


def _to_c_array_ptr(arr):
    if not hasattr(arr, "ctypes"):
        return None
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _from_c_array(c_array, size):
    result = ctypeslib.as_array(c_array, shape=(size,)).copy()
    return result


def linspace(start, stop, num=50):
    if _AVAILABLE:
        _LIB.linspace.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_size_t]
        _LIB.linspace.restype = ctypes.POINTER(ctypes.c_float)
        ptr = _LIB.linspace(start, stop, num)
        result = _from_c_array(ptr, num)
        _LIB.free_float_buffer(ptr, num)
        return result
    else:
        import numpy as np

        return np.linspace(start, stop, num)


def interp(x, xp, fp):
    if _AVAILABLE:
        x_size = x.size
        xp_size = xp.size
        fp_size = fp.size
        _LIB.interp.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        _LIB.interp.restype = ctypes.POINTER(ctypes.c_float)
        c_x, x_size = _to_c_array(x)
        c_xp, xp_size = _to_c_array(xp)
        c_fp, fp_size = _to_c_array(fp)
        ptr = _LIB.interp(c_x, x_size, c_xp, c_fp, fp_size)
        result = _from_c_array(ptr, x_size)
        _LIB.free_float_buffer(c_x, xp_size)
        _LIB.free_float_buffer(c_xp, fp_size)
        _LIB.free_float_buffer(c_fp, fp_size)
        _LIB.free_float_buffer(ptr, x_size)
        return result
    else:
        import numpy as np

        return np.interp(x, xp, fp)


# -----------------------------------------------------------------------------
# NumPy Compatibility Shims
# -----------------------------------------------------------------------------

array = _np.array
float32 = _np.float32
int16 = _np.int16


def clip(a, a_min, a_max):
    """Clip (limit) the values in an array."""
    if _AVAILABLE:
        # Note: clip_vec handles _available check too, but assumes specific signature.
        # Numpy clip handles scalars or different args.
        # For now, just delegate to clip_vec if it matches our use case, else fallback.
        try:
            return clip_vec(a, a_min, a_max)
        except Exception:
            # Fallback to numpy if rust-numpy clip fails
            from pocket_tts.numpy_rs import logger

            logger.debug("rust-numpy clip failing, falling back to numpy")
    return _np.clip(a, a_min, a_max)


prod = _np.prod
