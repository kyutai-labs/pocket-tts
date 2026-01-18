"""
Drop-in NumPy replacement using rust-numpy.

This module provides transparent NumPy replacement by attempting to import
rust-numpy and falling back to NumPy if unavailable.
"""

import sys
from typing import Any, Union, List, Optional, Sequence

# Try to import rust-numpy
try:
    import numpy_rs  # type: ignore

    _RUST_NUMPY_AVAILABLE = True
except ImportError:
    _RUST_NUMPY_AVAILABLE = False
    import numpy as np  # type: ignore

    if not any("pocket_tts" in mod for mod in sys.modules):
        print("Warning: rust-numpy not available, falling back to NumPy")


def _ensure_array(arr: Any) -> Any:
    """Ensure input is a numpy_rs.Array or convert to numpy array."""
    if _RUST_NUMPY_AVAILABLE and hasattr(arr, "array"):
        return arr
    elif _RUST_NUMPY_AVAILABLE and isinstance(arr, list):
        # Convert list to rust-numpy Array
        return numpy_rs.array(arr)
    else:
        # Fall back to numpy
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        return arr


def arange(start: float, stop: float, step: float | None = 1.0) -> Any:
    """Generate array of values (np.arange replacement).

    Args:
        start: Start value (inclusive)
        stop: Stop value (exclusive)
        step: Step size (optional, defaults to 1.0)

    Returns:
        Array of values from start to stop with given step
    """
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.arange(start, stop, step)
    else:
        return np.arange(start, stop, step)


def array(data: Any, dtype: Optional[str] = None) -> Any:
    """Create array from data (np.array replacement).

    Args:
        data: Input data (list, tuple, array, etc.)
        dtype: Data type (optional, currently ignored for compatibility)

    Returns:
        Array with given data
    """
    if _RUST_NUMPY_AVAILABLE:
        # Convert list/tuple to rust-numpy Array
        if isinstance(data, (list, tuple)):
            return numpy_rs.array(list(data) if isinstance(data, tuple) else data)
        else:
            return numpy_rs.array(data)
    else:
        return np.array(data, dtype=dtype)


def clip(a: Any, a_min: Optional[float] = None, a_max: Optional[float] = None) -> Any:
    """Clip values to be within a specified range (np.clip replacement).

    Args:
        a: Input array
        a_min: Minimum value (values below this are set to this)
        a_max: Maximum value (values above this are set to this)

    Returns:
        Array with clipped values
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.clip(a, a_min, a_max)
    else:
        return np.clip(a, a_min, a_max)


def min(a: Any) -> float:
    """Find minimum value in an array (np.min replacement).

    Args:
        a: Input array

    Returns:
        Minimum value
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.min(a)
    else:
        return float(np.min(a))


def max(a: Any) -> float:
    """Find maximum value in an array (np.max replacement).

    Args:
        a: Input array

    Returns:
        Maximum value
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.max(a)
    else:
        return float(np.max(a))


def mean(a: Any) -> float:
    """Calculate mean of array elements (np.mean replacement).

    Args:
        a: Input array

    Returns:
        Mean value
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.mean(a)
    else:
        return float(np.mean(a))


def median(a: Any) -> float:
    """Calculate median of array elements (np.median replacement).

    Args:
        a: Input array

    Returns:
        Median value
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.median(a)
    else:
        return float(np.median(a))


def sum(a: Any) -> float:
    """Calculate sum of array elements (np.sum replacement).

    Args:
        a: Input array

    Returns:
        Sum of all elements
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.sum(a)
    else:
        return float(np.sum(a))


def sqrt(a: Any) -> Any:
    """Calculate square root element-wise (np.sqrt replacement).

    Args:
        a: Input array

    Returns:
        Array with square root of each element
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.sqrt(a)
    else:
        return np.sqrt(a)


def log(a: Any) -> Any:
    """Compute natural logarithm element-wise (np.log replacement).

    Args:
        a: Input array

    Returns:
        Array with natural log of each element
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.log(a)
    else:
        return np.log(a)


def std(a: Any) -> float:
    """Calculate standard deviation (np.std replacement).

    Args:
        a: Input array

    Returns:
        Standard deviation
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.std(a)
    else:
        return float(np.std(a))


def var(a: Any) -> float:
    """Calculate variance (np.var replacement).

    Args:
        a: Input array

    Returns:
        Variance
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.var(a)
    else:
        return float(np.var(a))


def reshape(a: Any, new_shape: Sequence[int]) -> Any:
    """Reshape array to new shape (np.reshape replacement).

    Args:
        a: Input array
        new_shape: New shape

    Returns:
        Reshaped array
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.reshape(a, list(new_shape))
    else:
        return np.reshape(a, new_shape)


def transpose(a: Any) -> Any:
    """Transpose array (np.T replacement).

    Args:
        a: Input array

    Returns:
        Transposed array
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.transpose(a)
    else:
        return a.T


def concatenate(arrays: Sequence[Any], axis: Optional[int] = 0) -> Any:
    """Concatenate arrays (np.concatenate replacement).

    Args:
        arrays: Sequence of arrays to concatenate
        axis: Axis to concatenate along (optional, defaults to 0)

    Returns:
        Concatenated array
    """
    if _RUST_NUMPY_AVAILABLE:
        rust_arrays = [_ensure_array(arr) for arr in arrays]
        return numpy_rs.concatenate(rust_arrays)
    else:
        numpy_arrays = [_ensure_array(arr) for arr in arrays]
        return np.concatenate(numpy_arrays, axis=axis)


def vstack(arrays: Sequence[Any]) -> Any:
    """Stack arrays vertically (np.vstack replacement).

    Args:
        arrays: Sequence of 2D arrays to stack

    Returns:
        Vertically stacked array
    """
    if _RUST_NUMPY_AVAILABLE:
        rust_arrays = [_ensure_array(arr) for arr in arrays]
        return numpy_rs.vstack(rust_arrays)
    else:
        numpy_arrays = [_ensure_array(arr) for arr in arrays]
        return np.vstack(numpy_arrays)


def hstack(arrays: Sequence[Any]) -> Any:
    """Stack arrays horizontally (np.hstack replacement).

    Args:
        arrays: Sequence of 1D arrays to stack

    Returns:
        Horizontally stacked array
    """
    if _RUST_NUMPY_AVAILABLE:
        rust_arrays = [_ensure_array(arr) for arr in arrays]
        return numpy_rs.hstack(rust_arrays)
    else:
        numpy_arrays = [_ensure_array(arr) for arr in arrays]
        return np.hstack(numpy_arrays)


def zeros(shape: Sequence[int]) -> Any:
    """Generate array of zeros (np.zeros replacement).

    Args:
        shape: Shape of output array

    Returns:
        Array filled with zeros
    """
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.zeros(list(shape))
    else:
        return np.zeros(shape)


def ones(shape: Sequence[int]) -> Any:
    """Generate array of ones (np.ones replacement).

    Args:
        shape: Shape of output array

    Returns:
        Array filled with ones
    """
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.ones(list(shape))
    else:
        return np.ones(shape)


def eye(n: int, m: Optional[int] = None) -> Any:
    """Generate identity matrix (np.eye replacement).

    Args:
        n: Size of square matrix
        m: Number of columns (optional, defaults to n)

    Returns:
        Identity matrix of shape (n, m)
    """
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.eye(n, m)
    else:
        return np.eye(n, m)


def linspace(start: float, stop: float, num: int, endpoint: bool = False) -> Any:
    """Generate linearly spaced values (np.linspace replacement).

    Args:
        start: Start value
        stop: End value
        num: Number of values to generate
        endpoint: Whether to include stop value (optional, defaults to False)

    Returns:
        Array of linearly spaced values
    """
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.linspace(start, stop, num)
    else:
        return np.linspace(start, stop, num, endpoint=endpoint)


def interp(x: Any, xp: Any, fp: Any) -> Any:
    """Interpolate values (np.interp replacement).

    Args:
        x: X-coordinates at which to evaluate
        xp: X-coordinates of data points
        fp: Y-coordinates of data points

    Returns:
        Interpolated values
    """
    x = _ensure_array(x)
    xp = _ensure_array(xp)
    fp = _ensure_array(fp)

    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.interp(x, xp, fp)
    else:
        return np.interp(x, xp, fp)


def dot(a: Any, b: Any) -> Any:
    """Compute dot product (np.dot replacement).

    Args:
        a: First array
        b: Second array

    Returns:
        Dot product
    """
    a = _ensure_array(a)
    b = _ensure_array(b)

    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.dot(a, b)
    else:
        return np.dot(a, b)


def matmul(a: Any, b: Any) -> Any:
    """Matrix multiplication (np.matmul replacement).

    Args:
        a: First array
        b: Second array

    Returns:
        Matrix product
    """
    a = _ensure_array(a)
    b = _ensure_array(b)

    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.matmul(a, b)
    else:
        return np.matmul(a, b)


def abs(a: Any) -> Any:
    """Compute absolute values element-wise (np.abs replacement).

    Args:
        a: Input array

    Returns:
        Array with absolute values
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.abs(a)
    else:
        return np.abs(a)


def power(a: Any, n: float) -> Any:
    """Compute power element-wise (np.power replacement).

    Args:
        a: Input array
        n: Exponent

    Returns:
        Array raised to power n
    """
    a = _ensure_array(a)
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.power(a, n)
    else:
        return np.power(a, n)


# Export all functions to make them available
__all__ = [
    "arange",
    "array",
    "clip",
    "min",
    "max",
    "mean",
    "median",
    "sum",
    "sqrt",
    "log",
    "std",
    "var",
    "reshape",
    "transpose",
    "concatenate",
    "vstack",
    "hstack",
    "zeros",
    "ones",
    "eye",
    "linspace",
    "interp",
    "dot",
    "matmul",
    "abs",
    "power",
    "_RUST_NUMPY_AVAILABLE",
]
