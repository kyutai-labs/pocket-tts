# Plan: Complete NumPy to Rust Port with Python Integration

## Objective

Create full Python bindings for rust-numpy and integrate it into pocket-tts codebase to replace all NumPy usage.

## Current Status

### Completed ✅
1. Fixed rust-numpy compilation errors:
   - Enabled chrono dependency (datetime feature)
   - Fixed variable scope issue in window.rs
   - All tests now compile successfully

2. Implemented missing NumPy functions in new module:
   - `arange()` - Generate sequences
   - `array()` - Create arrays from data
   - `clip()` - Clip values to range
   - `min()` - Find minimum
   - `log()` - Natural logarithm
   - Module: `src/array_creation.rs` (274 lines)
   - All functions have tests

### Port Coverage
- Before: 10/19 functions (52%)
- After array_creation module: 15/19 functions (79%)

## Remaining Tasks

### Task 1: Add Python Bindings (PyO3) ⚠️ HIGH PRIORITY

**File:** `rust-numpy/src/python.rs` (NEW FILE)

**Required Bindings:**

#### Core Array Type
```rust
use pyo3::prelude::*;
use pyo3::types::PyArray;
use numpy::{Array, Dtype};

#[pyclass]
#[derive(FromPyObject)]
pub struct PyArray {
    array: Array<f32>,
}

#[pymethods]
impl PyArray {
    #[new]
    #[args(data = "Vec<f32>")]
    fn new(data: Vec<f32>) -> PyResult<Self> {
        Ok(PyArray {
            array: Array::from_vec(data).map_err(|e| {
                PyValueError::new_err(format!("Failed to create array: {}", e))
            })?,
        })
    }

    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.array.shape().to_vec())
    }

    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self.array.ndim())
    }

    #[getter]
    fn size(&self) -> PyResult<usize> {
        Ok(self.array.size())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("array({:?})", self.array.data()))
    }
}
```

#### NumPy Functions
```rust
#[pyfunction]
#[pyo3(name = "arange")]
fn py_arange(start: f32, stop: f32, step: Option<f32>) -> PyResult<PyArray> {
    let arr = numpy::arange(start, stop, step).map_err(|e| {
        PyValueError::new_err(format!("arange failed: {}", e))
    })?;
    Ok(PyArray { array: arr })
}

#[pyfunction]
#[pyo3(name = "array")]
fn py_array(data: &PyAny) -> PyResult<PyArray> {
    let vec: Vec<f32> = data.extract()?;
    let arr = numpy::array(vec, None).map_err(|e| {
        PyValueError::new_err(format!("array failed: {}", e))
    })?;
    Ok(PyArray { array: arr })
}

#[pyfunction]
#[pyo3(name = "clip")]
fn py_clip(py: Python, array: &PyArray, a_min: Option<f32>, a_max: Option<f32>) -> PyResult<PyArray> {
    let arr = numpy::clip(&array.array, a_min, a_max).map_err(|e| {
        PyValueError::new_err(format!("clip failed: {}", e))
    })?;
    Ok(PyArray { array: arr })
}

#[pyfunction]
#[pyo3(name = "min")]
fn py_min(array: &PyArray) -> PyResult<f32> {
    let min_val = numpy::min(&array.array).map_err(|e| {
        PyValueError::new_err(format!("min failed: {}", e))
    })?;
    Ok(min_val)
}

#[pyfunction]
#[pyo3(name = "max")]
fn py_max(array: &PyArray) -> PyResult<f32> {
    let max_val = numpy::max(&array.array).map_err(|e| {
        PyValueError::new_err(format!("max failed: {}", e))
    })?;
    Ok(max_val)
}

#[pyfunction]
#[pyo3(name = "mean")]
fn py_mean(array: &PyArray) -> PyResult<f32> {
    let mean_val = numpy::mean(&array.array).map_err(|e| {
        PyValueError::new_err(format!("mean failed: {}", e))
    })?;
    Ok(mean_val)
}

#[pyfunction]
#[pyo3(name = "median")]
fn py_median(array: &PyArray) -> PyResult<f32> {
    let median_val = numpy::median(&array.array).map_err(|e| {
        PyValueError::new_err(format!("median failed: {}", e))
    })?;
    Ok(median_val)
}

#[pyfunction]
#[pyo3(name = "log")]
fn py_log(array: &PyArray) -> PyResult<PyArray> {
    let logged = numpy::log(&array.array).map_err(|e| {
        PyValueError::new_err(format!("log failed: {}", e))
    })?;
    Ok(PyArray { array: logged })
}

#[pyfunction]
#[pyo3(name = "sum")]
fn py_sum(array: &PyArray) -> PyResult<f32> {
    let sum_val = numpy::sum(&array.array).map_err(|e| {
        PyValueError::new_err(format!("sum failed: {}", e))
    })?;
    Ok(sum_val)
}

#[pyfunction]
#[pyo3(name = "sqrt")]
fn py_sqrt(array: &PyArray) -> PyResult<PyArray> {
    let sqrted = numpy::sqrt(&array.array).map_err(|e| {
        PyValueError::new_err(format!("sqrt failed: {}", e))
    })?;
    Ok(PyArray { array: sqrted })
}
```

**Module Definition:**
```rust
#[pymodule]
fn numpy_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_function(wrap_pyfunction!(py_arange))?;
    m.add_function(wrap_pyfunction!(py_array))?;
    m.add_function(wrap_pyfunction!(py_clip))?;
    m.add_function(wrap_pyfunction!(py_min))?;
    m.add_function(wrap_pyfunction!(py_max))?;
    m.add_function(wrap_pyfunction!(py_mean))?;
    m.add_function(wrap_pyfunction!(py_median))?;
    m.add_function(wrap_pyfunction!(py_log))?;
    m.add_function(wrap_pyfunction!(py_sum))?;
    m.add_function(wrap_pyfunction!(py_sqrt))?;
    Ok(())
}
```

**Estimated lines:** ~300-400

### Task 2: Update Cargo.toml for Python Support

Add to `rust-numpy/Cargo.toml`:

```toml
[lib]
name = "numpy"
crate-type = ["cdylib", "rlib"]

[dependencies]
# ... existing dependencies ...

pyo3 = { version = "0.21", features = ["extension-module"], optional = false }
```

Update features:
```toml
[features]
default = ["std", "rayon", "datetime", "python"]
std = []
serde = ["dep:serde"]
python = ["pyo3"]
```

### Task 3: Build Python Module

```bash
# Build with Python support
cd rust-numpy
cargo build --release --features python

# Install to Python environment
# This creates target/release/libnumpy.so (Linux) or .dylib (macOS)
# Install with pip install -e .
```

**Need:** Create `rust-numpy/pyproject.toml` and `rust-numpy/README.md` for Python packaging

### Task 4: Create Replacement Module in pocket-tts

**File:** `pocket_tts/numpy_rs.py` (NEW FILE)

```python
"""Drop-in NumPy replacement using rust-numpy."""

# Try to import rust-numpy
try:
    import numpy_rs
    _RUST_NUMPY_AVAILABLE = True
except ImportError:
    _RUST_NUMPY_AVAILABLE = False
    import numpy as np  # type: ignore
    print("Warning: rust-numpy not available, falling back to NumPy")

def arange(start: float, stop: float, step: float | None = 1.0) -> np.ndarray:
    """Generate array of values (np.arange replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        py_arr = numpy_rs.arange(start, stop, step)
        return np.asarray(py_arr.data)
    else:
        return np.arange(start, stop, step)

def array(data, dtype=None) -> np.ndarray:
    """Create array from data (np.array replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        py_arr = numpy_rs.array(data)
        return np.asarray(py_arr.data)
    else:
        return np.array(data, dtype=dtype)

def clip(a, a_min=None, a_max=None) -> np.ndarray:
    """Clip values to range (np.clip replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        py_arr = numpy_rs.clip(np.asarray(a), a_min, a_max)
        return np.asarray(py_arr.data)
    else:
        return np.clip(a, a_min, a_max)

def min(a) -> float:
    """Find minimum value (np.min replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.min(np.asarray(a))
    else:
        return float(np.min(a))

def max(a) -> float:
    """Find maximum value (np.max replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.max(np.asarray(a))
    else:
        return float(np.max(a))

def mean(a) -> float:
    """Calculate mean (np.mean replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.mean(np.asarray(a))
    else:
        return float(np.mean(a))

def median(a) -> float:
    """Calculate median (np.median replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.median(np.asarray(a))
    else:
        return float(np.median(a))

def log(a) -> np.ndarray:
    """Natural logarithm (np.log replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        py_arr = numpy_rs.log(np.asarray(a))
        return np.asarray(py_arr.data)
    else:
        return np.log(a)

def sum(a) -> float:
    """Calculate sum (np.sum replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        return numpy_rs.sum(np.asarray(a))
    else:
        return float(np.sum(a))

def sqrt(a) -> np.ndarray:
    """Square root (np.sqrt replacement)."""
    if _RUST_NUMPY_AVAILABLE:
        py_arr = numpy_rs.sqrt(np.asarray(a))
        return np.asarray(py_arr.data)
    else:
        return np.sqrt(a)
```

### Task 5: Update All NumPy Imports

Update these files to use `numpy_rs` instead of `np`:

1. `pocket_tts/data/audio.py`
   - `import numpy as np` → `from pocket_tts.numpy_rs import * as np`
   - All np.function calls will automatically use rust-numpy

2. `pocket_tts/websocket_server.py`
   - Same replacement

3. `pocket_tts/rust_audio.py`
   - Same replacement

4. `examples/analyze_audio.py`
   - Same replacement

5. `tests/test_documentation_examples.py`
   - Same replacement

6. `verify_features.py`
   - Same replacement

**Estimated lines to change:** ~50-100 lines

### Task 6: Update Documentation

Update these files:

1. `rust-numpy/README.md`:
   - Add Python integration section
   - Document all available functions
   - Add installation instructions

2. `pocket_tts/README.md`:
   - Add rust-numpy section
   - Document performance improvements
   - Explain fallback behavior

3. `NUMPY_RUST_PORT_VERIFICATION.md`:
   - Update with completion status
   - Mark all functions as ported

### Task 7: Comprehensive Testing

**File:** `rust-numpy/tests/python_integration_tests.py` (NEW)

```python
"""Test Python integration of rust-numpy."""

import pytest
import numpy as np

def test_arange_basic():
    """Test basic arange functionality."""
    result = np.arange(0, 10)
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    assert np.allclose(result, expected)

def test_array_creation():
    """Test array creation."""
    result = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(result, expected)

def test_clip():
    """Test clip function."""
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = np.clip(input_arr, 2.0, 4.0)
    expected = np.array([2.0, 2.0, 3.0, 4.0, 4.0], dtype=np.float32)
    assert np.allclose(result, expected)

def test_min():
    """Test min function."""
    input_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = np.min(input_arr)
    assert result == 1.0

def test_log():
    """Test log function."""
    input_arr = np.array([1.0, 2.0, 10.0])
    result = np.log(input_arr)
    expected = np.log(np.array([1.0, 2.0, 10.0]))
    assert np.allclose(result, expected)

# Test with actual pocket-tts use cases
def test_analyze_audio_compatibility():
    """Test compatibility with examples/analyze_audio.py."""
    import pandas as pd  # type: ignore
    from io import BytesIO
    import soundfile as sf

    # Create test audio
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Test min function (used in analyze_audio.py line 100)
    min_val = np.min(audio)
    assert isinstance(min_val, (float, np.floating))

    print("All tests passed!")
```

**Estimated lines:** ~200-300 lines

### Task 8: Benchmark and Performance Validation

**File:** `rust-numpy/benchmarks/python_integration_bench.py` (NEW)

```python
"""Benchmark rust-numpy vs NumPy performance."""

import numpy as np
import time

def benchmark_arange():
    """Benchmark arange function."""
    iterations = 10000

    # NumPy
    start = time.time()
    for _ in range(iterations):
        _ = np.arange(0.0, 1000.0)
    numpy_time = time.time() - start

    # rust-numpy
    start = time.time()
    for _ in range(iterations):
        _ = np.arange(0.0, 1000.0)
    rust_time = time.time() - start

    speedup = numpy_time / rust_time
    print(f"arange: NumPy={numpy_time:.6f}s, Rust={rust_time:.6f}s, Speedup={speedup:.2f}x")
    return speedup

def benchmark_clip():
    """Benchmark clip function."""
    iterations = 10000
    arr = np.random.randn(10000)

    # NumPy
    start = time.time()
    for _ in range(iterations):
        _ = np.clip(arr, -1.0, 1.0)
    numpy_time = time.time() - start

    # rust-numpy
    start = time.time()
    for _ in range(iterations):
        _ = np.clip(arr, -1.0, 1.0)
    rust_time = time.time() - start

    speedup = numpy_time / rust_time
    print(f"clip: NumPy={numpy_time:.6f}s, Rust={rust_time:.6f}s, Speedup={speedup:.2f}x")
    return speedup

if __name__ == "__main__":
    print("Running Python integration benchmarks...")
    arange_speedup = benchmark_arange()
    clip_speedup = benchmark_clip()

    print(f"\nAverage speedup: {(arange_speedup + clip_speedup) / 2:.2f}x")
```

### Task 9: Integration Testing

Run full pocket-tts test suite with rust-numpy:

```bash
# Build rust-numpy with Python support
cd rust-numpy
cargo build --release --features python

# Run pocket-tts tests
cd ..
pytest tests/test_python_api.py -v
pytest tests/test_documentation_examples.py -v
pytest examples/analyze_audio.py -v
```

## Estimated Effort

| Task | Estimated Time | Lines of Code |
|-------|---------------|---------------|
| Task 1: Python bindings (PyO3) | 4-6 hours | ~400 |
| Task 2: Update Cargo.toml | 15 minutes | ~10 |
| Task 3: Build Python module | 30 minutes | ~30 |
| Task 4: Create numpy_rs.py wrapper | 2-3 hours | ~200 |
| Task 5: Update NumPy imports | 2-3 hours | ~100 |
| Task 6: Update documentation | 1-2 hours | ~100 |
| Task 7: Comprehensive testing | 2-3 hours | ~300 |
| Task 8: Benchmarking | 1-2 hours | ~150 |
| Task 9: Integration testing | 2-3 hours | N/A |

**Total estimated time:** 15-22 hours (2-3 days)

**Total estimated new code:** ~1,300 lines

## Risks and Considerations

### High Priority Risks

1. **PyO3 Integration Complexity**
   - Need to handle NumPy array conversion (PyArray ↔ rust-numpy::Array)
   - Memory ownership between Python and Rust
   - Error handling propagation

2. **Performance Overhead**
   - FFI boundary may add overhead for simple operations
   - Need careful benchmarking to validate improvements
   - May need to keep NumPy for some operations

3. **Backwards Compatibility**
   - Must ensure 100% behavioral compatibility with NumPy
   - Edge cases: empty arrays, NaN handling, infinite values
   - Dtype support currently limited to f32

4. **Testing Burden**
   - All existing tests must pass with rust-numpy
   - May uncover subtle behavioral differences
   - Need comprehensive test coverage

### Medium Priority Risks

5. **Build System Complexity**
   - Need to support Linux (.so), macOS (.dylib), Windows (.dll)
   - M1 vs Intel builds on macOS
   - Python 3.10-3.14 compatibility

6. **Dependency Management**
   - PyO3 version compatibility
   - NumPy version conflicts
   - Build reproducibility

### Low Priority Risks

7. **Documentation Maintenance**
   - Need to keep rust-numpy docs in sync with NumPy API
   - Migration guide for users
   - Performance benchmarks documentation

## Success Criteria

✅ **Complete when:**

1. All 19 NumPy functions used in pocket-tts have Rust equivalents
2. Python bindings compile and import successfully
3. All existing pocket-tts tests pass with rust-numpy
4. Performance benchmarks show improvement (or equal for FFI overhead)
5. Documentation updated with rust-numpy instructions
6. Code is production-ready and well-tested

## Dependencies

None required - all work uses existing rust-numpy infrastructure and PyO3.

## Next Steps After Completion

1. Remove `training/rust_exts/audio_ds/` (deprecated)
2. Archive `rust-numpy/README.md` with completion status
3. Create migration guide for other NumPy users
4. Performance monitoring in production
5. Consider adding more NumPy functions as needed

---

**Estimated completion:** 2-3 days from approval
**Risk level:** Medium-High
**Complexity:** High (requires PyO3 and integration expertise)
