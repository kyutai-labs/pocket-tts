//! Python bindings for rust-numpy using PyO3
//!
//! This module provides Python bindings that allow rust-numpy to be used
//! as a drop-in replacement for NumPy in Python code.

use numpy::array_creation;
use numpy::dtype::Dtype;
use numpy::dtype::DtypeKind;
use numpy::Array;
use pyo3::prelude::*;
use pyo3::types::{PyArray, PyComplex, PyReadonlyArray};

// Re-export commonly used types
pub type PyResult<T> = Result<T, PyErr>;

/// Python wrapper for Rust NumPy Array
///
/// This provides a Python class that wraps the internal Rust Array
/// and exposes NumPy-compatible methods.
#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    array: Array<f32>,
}

#[pymethods]
impl PyArray {
    #[new]
    #[args(data = "Vec<f32>", dtype = "Option<&str>")]
    fn new(py: Python, data: Vec<f32>, dtype: Option<&str>) -> PyResult<Self> {
        // Parse dtype (currently only support f32)
        let _dtype = dtype;

        let array = Array::from_vec(data)
            .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;

        Ok(PyArray { array })
    }

    /// Get the shape of the array
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.array.shape().to_vec())
    }

    /// Get the number of dimensions
    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self.array.ndim())
    }

    /// Get the total number of elements
    #[getter]
    fn size(&self) -> PyResult<usize> {
        Ok(self.array.size())
    }

    /// Get the underlying data as a Python list
    #[getter]
    fn data(&self) -> PyResult<Vec<f32>> {
        Ok(self.array.data().to_vec())
    }

    /// Convert to string representation
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("array({:?})", self.array.data()))
    }

    /// Get length (for len() support)
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.array.size())
    }

    /// Index into the array (supports a[i])
    fn __getitem__(&self, idx: isize) -> PyResult<f32> {
        let data = self.array.data();
        if idx < 0 || idx >= data.len() as isize {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        Ok(data[idx as usize])
    }

    /// Set array element (supports a[i] = x)
    fn __setitem__(&mut self, py: Python, idx: isize, value: f32) -> PyResult<()> {
        let data = self.array.data_mut();
        if idx < 0 || idx >= data.len() as isize {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        data[idx as usize] = value;
        Ok(())
    }
}

/// Create an array from a range of values (np.arange replacement)
#[pyfunction]
#[pyo3(name = "arange")]
fn py_arange(py: Python, start: f32, stop: f32, step: Option<f32>) -> PyResult<PyArray> {
    let arr = numpy::array_creation::arange(start, stop, step)
        .map_err(|e| PyValueError::new_err(format!("arange failed: {}", e)))?;
    Ok(PyArray { array: arr })
}

/// Create an array from Python data (np.array replacement)
#[pyfunction]
#[pyo3(name = "array")]
fn py_array(py: Python, data: &PyAny, dtype: Option<&str>) -> PyResult<PyArray> {
    let vec: Vec<f32> = data
        .extract()
        .map_err(|e| PyValueError::new_err(format!("Failed to extract Vec<f32>: {}", e)))?;

    let arr = numpy::array_creation::array(vec, None)
        .map_err(|e| PyValueError::new_err(format!("array failed: {}", e)))?;

    Ok(PyArray { array: arr })
}

/// Clip values to be within a specified range (np.clip replacement)
#[pyfunction]
#[pyo3(name = "clip")]
fn py_clip(
    py: Python,
    array: &PyArray,
    a_min: Option<f32>,
    a_max: Option<f32>,
) -> PyResult<PyArray> {
    let arr = numpy::array_creation::clip(&array.array, a_min, a_max)
        .map_err(|e| PyValueError::new_err(format!("clip failed: {}", e)))?;
    Ok(PyArray { array: arr })
}

/// Find minimum value in an array (np.min replacement)
#[pyfunction]
#[pyo3(name = "min")]
fn py_min(array: &PyArray) -> PyResult<f32> {
    let min_val = numpy::array_creation::min(&array.array)
        .map_err(|e| PyValueError::new_err(format!("min failed: {}", e)))?;
    Ok(min_val)
}

/// Find maximum value in an array (np.max replacement)
#[pyfunction]
#[pyo3(name = "max")]
fn py_max(array: &PyArray) -> PyResult<f32> {
    let max_val = numpy::max(&array.array)
        .map_err(|e| PyValueError::new_err(format!("max failed: {}", e)))?;
    Ok(max_val)
}

/// Calculate mean of array elements (np.mean replacement)
#[pyfunction]
#[pyo3(name = "mean")]
fn py_mean(array: &PyArray) -> PyResult<f32> {
    let mean_val = numpy::mean(&array.array)
        .map_err(|e| PyValueError::new_err(format!("mean failed: {}", e)))?;
    Ok(mean_val)
}

/// Calculate median of array elements (np.median replacement)
#[pyfunction]
#[pyo3(name = "median")]
fn py_median(array: &PyArray) -> PyResult<f32> {
    let median_val = numpy::median(&array.array)
        .map_err(|e| PyValueError::new_err(format!("median failed: {}", e)))?;
    Ok(median_val)
}

/// Compute natural logarithm element-wise (np.log replacement)
#[pyfunction]
#[pyo3(name = "log")]
fn py_log(array: &PyArray) -> PyResult<PyArray> {
    let logged = numpy::array_creation::log(&array.array)
        .map_err(|e| PyValueError::new_err(format!("log failed: {}", e)))?;
    Ok(PyArray { array: logged })
}

/// Calculate sum of array elements (np.sum replacement)
#[pyfunction]
#[pyo3(name = "sum")]
fn py_sum(array: &PyArray) -> PyResult<f32> {
    let sum_val = numpy::sum(&array.array)
        .map_err(|e| PyValueError::new_err(format!("sum failed: {}", e)))?;
    Ok(sum_val)
}

/// Compute square root element-wise (np.sqrt replacement)
#[pyfunction]
#[pyo3(name = "sqrt")]
fn py_sqrt(array: &PyArray) -> PyResult<PyArray> {
    let sqrted = numpy::sqrt(&array.array)
        .map_err(|e| PyValueError::new_err(format!("sqrt failed: {}", e)))?;
    Ok(PyArray { array: sqrted })
}

/// Calculate standard deviation (np.std replacement)
#[pyfunction]
#[pyo3(name = "std")]
fn py_std(array: &PyArray) -> PyResult<f32> {
    let std_val = numpy::std(&array.array)
        .map_err(|e| PyValueError::new_err(format!("std failed: {}", e)))?;
    Ok(std_val)
}

/// Calculate variance (np.var replacement)
#[pyfunction]
#[pyo3(name = "var")]
fn py_var(array: &PyArray) -> PyResult<f32> {
    let var_val = numpy::var(&array.array)
        .map_err(|e| PyValueError::new_err(format!("var failed: {}", e)))?;
    Ok(var_val)
}

/// Reshape array to new shape (np.reshape replacement)
#[pyfunction]
#[pyo3(name = "reshape")]
fn py_reshape(array: &PyArray, new_shape: Vec<usize>) -> PyResult<PyArray> {
    let reshaped = array
        .array
        .reshape(new_shape)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {}", e)))?;
    Ok(PyArray { array: reshaped })
}

/// Transpose array (np.T replacement)
#[pyfunction]
#[pyo3(name = "transpose")]
fn py_transpose(array: &PyArray) -> PyResult<PyArray> {
    let transposed = array
        .array
        .t()
        .map_err(|e| PyValueError::new_err(format!("transpose failed: {}", e)))?;
    Ok(PyArray { array: transposed })
}

/// Concatenate arrays (np.concatenate replacement)
#[pyfunction]
#[pyo3(name = "concatenate")]
fn py_concatenate(py: Python, arrays: Vec<&PyArray>) -> PyResult<PyArray> {
    let rust_arrays: Vec<&Array<f32>> = arrays.iter().map(|arr| &arr.array).collect();

    let concatenated = numpy::concatenate(&rust_arrays)
        .map_err(|e| PyValueError::new_err(format!("concatenate failed: {}", e)))?;

    Ok(PyArray {
        array: concatenated,
    })
}

/// Stack arrays vertically (np.vstack replacement)
#[pyfunction]
#[pyo3(name = "vstack")]
fn py_vstack(py: Python, arrays: Vec<&PyArray>) -> PyResult<PyArray> {
    let rust_arrays: Vec<&Array<f32>> = arrays.iter().map(|arr| &arr.array).collect();

    let stacked = numpy::vstack(&rust_arrays)
        .map_err(|e| PyValueError::new_err(format!("vstack failed: {}", e)))?;

    Ok(PyArray { array: stacked })
}

/// Stack arrays horizontally (np.hstack replacement)
#[pyfunction]
#[pyo3(name = "hstack")]
fn py_hstack(py: Python, arrays: Vec<&PyArray>) -> PyResult<PyArray> {
    let rust_arrays: Vec<&Array<f32>> = arrays.iter().map(|arr| &arr.array).collect();

    let stacked = numpy::hstack(&rust_arrays)
        .map_err(|e| PyValueError::new_err(format!("hstack failed: {}", e)))?;

    Ok(PyArray { array: stacked })
}

/// Generate array of zeros (np.zeros replacement)
#[pyfunction]
#[pyo3(name = "zeros")]
fn py_zeros(py: Python, shape: Vec<usize>) -> PyResult<PyArray> {
    let zeros = numpy::zeros(shape, numpy::dtype::Dtype::from_type::<f32>())
        .map_err(|e| PyValueError::new_err(format!("zeros failed: {}", e)))?;

    Ok(PyArray { array: zeros })
}

/// Generate array of ones (np.ones replacement)
#[pyfunction]
#[pyo3(name = "ones")]
fn py_ones(py: Python, shape: Vec<usize>) -> PyResult<PyArray> {
    let ones = numpy::ones(shape, numpy::dtype::Dtype::from_type::<f32>())
        .map_err(|e| PyValueError::new_err(format!("ones failed: {}", e)))?;

    Ok(PyArray { array: ones })
}

/// Generate array of identity matrix (np.eye replacement)
#[pyfunction]
#[pyo3(name = "eye")]
fn py_eye(py: Python, n: usize, m: Option<usize>) -> PyResult<PyArray> {
    let eye = numpy::eye(n, m).map_err(|e| PyValueError::new_err(format!("eye failed: {}", e)))?;

    Ok(PyArray { array: eye })
}

/// Create linspace array (np.linspace replacement)
#[pyfunction]
#[pyo3(name = "linspace")]
fn py_linspace(py: Python, start: f32, stop: f32, num: usize) -> PyResult<PyArray> {
    let linspace = numpy::linspace(start, stop, num)
        .map_err(|e| PyValueError::new_err(format!("linspace failed: {}", e)))?;

    Ok(PyArray { array: linspace })
}

/// Interpolate values (np.interp replacement)
#[pyfunction]
#[pyo3(name = "interp")]
fn py_interp(py: Python, x: &PyArray, xp: &PyArray, fp: &PyArray) -> PyResult<PyArray> {
    let interpolated = numpy::interp(&x.array, &xp.array, &fp.array)
        .map_err(|e| PyValueError::new_err(format!("interp failed: {}", e)))?;

    Ok(PyArray {
        array: interpolated,
    })
}

/// Dot product of arrays (np.dot replacement)
#[pyfunction]
#[pyo3(name = "dot")]
fn py_dot(array1: &PyArray, array2: &PyArray) -> PyResult<PyArray> {
    let result = numpy::dot(&array1.array, &array2.array)
        .map_err(|e| PyValueError::new_err(format!("dot failed: {}", e)))?;

    Ok(PyArray { array: result })
}

/// Matrix multiplication (np.matmul replacement)
#[pyfunction]
#[pyo3(name = "matmul")]
fn py_matmul(array1: &PyArray, array2: &PyArray) -> PyResult<PyArray> {
    let result = numpy::matmul(&array1.array, &array2.array)
        .map_err(|e| PyValueError::new_err(format!("matmul failed: {}", e)))?;

    Ok(PyArray { array: result })
}

/// Module definition
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
    m.add_function(wrap_pyfunction!(py_std))?;
    m.add_function(wrap_pyfunction!(py_var))?;
    m.add_function(wrap_pyfunction!(py_reshape))?;
    m.add_function(wrap_pyfunction!(py_transpose))?;
    m.add_function(wrap_pyfunction!(py_concatenate))?;
    m.add_function(wrap_pyfunction!(py_vstack))?;
    m.add_function(wrap_pyfunction!(py_hstack))?;
    m.add_function(wrap_pyfunction!(py_zeros))?;
    m.add_function(wrap_pyfunction!(py_ones))?;
    m.add_function(wrap_pyfunction!(py_eye))?;
    m.add_function(wrap_pyfunction!(py_linspace))?;
    m.add_function(wrap_pyfunction!(py_interp))?;
    m.add_function(wrap_pyfunction!(py_dot))?;
    m.add_function(wrap_pyfunction!(py_matmul))?;

    Ok(())
}
