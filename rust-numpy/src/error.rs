use thiserror::Error;

/// Comprehensive error types matching NumPy exceptions
#[derive(Debug, Error, Clone)]
pub enum NumPyError {
    #[error("Array shapes do not match: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Array is not contiguous")]
    NotContiguous,

    #[error("Index {index} is out of bounds for array of size {size}")]
    IndexError { index: usize, size: usize },

    #[error("Cannot cast from {from} to {to}")]
    CastError { from: String, to: String },

    #[error("Division by zero")]
    DivideByZero,

    #[error("Value {value} is out of range for dtype {dtype}")]
    ValueError { value: String, dtype: String },

    #[error("Invalid dtype: {dtype}")]
    InvalidDtype { dtype: String },

    #[error("Invalid operation: {operation}")]
    InvalidOperation { operation: String },

    #[error("Memory allocation failed: requested {size} bytes")]
    MemoryError { size: usize },

    #[error("Operation not implemented: {operation}")]
    NotImplemented { operation: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Backend error: {backend} - {message}")]
    BackendError { backend: String, message: String },

    #[error("UFunc error: {ufunc} - {message}")]
    UfuncError { ufunc: String, message: String },

    #[error("Linear algebra error: {operation} - {message}")]
    LinAlgError { operation: String, message: String },

    #[error("FFT error: {message}")]
    FftError { message: String },

    #[error("Random state error: {message}")]
    RandomError { message: String },

    #[error("Slice error: {message}")]
    SliceError { message: String },

    #[error("Broadcast error: cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    #[error("Stride error: {message}")]
    StrideError { message: String },

    #[error("View error: {message}")]
    ViewError { message: String },

    #[error("File format error: {format} - {message}")]
    FileFormatError { format: String, message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Thread error: {message}")]
    ThreadError { message: String },

    #[error("Polynomial error: {message}")]
    PolynomialError { message: String },

    #[error("Datetime error: {message}")]
    DatetimeError { message: String },

    #[error("Window function error: {message}")]
    WindowError { message: String },
}

impl NumPyError {
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, actual }
    }

    pub fn index_error(index: usize, size: usize) -> Self {
        Self::IndexError { index, size }
    }

    pub fn cast_error(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::CastError {
            from: from.into(),
            to: to.into(),
        }
    }

    pub fn value_error(value: impl Into<String>, dtype: impl Into<String>) -> Self {
        Self::ValueError {
            value: value.into(),
            dtype: dtype.into(),
        }
    }

    pub fn invalid_dtype(dtype: impl Into<String>) -> Self {
        Self::InvalidDtype {
            dtype: dtype.into(),
        }
    }

    /// Dtype validation error (alias for invalid_dtype)
    pub fn dtype_error(message: impl Into<String>) -> Self {
        Self::InvalidDtype {
            dtype: message.into(),
        }
    }

    pub fn memory_error(size: usize) -> Self {
        Self::MemoryError { size }
    }

    pub fn not_implemented(operation: impl Into<String>) -> Self {
        Self::NotImplemented {
            operation: operation.into(),
        }
    }

    pub fn io_error(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    pub fn backend_error(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
        }
    }

    pub fn ufunc_error(ufunc: impl Into<String>, message: impl Into<String>) -> Self {
        Self::UfuncError {
            ufunc: ufunc.into(),
            message: message.into(),
        }
    }

    pub fn linalg_error(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::LinAlgError {
            operation: operation.into(),
            message: message.into(),
        }
    }

    pub fn fft_error(message: impl Into<String>) -> Self {
        Self::FftError {
            message: message.into(),
        }
    }

    pub fn random_error(message: impl Into<String>) -> Self {
        Self::RandomError {
            message: message.into(),
        }
    }

    pub fn slice_error(message: impl Into<String>) -> Self {
        Self::SliceError {
            message: message.into(),
        }
    }

    pub fn broadcast_error(shape1: Vec<usize>, shape2: Vec<usize>) -> Self {
        Self::BroadcastError { shape1, shape2 }
    }

    pub fn stride_error(message: impl Into<String>) -> Self {
        Self::StrideError {
            message: message.into(),
        }
    }

    pub fn view_error(message: impl Into<String>) -> Self {
        Self::ViewError {
            message: message.into(),
        }
    }

    pub fn file_format_error(format: impl Into<String>, message: impl Into<String>) -> Self {
        Self::FileFormatError {
            format: format.into(),
            message: message.into(),
        }
    }

    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    pub fn thread_error(message: impl Into<String>) -> Self {
        Self::ThreadError {
            message: message.into(),
        }
    }

    pub fn invalid_operation(operation: impl Into<String>) -> Self {
        Self::InvalidOperation {
            operation: operation.into(),
        }
    }

    pub fn from_linalg_error<E: std::fmt::Display>(err: E) -> Self {
        Self::LinAlgError {
            operation: "linalg_operation".to_string(),
            message: err.to_string(),
        }
    }

    pub fn polynomial_error(message: impl Into<String>) -> Self {
        Self::PolynomialError {
            message: message.into(),
        }
    }

    pub fn datetime_error(message: impl Into<String>) -> Self {
        Self::DatetimeError {
            message: message.into(),
        }
    }

    pub fn window_error(message: impl Into<String>) -> Self {
        Self::WindowError {
            message: message.into(),
        }
    }

    /// Simple value error with just a message (dtype defaults to "value")
    pub fn invalid_value(message: impl Into<String>) -> Self {
        Self::ValueError {
            value: message.into(),
            dtype: "value".to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, NumPyError>;

impl From<std::io::Error> for NumPyError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            message: err.to_string(),
        }
    }
}

// impl From<std::alloc::AllocError> for NumPyError {
//     fn from(_: std::alloc::AllocError) -> Self {
//         Self::MemoryError { size: 0 }
//     }
// }

#[cfg(feature = "blas")]
impl From<openblas_src::Error> for NumPyError {
    fn from(err: openblas_src::Error) -> Self {
        Self::BackendError {
            backend: "OpenBLAS".to_string(),
            message: err.to_string(),
        }
    }
}
