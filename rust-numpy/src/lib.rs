//! Rust NumPy - 100% pure-Rust NumPy library with full API parity
//!
//! This library provides complete compatibility with Python's NumPy API,
//! including all modules, functions, and data types.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use numpy::*;
//!
//! let a = array![1, 2, 3, 4, 5];
//! let b = Array::<f64>::zeros(vec![3, 4]);
//! println!("Array a: {:?}", a);
//! println!("Zeros array shape: {:?}", b.shape());
//! ```

pub mod array;
pub mod array_creation;
pub mod array_manipulation;
pub mod advanced_broadcast;
pub mod bitwise;
pub mod broadcasting;
pub mod comparison_ufuncs;
pub mod constants;
pub mod datetime;
pub mod dtype;
#[cfg(test)]
mod dtype_tests;
pub mod error;
pub mod fft;
pub mod io;
pub mod linalg;
pub mod math_ufuncs;
pub mod memory;
pub mod parallel_ops;
pub mod polynomial;
pub mod random;
pub mod set_ops;
pub mod simd_ops;
pub mod slicing;
pub mod sorting;
pub mod strides;
pub mod ufunc;
pub mod ufunc_ops;
pub mod window;

#[cfg(feature = "std")]
// Modules system - structure ready for expansion
#[cfg(feature = "python")]
pub mod python;

// Re-export key types for convenience
pub use array::Array;
pub use array_creation::{arange, array, clip, log, min};
pub use array_manipulation::exports::*;
pub use bitwise::*;
pub use dtype::{Dtype, DtypeKind};
pub use error::{NumPyError, Result};
pub use io::*;
pub use ufunc_ops::UfuncEngine;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default float type (f64)
pub type Float = f64;

/// Default integer type (i64)
pub type Int = i64;

/// Default complex type (Complex<f64>)
pub type Complex = num_complex::Complex<f64>;

// Re-export common constants
pub use constants::*;

/// Create array macro for convenient array creation
#[macro_export]
macro_rules! array {
    ($($expr:expr),*) => {
        {
            let data = [$($expr),*];
            $crate::Array::from_vec(data.to_vec())
        }
    };
}

/// Create 2D array macro
#[macro_export]
macro_rules! array2 {
    ($([$($expr:expr),*]),*) => {
        {
            let rows = [$([$($expr),*],)*];
            let flat: Vec<_> = rows.into_iter().flat_map(|row| row.into_iter()).collect();
            let shape = [rows.len(), if rows.len() > 0 { rows[0].len() } else { 0 }];
            $crate::Array::from_shape_vec(shape.to_vec(), flat).unwrap()
        }
    };
}
