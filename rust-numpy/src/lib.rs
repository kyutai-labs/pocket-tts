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

// Temporarily allow these lints to be fixed incrementally
// TODO: Remove these allows as code is cleaned up (tracked in issue #30)
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::new_without_default)]
#![allow(clippy::module_inception)]
#![allow(clippy::same_item_push)]
#![allow(clippy::cast_sign_loss)]
#![allow(suspicious_double_ref_op)]
#![allow(clippy::inherent_to_string_shadow_display)]
#![allow(dead_code)]

pub mod advanced_broadcast;
pub mod array;
pub mod array_creation;
pub mod array_extra;
pub mod array_manipulation;
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
#[cfg(test)]
mod fft_tests;
pub mod iterator;
pub mod linalg;
pub mod math_ufuncs;
pub mod matrix;
pub mod memory;
pub mod modules;
pub mod random;
pub mod rec;
pub mod set_ops;
pub mod slicing;
pub mod sorting;
pub mod statistics;
pub mod strides;
pub mod type_promotion;
pub mod ufunc;
pub mod ufunc_ops;
pub mod window;

// Re-export key types for convenience
pub use crate::array_extra::exports::*;
pub use crate::comparison_ufuncs::exports::*;
pub use crate::fft::*;
pub use crate::matrix::exports::*;
pub use crate::modules::ma::exports::*;
pub use crate::modules::testing::exports::*;
pub use array::Array;
pub use bitwise::*;
pub use dtype::{Casting, Dtype, DtypeKind};
pub use error::{NumPyError, Result};
pub use linalg::norm;
pub use rec::{array as rec_array, fromarrays, fromrecords, RecArray};
pub use set_ops::exports::*;
pub use statistics::{ptp, std, var};
pub use type_promotion::promote_types;
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
pub use array_creation::{copy, frombuffer, fromfunction, fromiter};
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
            $crate::Array::from_shape_vec(shape.to_vec(), flat)
        }
    };
}

/// Create 3D array macro
#[macro_export]
macro_rules! array3 {
    ($([$([$($expr:expr),*]),*]),*) => {
        {
            // Use nested vecs to collect structure
            let pages = vec![$(
                vec![$(
                    vec![$( $expr ),*]
                ),*]
            ),*];

            let mut flat = Vec::new();
            let mut dim1 = 0;
            let mut dim2 = 0;
            let mut dim3 = 0;

            dim1 = pages.len();
            if dim1 > 0 {
                dim2 = pages[0].len();
                if dim2 > 0 {
                    dim3 = pages[0][0].len();
                }
            }

            for page in pages {
                for row in page {
                    for elem in row {
                        flat.push(elem);
                    }
                }
            }

            let shape = vec![dim1, dim2, dim3];
            $crate::Array::from_shape_vec(shape, flat)
        }
    };
}
