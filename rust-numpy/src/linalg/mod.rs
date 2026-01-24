use num_complex::Complex;
use num_traits::{Float, One, Zero};

pub mod decompositions;
pub mod eigen;
pub mod einsum;
pub mod norms;
pub mod products;
pub mod solvers;

pub trait LinalgScalar:
    Copy
    + Clone
    + Default
    + 'static
    + Zero
    + One
    + std::ops::Neg<Output = Self>
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    type Real: Float;

    fn abs(self) -> Self::Real;
    fn sqrt(self) -> Self;
    fn conj(self) -> Self;
    fn is_positive(self) -> bool;
    fn from_real(r: Self::Real) -> Self;
}

impl LinalgScalar for f32 {
    type Real = f32;

    fn abs(self) -> Self::Real {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn conj(self) -> Self {
        self
    }

    fn is_positive(self) -> bool {
        self > 0.0
    }

    fn from_real(r: Self::Real) -> Self {
        r
    }
}

impl LinalgScalar for f64 {
    type Real = f64;

    fn abs(self) -> Self::Real {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn conj(self) -> Self {
        self
    }

    fn is_positive(self) -> bool {
        self > 0.0
    }

    fn from_real(r: Self::Real) -> Self {
        r
    }
}

impl LinalgScalar for Complex<f32> {
    type Real = f32;

    fn abs(self) -> Self::Real {
        self.norm()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    fn is_positive(self) -> bool {
        self.im == 0.0 && self.re > 0.0
    }

    fn from_real(r: Self::Real) -> Self {
        Complex::new(r, 0.0)
    }
}

impl LinalgScalar for Complex<f64> {
    type Real = f64;

    fn abs(self) -> Self::Real {
        self.norm()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    fn is_positive(self) -> bool {
        self.im == 0.0 && self.re > 0.0
    }

    fn from_real(r: Self::Real) -> Self {
        Complex::new(r, 0.0)
    }
}

// Re-export public API
pub use decompositions::*;
pub use eigen::*;
pub use einsum::{einsum, einsum_path};
pub use norms::*;
pub use products::{
    cross, diagonal, dot, dot_nd, inner, kron, matmul, matrix_power, matrix_transpose, multi_dot,
    outer, tensordot, trace, vdot, vecdot,
};
pub use solvers::{det, inv, lstsq, pinv, solve, tensor_inv, tensor_solve};

/// Alias for NumPyError used in linear algebra operations.
pub use crate::error::NumPyError as LinAlgError;
