// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Mathematical universal functions (ufuncs)
//!
//! This module provides comprehensive mathematical operations including:
//! - Trigonometric functions (sin, cos, tan, etc.)
//! - Hyperbolic functions (sinh, cosh, tanh, etc.)
//! - Exponential and logarithmic functions (exp, log, etc.)
//! - Rounding functions (round, floor, ceil, etc.)

use crate::array::Array;
use crate::broadcasting::{broadcast_arrays, compute_broadcast_shape};
use crate::dtype::DtypeKind;
use crate::error::{NumPyError, Result};
use crate::ufunc::{get_ufunc, get_ufunc_typed, Ufunc};
use num_traits::{FloatConst, Zero};
use std::f64::consts;
use std::marker::PhantomData;

/// Trait for trigonometric operations
pub trait TrigOps<T>: Send + Sync {
    fn sin(&self) -> T;
    fn cos(&self) -> T;
    fn tan(&self) -> T;
    fn arcsin(&self) -> Result<T>;
    fn arccos(&self) -> Result<T>;
    fn arctan(&self) -> T;
    fn arctan2(&self, other: &T) -> T;
    fn hypot(&self, other: &T) -> T;
    fn degrees(&self) -> T;
    fn radians(&self) -> T;
}

/// Trait for hyperbolic operations
pub trait HyperbolicOps<T>: Send + Sync {
    fn sinh(&self) -> T;
    fn cosh(&self) -> T;
    fn tanh(&self) -> T;
    fn arcsinh(&self) -> T;
    fn arccosh(&self) -> Result<T>;
    fn arctanh(&self) -> Result<T>;
}

/// Trait for exponential and logarithmic operations
pub trait ExpLogOps<T>: Send + Sync {
    fn exp(&self) -> T;
    fn exp2(&self) -> T;
    fn expm1(&self) -> T;
    fn log(&self) -> Result<T>;
    fn log2(&self) -> Result<T>;
    fn log10(&self) -> Result<T>;
    fn log1p(&self) -> T;
    fn logaddexp(&self, other: &T) -> T;
    fn logaddexp2(&self, other: &T) -> T;
}

/// Trait for rounding operations
pub trait RoundingOps<T>: Send + Sync {
    fn round(&self, decimals: isize) -> T;
    fn rint(&self) -> T;
    fn floor(&self) -> T;
    fn ceil(&self) -> T;
    fn trunc(&self) -> T;
    fn fix(&self) -> T;
}

/// Mathematical unary ufunc
pub struct MathUnaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> MathUnaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> T + Send + Sync,
{
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for MathUnaryUfunc<T, F>
where
    T: Clone + 'static + Send + Sync,
    F: Fn(&T) -> T + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn nin(&self) -> usize {
        1
    }

    fn nout(&self) -> usize {
        1
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Float, DtypeKind::Complex]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 1 && input_types[0] == std::any::type_name::<T>()
    }

    fn input_dtypes(&self) -> Vec<crate::dtype::Dtype> {
        vec![crate::dtype::Dtype::from_type::<T>()]
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                self.name(),
                format!(
                    "Expected 1 input, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input = inputs[0]
            .as_any()
            .downcast_ref::<Array<T>>()
            .ok_or_else(|| NumPyError::invalid_operation("Failed to downcast input array"))?;
        let output = outputs[0]
            .as_any_mut()
            .downcast_mut::<Array<T>>()
            .ok_or_else(|| NumPyError::invalid_operation("Failed to downcast output array"))?;

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        for i in 0..input.size() {
            if mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let Some(a) = input.get(i) {
                    let result = (self.operation)(a);
                    output.set(i, result)?;
                }
            }
        }

        Ok(())
    }
}

/// Mathematical binary ufunc
pub struct MathBinaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T, &T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> MathBinaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T, &T) -> T + Send + Sync,
{
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for MathBinaryUfunc<T, F>
where
    T: Clone + 'static + Send + Sync + Default,
    F: Fn(&T, &T) -> T + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn nin(&self) -> usize {
        2
    }

    fn nout(&self) -> usize {
        1
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Float, DtypeKind::Complex]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 2 && input_types.iter().all(|&t| t == std::any::type_name::<T>())
    }

    fn input_dtypes(&self) -> Vec<crate::dtype::Dtype> {
        vec![
            crate::dtype::Dtype::from_type::<T>(),
            crate::dtype::Dtype::from_type::<T>(),
        ]
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                self.name(),
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input0 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let input1 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        let broadcasted = broadcast_arrays(&[input0, input1])?;
        let arr0 = &broadcasted[0];
        let arr1 = &broadcasted[1];

        for i in 0..output.size() {
            if mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let (Some(a), Some(b)) = (arr0.get(i), arr1.get(i)) {
                    let result = (self.operation)(a, b);
                    output.set(i, result)?;
                }
            }
        }

        Ok(())
    }
}

/// Special unary ufunc for operations that can return errors
pub struct MathUnaryUfuncWithResult<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> Result<T> + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> MathUnaryUfuncWithResult<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> Result<T> + Send + Sync,
{
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for MathUnaryUfuncWithResult<T, F>
where
    T: Clone + 'static + Send + Sync,
    F: Fn(&T) -> Result<T> + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn nin(&self) -> usize {
        1
    }

    fn nout(&self) -> usize {
        1
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Float, DtypeKind::Complex]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 1 && input_types[0] == std::any::type_name::<T>()
    }

    fn input_dtypes(&self) -> Vec<crate::dtype::Dtype> {
        vec![crate::dtype::Dtype::from_type::<T>()]
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                self.name(),
                format!(
                    "Expected 1 input, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input = inputs[0]
            .as_any()
            .downcast_ref::<Array<T>>()
            .ok_or_else(|| NumPyError::invalid_operation("Failed to downcast input array"))?;
        let output = outputs[0]
            .as_any_mut()
            .downcast_mut::<Array<T>>()
            .ok_or_else(|| NumPyError::invalid_operation("Failed to downcast output array"))?;

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        for i in 0..input.size() {
            if mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let Some(a) = input.get(i) {
                    let result = (self.operation)(a)?;
                    output.set(i, result)?;
                }
            }
        }

        Ok(())
    }
}

// Macro to implement trigonometric operations for floating point types
macro_rules! impl_trig_ops_float {
    ($($t:ty),*) => {
        $(
            impl TrigOps<$t> for $t {
                fn sin(&self) -> $t {
                    num_traits::Float::sin(*self)
                }

                fn cos(&self) -> $t {
                    num_traits::Float::cos(*self)
                }

                fn tan(&self) -> $t {
                    num_traits::Float::tan(*self)
                }

                fn arcsin(&self) -> Result<$t> {
                    if *self < -1.0 || *self > 1.0 {
                        return Err(NumPyError::value_error(format!("arcsin domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    Ok(self.asin())
                }

                fn arccos(&self) -> Result<$t> {
                    if *self < -1.0 || *self > 1.0 {
                        return Err(NumPyError::value_error(format!("arccos domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    Ok(self.acos())
                }

                fn arctan(&self) -> $t {
                    self.atan()
                }

                fn arctan2(&self, other: &$t) -> $t {
                    self.atan2(*other)
                }

                fn hypot(&self, other: &$t) -> $t {
                    num_traits::Float::hypot(*self, *other)
                }

                fn degrees(&self) -> $t {
                    *self * (180.0 / consts::PI) as $t
                }

                fn radians(&self) -> $t {
                    *self * (consts::PI / 180.0) as $t
                }
            }

            impl HyperbolicOps<$t> for $t {
                fn sinh(&self) -> $t {
                    num_traits::Float::sinh(*self)
                }

                fn cosh(&self) -> $t {
                    num_traits::Float::cosh(*self)
                }

                fn tanh(&self) -> $t {
                    num_traits::Float::tanh(*self)
                }

                fn arcsinh(&self) -> $t {
                    self.asinh()
                }

                fn arccosh(&self) -> Result<$t> {
                    if *self < 1.0 {
                        return Err(NumPyError::value_error(format!("arccosh domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    Ok(self.acosh())
                }

                fn arctanh(&self) -> Result<$t> {
                    if *self <= -1.0 || *self >= 1.0 {
                        return Err(NumPyError::value_error(format!("arctanh domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    Ok(self.atanh())
                }
            }

            impl ExpLogOps<$t> for $t {
                fn exp(&self) -> $t {
                    num_traits::Float::exp(*self)
                }

                fn exp2(&self) -> $t {
                    num_traits::Float::exp2(*self)
                }

                fn expm1(&self) -> $t {
                    <$t as num_traits::Float>::exp_m1(*self)
                }

                fn log(&self) -> Result<$t> {
                    if *self <= 0.0 {
                        return Err(NumPyError::value_error(format!("log domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    Ok(self.ln())
                }

                fn log2(&self) -> Result<$t> {
                    if *self <= 0.0 {
                        return Err(NumPyError::value_error(format!("log2 domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    self.log2()
                }

                fn log10(&self) -> Result<$t> {
                    if *self <= 0.0 {
                        return Err(NumPyError::value_error(format!("log10 domain error: {}", self),
                            "float64".to_string(),  ));
                    }
                    self.log10()
                }

                fn log1p(&self) -> $t {
                    self.ln_1p()
                }

                fn logaddexp(&self, other: &$t) -> $t {
                    let max_val = if *self > *other { *self } else { *other };
                    let min_val = if *self > *other { *other } else { *self };
                    max_val + (1.0 + (-min_val + max_val).exp()).ln()
                }

                fn logaddexp2(&self, other: &$t) -> $t {
                    let max_val = if *self > *other { *self } else { *other };
                    let min_val = if *self > *other { *other } else { *self };
                    max_val + (1.0 + (-min_val + max_val).exp2()).log2()
                }
            }

            impl RoundingOps<$t> for $t {
                fn round(&self, decimals: isize) -> $t {
                    if decimals == 0 {
                        num_traits::Float::round(*self)
                    } else {
                        let factor = 10.0_f64.powi(decimals as i32) as $t;
                        (self * factor).round() / factor
                    }
                }

                fn rint(&self) -> $t {
                    num_traits::Float::round(*self)
                }

                fn floor(&self) -> $t {
                    num_traits::Float::floor(*self)
                }

                fn ceil(&self) -> $t {
                    num_traits::Float::ceil(*self)
                }

                fn trunc(&self) -> $t {
                    num_traits::Float::trunc(*self)
                }

                fn fix(&self) -> $t {
                    if *self >= 0.0 {
                        self.floor()
                    } else {
                        self.ceil()
                    }
                }
            }
        )*
    }
}

// Macro to implement trigonometric operations for complex types
macro_rules! impl_trig_ops_complex {
    ($($t:ty),*) => {
        $(
            impl TrigOps<$t> for $t {
                fn sin(&self) -> $t {
                    num_complex::Complex::sin(*self)
                }

                fn cos(&self) -> $t {
                    num_complex::Complex::cos(*self)
                }

                fn tan(&self) -> $t {
                    num_complex::Complex::tan(*self)
                }

                fn arcsin(&self) -> Result<$t> {
                    Ok(-<$t>::i() * (<$t>::i() * self + (1.0 - self * self).sqrt()).ln())
                }

                fn arccos(&self) -> Result<$t> {
                    Ok(-<$t>::i() * (self + <$t>::i() * (1.0 - self * self).sqrt()).ln())
                }

                fn arctan(&self) -> $t {
                    num_complex::Complex::new(0.0, 0.5) * ((num_complex::Complex::new(0.0, 1.0) + self) / (num_complex::Complex::new(0.0, 1.0) - self)).ln()
                }

                fn arctan2(&self, other: &$t) -> $t {
                    self.arctan() - other.arctan()
                }

                fn hypot(&self, other: &$t) -> $t {
                    (self * self + other * other).sqrt()
                }

                fn degrees(&self) -> $t {
                     let pi = <$t as num_complex::ComplexFloat>::Real::PI();
                     let val_180: <$t as num_complex::ComplexFloat>::Real = num_traits::NumCast::from(180.0).unwrap();
                     let factor = val_180 / pi;
                     (self * <$t>::new(factor, <$t as num_complex::ComplexFloat>::Real::zero())).sin()
                }

                fn radians(&self) -> $t {
                     let pi = <$t as num_complex::ComplexFloat>::Real::PI();
                     let val_180: <$t as num_complex::ComplexFloat>::Real = num_traits::NumCast::from(180.0).unwrap();
                     let factor = pi / val_180;
                     (self * <$t>::new(factor, <$t as num_complex::ComplexFloat>::Real::zero())).asin()
                }
            }

            impl HyperbolicOps<$t> for $t {
                fn sinh(&self) -> $t {
                    num_complex::Complex::sinh(*self)
                }

                fn cosh(&self) -> $t {
                    num_complex::Complex::cosh(*self)
                }

                fn tanh(&self) -> $t {
                    num_complex::Complex::tanh(*self)
                }

                fn arcsinh(&self) -> $t {
                    (self + (self * self + 1.0).sqrt()).ln()
                }

                fn arccosh(&self) -> Result<$t> {
                    Ok((self + (self - 1.0).sqrt() * (self + 1.0).sqrt()).ln())
                }

                fn arctanh(&self) -> Result<$t> {
                    Ok(0.5 * ((1.0 + self) / (1.0 - self)).ln())
                }
            }

            impl ExpLogOps<$t> for $t {
                fn exp(&self) -> $t {
                    num_complex::Complex::exp(*self)
                }

                fn exp2(&self) -> $t {
                    let ln2: <$t as num_complex::ComplexFloat>::Real = num_traits::NumCast::from(std::f64::consts::LN_2).unwrap();
                    (self * <$t>::new(ln2, <$t as num_complex::ComplexFloat>::Real::zero())).exp()
                }

                fn expm1(&self) -> $t {
                    self.exp() - 1.0
                }

                fn log(&self) -> Result<$t> {
                    Ok(self.ln())
                }

                fn log2(&self) -> Result<$t> {
                    let ln2: <$t as num_complex::ComplexFloat>::Real = num_traits::NumCast::from(std::f64::consts::LN_2).unwrap();
                    Ok(self.ln() / <$t>::new(ln2, <$t as num_complex::ComplexFloat>::Real::zero()))
                }

                fn log10(&self) -> Result<$t> {
                    let ln10: <$t as num_complex::ComplexFloat>::Real = num_traits::NumCast::from(std::f64::consts::LN_10).unwrap();
                    Ok(self.ln() / <$t>::new(ln10, <$t as num_complex::ComplexFloat>::Real::zero()))
                }

                fn log1p(&self) -> $t {
                    (*self + 1.0).ln()
                }

                fn logaddexp(&self, other: &$t) -> $t {
                    let max_val = if self.norm() > other.norm() { *self } else { *other };
                    let min_val = if self.norm() > other.norm() { *other } else { *self };
                    max_val + (1.0 + (min_val - max_val).exp()).ln()
                }

                fn logaddexp2(&self, other: &$t) -> $t {
                    let max_val = if self.norm() > other.norm() { *self } else { *other };
                    let min_val = if self.norm() > other.norm() { *other } else { *self };
                    max_val + (1.0 + (min_val - max_val).exp2()).log2()
                }
            }

            impl RoundingOps<$t> for $t {
                fn round(&self, decimals: isize) -> $t {
                    let factor = 10.0_f64.powi(decimals as i32);
                    let factor_real: <$t as num_complex::ComplexFloat>::Real = num_traits::NumCast::from(factor).unwrap();
                    num_complex::Complex::new(
                        (self.re * factor_real).round() / factor_real,
                        (self.im * factor_real).round() / factor_real,
                    )
                }

                fn rint(&self) -> $t {
                    self.round(0)
                }

                fn floor(&self) -> $t {
                    num_complex::Complex::new(self.re.floor(), self.im.floor())
                }

                fn ceil(&self) -> $t {
                    num_complex::Complex::new(self.re.ceil(), self.im.ceil())
                }

                fn trunc(&self) -> $t {
                    num_complex::Complex::new(self.re.trunc(), self.im.trunc())
                }

                fn fix(&self) -> $t {
                    num_complex::Complex::new(
                        if self.re >= 0.0 { self.re.floor() } else { self.re.ceil() },
                        if self.im >= 0.0 { self.im.floor() } else { self.im.ceil() },
                    )
                }
            }
        )*
    }
}

// Implement operations for floating point types
impl_trig_ops_float!(f32, f64);

// Implement operations for complex types
impl_trig_ops_complex!(num_complex::Complex32, num_complex::Complex64);

/// Compute sine of each element
#[cfg(feature = "simd")]
pub fn sin<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    #[cfg(target_arch = "x86_64")]
    {
        return sin_simd(x);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        return sin_scalar(x);
    }
}

#[cfg(not(feature = "simd"))]
pub fn sin<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    sin_scalar(x)
}

/// Compute sine using SIMD (AVX2/SSE) on x86_64
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
fn sin_simd<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    use crate::simd_ops;

    if let Some(ufunc) = get_ufunc("sin") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "sin",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute sine using scalar operations (non-SIMD or fallback)
fn sin_scalar<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("sin") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "sin",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute cosine of each element
#[cfg(feature = "simd")]
pub fn cos<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    #[cfg(target_arch = "x86_64")]
    {
        return cos_simd(x);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        return cos_scalar(x);
    }
}

#[cfg(not(feature = "simd"))]
pub fn cos<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    cos_scalar(x)
}

/// Compute cosine using SIMD (AVX2/SSE) on x86_64
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
fn cos_simd<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("cos") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "cos",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute cosine using scalar operations (non-SIMD or fallback)
fn cos_scalar<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("cos") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "cos",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute tangent of each element
pub fn tan<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("tan") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "tan",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute arcsine of each element
pub fn arcsin<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arcsin") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arcsin",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute arccosine of each element
pub fn arccos<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arccos") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arccos",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute arctangent of each element
pub fn arctan<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arctan") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arctan",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute arctangent of y/x element-wise
pub fn arctan2<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arctan2") {
        let broadcast_shape = compute_broadcast_shape(x1.shape(), x2.shape());
        let mut output = Array::from_data(
            vec![T::default(); broadcast_shape.iter().product::<usize>()],
            broadcast_shape,
        );
        ufunc.execute(&[x1, x2], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arctan2",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute hypotenuse element-wise
pub fn hypot<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("hypot") {
        let broadcast_shape = compute_broadcast_shape(x1.shape(), x2.shape());
        let mut output = Array::from_data(
            vec![T::default(); broadcast_shape.iter().product::<usize>()],
            broadcast_shape,
        );
        ufunc.execute(&[x1, x2], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "hypot",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Convert angles from radians to degrees
pub fn degrees<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("degrees") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "degrees",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Convert angles from degrees to radians
pub fn radians<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + TrigOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("radians") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "radians",
            "Ufunc not found".to_string(),
        ))
    }
}

// Hyperbolic functions

/// Compute hyperbolic sine of each element
pub fn sinh<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + HyperbolicOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("sinh") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "sinh",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute hyperbolic cosine of each element
pub fn cosh<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + HyperbolicOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("cosh") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "cosh",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute hyperbolic tangent of each element
pub fn tanh<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + HyperbolicOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("tanh") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "tanh",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute inverse hyperbolic sine of each element
pub fn arcsinh<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + HyperbolicOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arcsinh") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arcsinh",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute inverse hyperbolic cosine of each element
pub fn arccosh<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + HyperbolicOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arccosh") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arccosh",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute inverse hyperbolic tangent of each element
pub fn arctanh<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + HyperbolicOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("arctanh") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "arctanh",
            "Ufunc not found".to_string(),
        ))
    }
}

// Exponential and logarithmic functions

/// Compute exponential of each element
#[cfg(feature = "simd")]
pub fn exp<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    #[cfg(target_arch = "x86_64")]
    {
        return exp_simd(x);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        return exp_scalar(x);
    }
}

#[cfg(not(feature = "simd"))]
pub fn exp<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    exp_scalar(x)
}

/// Compute exponential using SIMD (AVX2) on x86_64
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
fn exp_simd<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("exp") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "exp",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute exponential using scalar operations (non-SIMD or fallback)
fn exp_scalar<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("exp") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "exp",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute 2**x of each element
pub fn exp2<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("exp2") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "exp2",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute exp(x) - 1 of each element
pub fn expm1<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("expm1") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "expm1",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute natural logarithm of each element
#[cfg(feature = "simd")]
pub fn log<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    #[cfg(target_arch = "x86_64")]
    {
        return log_simd(x);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        return log_scalar(x);
    }
}

#[cfg(not(feature = "simd"))]
pub fn log<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    log_scalar(x)
}

/// Compute logarithm using SIMD (AVX2) on x86_64
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
fn log_simd<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("log") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "log",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute logarithm using scalar operations (non-SIMD or fallback)
fn log_scalar<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("log") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "log",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute base-2 logarithm of each element
pub fn log2<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("log2") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "log2",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute base-10 logarithm of each element
pub fn log10<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("log10") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "log10",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute log(1 + x) of each element
pub fn log1p<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("log1p") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "log1p",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute log(exp(x1) + exp(x2)) element-wise
pub fn logaddexp<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("logaddexp") {
        let broadcast_shape = compute_broadcast_shape(x1.shape(), x2.shape());
        let mut output = Array::from_data(
            vec![T::default(); broadcast_shape.iter().product::<usize>()],
            broadcast_shape,
        );
        ufunc.execute(&[x1, x2], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "logaddexp",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute log2(exp2(x1) + exp2(x2)) element-wise
pub fn logaddexp2<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + ExpLogOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("logaddexp2") {
        let broadcast_shape = compute_broadcast_shape(x1.shape(), x2.shape());
        let mut output = Array::from_data(
            vec![T::default(); broadcast_shape.iter().product::<usize>()],
            broadcast_shape,
        );
        ufunc.execute(&[x1, x2], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "logaddexp2",
            "Ufunc not found".to_string(),
        ))
    }
}

// Rounding functions

/// Round array to given number of decimals
pub fn round_<T>(a: &Array<T>, decimals: isize) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    let mut output = Array::from_data(vec![T::default(); a.size()], a.shape().to_vec());

    for i in 0..a.size() {
        if let Some(val) = a.get(i) {
            let rounded = val.round(decimals);
            output.set(i, rounded)?;
        }
    }

    Ok(output)
}

/// Alias for round_ function (NumPy compatibility)
pub fn around<T>(a: &Array<T>, decimals: isize) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    round_(a, decimals)
}

/// Round to nearest integer
pub fn rint<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("rint") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "rint",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Floor of each element
pub fn floor<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("floor") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "floor",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Ceiling of each element
pub fn ceil<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("ceil") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "ceil",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Truncate each element toward zero
pub fn trunc<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("trunc") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "trunc",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Round each element toward zero
pub fn fix<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + RoundingOps<T> + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("fix") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "fix",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute the absolute value element-wise
pub fn absolute<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Signed + 'static,
{
    let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            output.set(i, val.abs())?;
        }
    }
    Ok(output)
}

/// Alias for absolute (NumPy compatibility)
pub fn fabs<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            output.set(i, val.abs())?;
        }
    }
    Ok(output)
}

/// Returns an element-wise indication of the sign of a number
pub fn sign<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Signed + PartialOrd + 'static,
{
    let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            let s = if *val > T::zero() {
                T::one()
            } else if *val < T::zero() {
                T::zero() - T::one()
            } else {
                T::zero()
            };
            output.set(i, s)?;
        }
    }
    Ok(output)
}

/// Returns element-wise true where signbit is set (less than zero)
pub fn signbit<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let mut output = Array::from_data(vec![false; x.size()], x.shape().to_vec());
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            // Using signum() and checking for negative is more robust for some float implementations
            // or just use is_sign_negative() if it's reliable.
            // Let's use a combination or check bit directly if possible.
            // NumPy's signbit is true if the sign bit is set.
            output.set(i, val.is_sign_negative())?;
        }
    }
    Ok(output)
}

/// Change the sign of x1 to that of x2, element-wise
pub fn copysign<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let broadcast_shape = compute_broadcast_shape(x1.shape(), x2.shape());
    let mut output = Array::from_data(
        vec![T::default(); broadcast_shape.iter().product()],
        broadcast_shape.clone(),
    );

    for i in 0..output.size() {
        let multi_idx = crate::strides::compute_multi_indices(i, &broadcast_shape);
        if let (Ok(v1), Ok(v2)) = (x1.get_by_indices(&multi_idx), x2.get_by_indices(&multi_idx)) {
            output.set(i, v1.copysign(*v2))?;
        }
    }
    Ok(output)
}

// Floating-point checking functions

/// Floating-point checking ufunc that returns boolean array
pub struct FloatCheckUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> bool + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> FloatCheckUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> bool + Send + Sync,
{
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for FloatCheckUfunc<T, F>
where
    T: Clone + 'static + Send + Sync,
    F: Fn(&T) -> bool + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn nin(&self) -> usize {
        1
    }

    fn nout(&self) -> usize {
        1
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Float, DtypeKind::Complex]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 1 && input_types[0] == std::any::type_name::<T>()
    }

    fn input_dtypes(&self) -> Vec<crate::dtype::Dtype> {
        vec![crate::dtype::Dtype::from_type::<T>()]
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
        _where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                self.name(),
                format!(
                    "Expected 1 input, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<bool>) };

        for i in 0..input.size() {
            if let Some(a) = input.get(i) {
                let result = (self.operation)(a);
                output.set(i, result)?;
            }
        }

        Ok(())
    }
}

/// Test element-wise for NaN and return result as a boolean array
pub fn isnan<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + 'static + Send + Sync,
{
    if let Some(ufunc) = get_ufunc_typed::<T>("isnan") {
        let mut output = Array::from_data(vec![false; x.size()], x.shape().to_vec());
        let x_ref: &dyn crate::ufunc::ArrayView = x;
        let out_ref: &mut dyn crate::ufunc::ArrayViewMut = &mut output;
        ufunc.execute(&[x_ref], &mut [out_ref], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "isnan",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Test element-wise for positive or negative infinity and return result as a boolean array
pub fn isinf<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + 'static + Send + Sync,
{
    if let Some(ufunc) = get_ufunc_typed::<T>("isinf") {
        let mut output = Array::from_data(vec![false; x.size()], x.shape().to_vec());
        let x_ref: &dyn crate::ufunc::ArrayView = x;
        let out_ref: &mut dyn crate::ufunc::ArrayViewMut = &mut output;
        ufunc.execute(&[x_ref], &mut [out_ref], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "isinf",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Test element-wise for finiteness (not NaN or infinity) and return result as a boolean array
pub fn isfinite<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + 'static + Send + Sync,
{
    if let Some(ufunc) = get_ufunc_typed::<T>("isfinite") {
        let mut output = Array::from_data(vec![false; x.size()], x.shape().to_vec());
        let x_ref: &dyn crate::ufunc::ArrayView = x;
        let out_ref: &mut dyn crate::ufunc::ArrayViewMut = &mut output;
        ufunc.execute(&[x_ref], &mut [out_ref], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "isfinite",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Test element-wise for negative infinity and return result as a boolean array
pub fn isneginf<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + 'static + Send + Sync,
{
    if let Some(ufunc) = get_ufunc_typed::<T>("isneginf") {
        let mut output = Array::from_data(vec![false; x.size()], x.shape().to_vec());
        let x_ref: &dyn crate::ufunc::ArrayView = x;
        let out_ref: &mut dyn crate::ufunc::ArrayViewMut = &mut output;
        ufunc.execute(&[x_ref], &mut [out_ref], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "isneginf",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Test element-wise for positive infinity and return result as a boolean array
pub fn isposinf<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + 'static + Send + Sync,
{
    if let Some(ufunc) = get_ufunc_typed::<T>("isposinf") {
        let mut output = Array::from_data(vec![false; x.size()], x.shape().to_vec());
        let x_ref: &dyn crate::ufunc::ArrayView = x;
        let out_ref: &mut dyn crate::ufunc::ArrayViewMut = &mut output;
        ufunc.execute(&[x_ref], &mut [out_ref], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "isposinf",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Return the phase angle (also called argument) of a complex number
/// For real input, the result is 0 for positive numbers and pi for negative numbers
pub fn angle(z: &Array<num_complex::Complex64>) -> Result<Array<f64>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(val.im.atan2(val.re));
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the phase angle for Complex32 arrays
pub fn angle32(z: &Array<num_complex::Complex32>) -> Result<Array<f32>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(val.im.atan2(val.re));
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the real part of the Complex64 number
pub fn real(z: &Array<num_complex::Complex64>) -> Result<Array<f64>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(val.re);
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the real part of the Complex32 number
pub fn real32(z: &Array<num_complex::Complex32>) -> Result<Array<f32>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(val.re);
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the imaginary part of the Complex64 number
pub fn imag(z: &Array<num_complex::Complex64>) -> Result<Array<f64>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(val.im);
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the imaginary part of the Complex32 number
pub fn imag32(z: &Array<num_complex::Complex32>) -> Result<Array<f32>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(val.im);
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the complex conjugate, element-wise
/// The conjugate of a complex number is obtained by changing the sign of its imaginary part
pub fn conj(z: &Array<num_complex::Complex64>) -> Result<Array<num_complex::Complex64>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(num_complex::Complex64::new(val.re, -val.im));
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the complex conjugate for Complex32 arrays
pub fn conj32(z: &Array<num_complex::Complex32>) -> Result<Array<num_complex::Complex32>> {
    let mut data = Vec::with_capacity(z.size());
    for i in 0..z.size() {
        if let Some(val) = z.get(i) {
            data.push(num_complex::Complex32::new(val.re, -val.im));
        }
    }
    Ok(Array::from_data(data, z.shape().to_vec()))
}

/// Return the complex conjugate, element-wise (alias for conj)
pub fn conjugate(z: &Array<num_complex::Complex64>) -> Result<Array<num_complex::Complex64>> {
    conj(z)
}

/// Return the complex conjugate for Complex32 arrays (alias for conj32)
pub fn conjugate32(z: &Array<num_complex::Complex32>) -> Result<Array<num_complex::Complex32>> {
    conj32(z)
}

/// Return the real part of the array if the imaginary part is close to zero
/// If the imaginary part is not close to zero, return the array as is
pub fn real_if_close(
    a: &Array<num_complex::Complex64>,
    tol: Option<f64>,
) -> Result<Array<num_complex::Complex64>> {
    let tolerance = tol.unwrap_or(1e-10);
    let mut data = Vec::with_capacity(a.size());

    for i in 0..a.size() {
        if let Some(val) = a.get(i) {
            if val.im.abs() < tolerance {
                // Return real part as a complex number with zero imaginary part
                data.push(num_complex::Complex64::new(val.re, 0.0));
            } else {
                data.push(val.clone());
            }
        }
    }
    Ok(Array::from_data(data, a.shape().to_vec()))
}

/// Return the real part of the array if the imaginary part is close to zero (Complex32 version)
pub fn real_if_close32(
    a: &Array<num_complex::Complex32>,
    tol: Option<f32>,
) -> Result<Array<num_complex::Complex32>> {
    let tolerance = tol.unwrap_or(1e-10);
    let mut data = Vec::with_capacity(a.size());

    for i in 0..a.size() {
        if let Some(val) = a.get(i) {
            if val.im.abs() < tolerance {
                data.push(num_complex::Complex32::new(val.re, 0.0));
            } else {
                data.push(val.clone());
            }
        }
    }
    Ok(Array::from_data(data, a.shape().to_vec()))
}

/// Register all mathematical ufuncs
pub fn register_math_ufuncs(registry: &mut crate::ufunc::UfuncRegistry) {
    // Trigonometric functions
    registry.register(Box::new(MathUnaryUfunc::new("sin", |x: &f32| x.sin())));
    registry.register(Box::new(MathUnaryUfunc::new("sin", |x: &f64| x.sin())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "sin",
        |x: &num_complex::Complex32| x.sin(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "sin",
        |x: &num_complex::Complex64| x.sin(),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("cos", |x: &f32| x.cos())));
    registry.register(Box::new(MathUnaryUfunc::new("cos", |x: &f64| x.cos())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "cos",
        |x: &num_complex::Complex32| x.cos(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "cos",
        |x: &num_complex::Complex64| x.cos(),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("tan", |x: &f32| x.tan())));
    registry.register(Box::new(MathUnaryUfunc::new("tan", |x: &f64| x.tan())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "tan",
        |x: &num_complex::Complex32| x.tan(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "tan",
        |x: &num_complex::Complex64| x.tan(),
    )));

    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arcsin",
        |x: &f32| {
            if *x < -1.0 || *x > 1.0 {
                Err(NumPyError::value_error(
                    format!("arcsin domain error: {}", x),
                    "float32".to_string(),
                ))
            } else {
                Ok(x.asin())
            }
        },
    )));
    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arcsin",
        |x: &f64| {
            if *x < -1.0 || *x > 1.0 {
                Err(NumPyError::value_error(
                    format!("arcsin domain error: {}", x),
                    "float64".to_string(),
                ))
            } else {
                Ok(x.asin())
            }
        },
    )));

    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arccos",
        |x: &f32| {
            if *x < -1.0 || *x > 1.0 {
                Err(NumPyError::value_error(
                    format!("arccos domain error: {}", x),
                    "float32".to_string(),
                ))
            } else {
                Ok(x.acos())
            }
        },
    )));
    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arccos",
        |x: &f64| {
            if *x < -1.0 || *x > 1.0 {
                Err(NumPyError::value_error(
                    format!("arccos domain error: {}", x),
                    "float64".to_string(),
                ))
            } else {
                Ok(x.acos())
            }
        },
    )));

    registry.register(Box::new(MathUnaryUfunc::new("arctan", |x: &f32| x.atan())));
    registry.register(Box::new(MathUnaryUfunc::new("arctan", |x: &f64| x.atan())));

    registry.register(Box::new(MathBinaryUfunc::new(
        "arctan2",
        |x: &f32, y: &f32| x.atan2(*y),
    )));
    registry.register(Box::new(MathBinaryUfunc::new(
        "arctan2",
        |x: &f64, y: &f64| x.atan2(*y),
    )));

    registry.register(Box::new(MathBinaryUfunc::new(
        "hypot",
        |x: &f32, y: &f32| (*x).hypot(*y),
    )));
    registry.register(Box::new(MathBinaryUfunc::new(
        "hypot",
        |x: &f64, y: &f64| (*x).hypot(*y),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("degrees", |x: &f32| {
        x * (180.0 / consts::PI as f32)
    })));
    registry.register(Box::new(MathUnaryUfunc::new("degrees", |x: &f64| {
        x * (180.0 / consts::PI)
    })));

    registry.register(Box::new(MathUnaryUfunc::new("radians", |x: &f32| {
        x * (consts::PI as f32 / 180.0)
    })));
    registry.register(Box::new(MathUnaryUfunc::new("radians", |x: &f64| {
        x * (consts::PI / 180.0)
    })));

    // Hyperbolic functions
    registry.register(Box::new(MathUnaryUfunc::new("sinh", |x: &f32| x.sinh())));
    registry.register(Box::new(MathUnaryUfunc::new("sinh", |x: &f64| x.sinh())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "sinh",
        |x: &num_complex::Complex32| x.sinh(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "sinh",
        |x: &num_complex::Complex64| x.sinh(),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("cosh", |x: &f32| x.cosh())));
    registry.register(Box::new(MathUnaryUfunc::new("cosh", |x: &f64| x.cosh())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "cosh",
        |x: &num_complex::Complex32| x.cosh(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "cosh",
        |x: &num_complex::Complex64| x.cosh(),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("tanh", |x: &f32| x.tanh())));
    registry.register(Box::new(MathUnaryUfunc::new("tanh", |x: &f64| x.tanh())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "tanh",
        |x: &num_complex::Complex32| x.tanh(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "tanh",
        |x: &num_complex::Complex64| x.tanh(),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("arcsinh", |x: &f32| {
        x.asinh()
    })));
    registry.register(Box::new(MathUnaryUfunc::new("arcsinh", |x: &f64| {
        x.asinh()
    })));

    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arccosh",
        |x: &f32| {
            if *x < 1.0 {
                Err(NumPyError::value_error(
                    format!("arccosh domain error: {}", x),
                    "float32".to_string(),
                ))
            } else {
                Ok(x.acosh())
            }
        },
    )));
    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arccosh",
        |x: &f64| {
            if *x < 1.0 {
                Err(NumPyError::value_error(
                    format!("arccosh domain error: {}", x),
                    "float64".to_string(),
                ))
            } else {
                Ok(x.acosh())
            }
        },
    )));

    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arctanh",
        |x: &f32| {
            if *x <= -1.0 || *x >= 1.0 {
                Err(NumPyError::value_error(
                    format!("arctanh domain error: {}", x),
                    "float32".to_string(),
                ))
            } else {
                Ok(x.atanh())
            }
        },
    )));
    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "arctanh",
        |x: &f64| {
            if *x <= -1.0 || *x >= 1.0 {
                Err(NumPyError::value_error(
                    format!("arctanh domain error: {}", x),
                    "float64".to_string(),
                ))
            } else {
                Ok(x.atanh())
            }
        },
    )));

    // Exponential and logarithmic functions
    registry.register(Box::new(MathUnaryUfunc::new("exp", |x: &f32| x.exp())));
    registry.register(Box::new(MathUnaryUfunc::new("exp", |x: &f64| x.exp())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "exp",
        |x: &num_complex::Complex32| x.exp(),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "exp",
        |x: &num_complex::Complex64| x.exp(),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("exp2", |x: &f32| x.exp2())));
    registry.register(Box::new(MathUnaryUfunc::new("exp2", |x: &f64| x.exp2())));

    registry.register(Box::new(MathUnaryUfunc::new("expm1", |x: &f32| x.expm1())));
    registry.register(Box::new(MathUnaryUfunc::new("expm1", |x: &f64| x.expm1())));

    registry.register(Box::new(MathUnaryUfuncWithResult::new("log", |x: &f32| {
        if *x <= 0.0 {
            Err(NumPyError::value_error(
                format!("log domain error: {}", x),
                "float32".to_string(),
            ))
        } else {
            Ok(x.ln())
        }
    })));
    registry.register(Box::new(MathUnaryUfuncWithResult::new("log", |x: &f64| {
        if *x <= 0.0 {
            Err(NumPyError::value_error(
                format!("log domain error: {}", x),
                "float64".to_string(),
            ))
        } else {
            Ok(x.ln())
        }
    })));

    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "log2",
        |x: &f32| {
            if *x <= 0.0 {
                Err(NumPyError::value_error(
                    format!("log2 domain error: {}", x),
                    "float32".to_string(),
                ))
            } else {
                x.log2()
            }
        },
    )));
    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "log2",
        |x: &f64| {
            if *x <= 0.0 {
                Err(NumPyError::value_error(
                    format!("log2 domain error: {}", x),
                    "float64".to_string(),
                ))
            } else {
                x.log2()
            }
        },
    )));

    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "log10",
        |x: &f32| {
            if *x <= 0.0 {
                Err(NumPyError::value_error(
                    format!("log10 domain error: {}", x),
                    "float32".to_string(),
                ))
            } else {
                x.log10()
            }
        },
    )));
    registry.register(Box::new(MathUnaryUfuncWithResult::new(
        "log10",
        |x: &f64| {
            if *x <= 0.0 {
                Err(NumPyError::value_error(
                    format!("log10 domain error: {}", x),
                    "float64".to_string(),
                ))
            } else {
                x.log10()
            }
        },
    )));

    registry.register(Box::new(MathUnaryUfunc::new("log1p", |x: &f32| x.ln_1p())));
    registry.register(Box::new(MathUnaryUfunc::new("log1p", |x: &f64| x.ln_1p())));

    registry.register(Box::new(MathBinaryUfunc::new(
        "logaddexp",
        |x: &f32, y: &f32| {
            let max_val = if *x > *y { *x } else { *y };
            let min_val = if *x > *y { *y } else { *x };
            max_val + (1.0 + (-min_val + max_val).exp()).ln()
        },
    )));
    registry.register(Box::new(MathBinaryUfunc::new(
        "logaddexp",
        |x: &f64, y: &f64| {
            let max_val = if *x > *y { *x } else { *y };
            let min_val = if *x > *y { *y } else { *x };
            max_val + (1.0 + (-min_val + max_val).exp()).ln()
        },
    )));

    registry.register(Box::new(MathBinaryUfunc::new(
        "logaddexp2",
        |x: &f32, y: &f32| {
            let max_val = if *x > *y { *x } else { *y };
            let min_val = if *x > *y { *y } else { *x };
            max_val + (1.0 + (-min_val + max_val).exp2()).log2()
        },
    )));
    registry.register(Box::new(MathBinaryUfunc::new(
        "logaddexp2",
        |x: &f64, y: &f64| {
            let max_val = if *x > *y { *x } else { *y };
            let min_val = if *x > *y { *y } else { *x };
            max_val + (1.0 + (-min_val + max_val).exp2()).log2()
        },
    )));

    // Rounding functions
    registry.register(Box::new(MathUnaryUfunc::new("rint", |x: &f32| {
        (*x).round()
    })));
    registry.register(Box::new(MathUnaryUfunc::new("rint", |x: &f64| {
        (*x).round()
    })));

    registry.register(Box::new(MathUnaryUfunc::new("floor", |x: &f32| x.floor())));
    registry.register(Box::new(MathUnaryUfunc::new("floor", |x: &f64| x.floor())));

    registry.register(Box::new(MathUnaryUfunc::new("ceil", |x: &f32| x.ceil())));
    registry.register(Box::new(MathUnaryUfunc::new("ceil", |x: &f64| x.ceil())));

    registry.register(Box::new(MathUnaryUfunc::new("trunc", |x: &f32| x.trunc())));
    registry.register(Box::new(MathUnaryUfunc::new("trunc", |x: &f64| x.trunc())));

    registry.register(Box::new(MathUnaryUfunc::new("fix", |x: &f32| {
        if *x >= 0.0 {
            x.floor()
        } else {
            x.ceil()
        }
    })));
    registry.register(Box::new(MathUnaryUfunc::new("fix", |x: &f64| {
        if *x >= 0.0 {
            x.floor()
        } else {
            x.ceil()
        }
    })));

    // Floating-point checking functions
    registry.register(Box::new(FloatCheckUfunc::new("isnan", |x: &f32| {
        x.is_nan()
    })));
    registry.register(Box::new(FloatCheckUfunc::new("isnan", |x: &f64| {
        x.is_nan()
    })));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isnan",
        |x: &num_complex::Complex32| x.re.is_nan() || x.im.is_nan(),
    )));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isnan",
        |x: &num_complex::Complex64| x.re.is_nan() || x.im.is_nan(),
    )));

    registry.register(Box::new(FloatCheckUfunc::new("isinf", |x: &f32| {
        x.is_infinite()
    })));
    registry.register(Box::new(FloatCheckUfunc::new("isinf", |x: &f64| {
        x.is_infinite()
    })));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isinf",
        |x: &num_complex::Complex32| x.re.is_infinite() || x.im.is_infinite(),
    )));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isinf",
        |x: &num_complex::Complex64| x.re.is_infinite() || x.im.is_infinite(),
    )));

    registry.register(Box::new(FloatCheckUfunc::new("isfinite", |x: &f32| {
        x.is_finite()
    })));
    registry.register(Box::new(FloatCheckUfunc::new("isfinite", |x: &f64| {
        x.is_finite()
    })));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isfinite",
        |x: &num_complex::Complex32| x.re.is_finite() && x.im.is_finite(),
    )));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isfinite",
        |x: &num_complex::Complex64| x.re.is_finite() && x.im.is_finite(),
    )));

    registry.register(Box::new(FloatCheckUfunc::new("isneginf", |x: &f32| {
        *x == f32::NEG_INFINITY
    })));
    registry.register(Box::new(FloatCheckUfunc::new("isneginf", |x: &f64| {
        *x == f64::NEG_INFINITY
    })));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isneginf",
        |x: &num_complex::Complex32| x.re == f32::NEG_INFINITY,
    )));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isneginf",
        |x: &num_complex::Complex64| x.re == f64::NEG_INFINITY,
    )));

    registry.register(Box::new(FloatCheckUfunc::new("isposinf", |x: &f32| {
        *x == f32::INFINITY
    })));
    registry.register(Box::new(FloatCheckUfunc::new("isposinf", |x: &f64| {
        *x == f64::INFINITY
    })));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isposinf",
        |x: &num_complex::Complex32| x.re == f32::INFINITY,
    )));
    registry.register(Box::new(FloatCheckUfunc::new(
        "isposinf",
        |x: &num_complex::Complex64| x.re == f64::INFINITY,
    )));

    // Sign and absolute value functions
    registry.register(Box::new(MathUnaryUfunc::new("sign", |x: &f32| {
        if x.is_nan() {
            f32::NAN
        } else if *x == 0.0 {
            0.0
        } else if x.is_sign_negative() {
            -1.0
        } else {
            1.0
        }
    })));
    registry.register(Box::new(MathUnaryUfunc::new("sign", |x: &f64| {
        if x.is_nan() {
            f64::NAN
        } else if *x == 0.0 {
            0.0
        } else if x.is_sign_negative() {
            -1.0
        } else {
            1.0
        }
    })));

    registry.register(Box::new(MathUnaryUfunc::new("signbit", |x: &f32| {
        if x.is_nan() || x.is_sign_negative() {
            1.0
        } else {
            0.0
        }
    })));
    registry.register(Box::new(MathUnaryUfunc::new("signbit", |x: &f64| {
        if x.is_nan() || x.is_sign_negative() {
            1.0
        } else {
            0.0
        }
    })));

    registry.register(Box::new(MathBinaryUfunc::new(
        "copysign",
        |x: &f32, y: &f32| {
            if y.is_nan() {
                f32::NAN
            } else {
                x.abs() * y.signum()
            }
        },
    )));
    registry.register(Box::new(MathBinaryUfunc::new(
        "copysign",
        |x: &f64, y: &f64| {
            if y.is_nan() {
                f64::NAN
            } else {
                x.abs() * y.signum()
            }
        },
    )));

    registry.register(Box::new(MathUnaryUfunc::new("absolute", |x: &f32| x.abs())));
    registry.register(Box::new(MathUnaryUfunc::new("absolute", |x: &f64| x.abs())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "absolute",
        |x: &num_complex::Complex32| num_complex::Complex32::new(x.re.abs(), x.im.abs()),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "absolute",
        |x: &num_complex::Complex64| num_complex::Complex64::new(x.re.abs(), x.im.abs()),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("abs", |x: &f32| x.abs())));
    registry.register(Box::new(MathUnaryUfunc::new("abs", |x: &f64| x.abs())));
    registry.register(Box::new(MathUnaryUfunc::new(
        "abs",
        |x: &num_complex::Complex32| num_complex::Complex32::new(x.re.abs(), x.im.abs()),
    )));
    registry.register(Box::new(MathUnaryUfunc::new(
        "abs",
        |x: &num_complex::Complex64| num_complex::Complex64::new(x.re.abs(), x.im.abs()),
    )));

    registry.register(Box::new(MathUnaryUfunc::new("fabs", |x: &f32| x.abs())));
    registry.register(Box::new(MathUnaryUfunc::new("fabs", |x: &f64| x.abs())));
}

/// Compute sinc function: sin(pi*x) / (pi*x)
/// For x=0, sinc(0) = 1.0
pub fn sinc<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + num_traits::FloatConst + 'static + Send + Sync,
{
    let mut data = Vec::with_capacity(x.size());
    let pi = T::PI();
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            if val.is_zero() {
                data.push(T::one());
            } else {
                let pix = pi * *val;
                data.push(pix.sin() / pix);
            }
        }
    }
    Ok(Array::from_data(data, x.shape().to_vec()))
}

/// Modified Bessel function of the first kind, order 0.
/// This is a polynomial approximation for I0(x).
pub fn i0<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static + Send + Sync,
{
    let mut data = Vec::with_capacity(x.size());
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            let ax = val.abs();
            let y: T;
            if ax < T::from(3.75).unwrap() {
                let t = ax / T::from(3.75).unwrap();
                let t2 = t * t;
                y = T::one()
                    + t2 * (T::from(3.5156229).unwrap()
                        + t2 * (T::from(3.0899424).unwrap()
                            + t2 * (T::from(1.2067492).unwrap()
                                + t2 * (T::from(0.2659732).unwrap()
                                    + t2 * (T::from(0.0360768).unwrap()
                                        + t2 * T::from(0.0045813).unwrap())))));
            } else {
                let t = T::from(3.75).unwrap() / ax;
                y = ax.exp() / ax.sqrt()
                    * (T::from(0.39894228).unwrap()
                        + t * (T::from(0.01328592).unwrap()
                            + t * (T::from(0.00225319).unwrap()
                                + t * (T::from(-0.00157565).unwrap()
                                    + t * (T::from(0.00916281).unwrap()
                                        + t * (T::from(-0.02057706).unwrap()
                                            + t * (T::from(0.02635537).unwrap()
                                                + t * (T::from(-0.01647633).unwrap()
                                                    + t * T::from(0.00392377).unwrap()))))))));
            }
            data.push(y);
        }
    }
    Ok(Array::from_data(data, x.shape().to_vec()))
}

/// Compute the Heaviside step function.
/// heaviside(x, h0) = 0 if x < 0, h0 if x == 0, 1 if x > 0
pub fn heaviside<T>(x: &Array<T>, h0: T) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + num_traits::Zero + num_traits::One + 'static + Send + Sync,
{
    let mut data = Vec::with_capacity(x.size());
    for i in 0..x.size() {
        if let Some(val) = x.get(i) {
            if *val < T::zero() {
                data.push(T::zero());
            } else if *val > T::zero() {
                data.push(T::one());
            } else {
                data.push(h0.clone());
            }
        }
    }
    Ok(Array::from_data(data, x.shape().to_vec()))
}

/// Discrete, linear convolution of two one-dimensional sequences.
pub fn convolve<T>(a: &Array<T>, v: &Array<T>, mode: &str) -> Result<Array<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static
        + Send
        + Sync,
{
    let a_data = a.data();
    let v_data = v.data();
    let n = a_data.len();
    let m = v_data.len();

    if n == 0 || m == 0 {
        return Err(NumPyError::invalid_value(
            "convolve: inputs must be non-empty",
        ));
    }

    let full_len = n + m - 1;
    let mut full_result = vec![T::default(); full_len];

    for i in 0..n {
        for j in 0..m {
            let prod = a_data[i].clone() * v_data[j].clone();
            full_result[i + j] = full_result[i + j].clone() + prod;
        }
    }

    match mode {
        "full" => Ok(Array::from_vec(full_result)),
        "same" => {
            let start = (m - 1) / 2;
            let end = start + n;
            Ok(Array::from_vec(full_result[start..end].to_vec()))
        }
        "valid" => {
            if n >= m {
                let start = m - 1;
                let end = n;
                Ok(Array::from_vec(full_result[start..end].to_vec()))
            } else {
                let start = n - 1;
                let end = m;
                Ok(Array::from_vec(full_result[start..end].to_vec()))
            }
        }
        _ => Err(NumPyError::invalid_value(
            "mode must be 'full', 'same', or 'valid'",
        )),
    }
}

/// Cross-correlation of two 1-dimensional sequences.
pub fn correlate<T>(a: &Array<T>, v: &Array<T>, mode: &str) -> Result<Array<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static
        + Send
        + Sync,
{
    // correlate(a, v) = convolve(a, v[::-1])
    let v_data = v.data();
    let v_reversed: Vec<T> = v_data.iter().rev().cloned().collect();
    let v_rev = Array::from_vec(v_reversed);
    convolve(a, &v_rev, mode)
}

#[cfg(test)]

mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn test_trig_functions() {
        let x = Array::from_data(vec![0.0, consts::PI / 4.0, consts::PI / 2.0], vec![3]);

        let sin_result = sin(&x).unwrap();
        let cos_result = cos(&x).unwrap();
        let tan_result = tan(&x).unwrap();

        assert_eq!(sin_result.size(), 3);
        assert_eq!(cos_result.size(), 3);
        assert_eq!(tan_result.size(), 3);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let x = Array::from_data(vec![0.0, 1.0, 2.0], vec![3]);

        let sinh_result = sinh(&x).unwrap();
        let cosh_result = cosh(&x).unwrap();
        let tanh_result = tanh(&x).unwrap();

        assert_eq!(sinh_result.size(), 3);
        assert_eq!(cosh_result.size(), 3);
        assert_eq!(tanh_result.size(), 3);
    }

    #[test]
    fn test_exp_log_functions() {
        let x = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);

        let exp_result = exp(&x).unwrap();
        let log_result = log(&x).unwrap();

        assert_eq!(exp_result.size(), 3);
        assert_eq!(log_result.size(), 3);
    }

    #[test]
    fn test_rounding_functions() {
        let x = Array::from_data(vec![1.2, 2.7, -3.4], vec![3]);

        let floor_result = floor(&x).unwrap();
        let ceil_result = ceil(&x).unwrap();
        let round_result = round_(&x, 0).unwrap();

        assert_eq!(floor_result.size(), 3);
        assert_eq!(ceil_result.size(), 3);
        assert_eq!(round_result.size(), 3);
    }

    #[test]
    fn test_isnan() {
        let x = Array::from_data(vec![1.0, f64::NAN, 3.0], vec![3]);

        let result = isnan(&x).unwrap();

        assert_eq!(result.size(), 3);
        assert!(!result.get(0).unwrap()); // 1.0 is not NaN
        assert!(result.get(1).unwrap()); // NaN is NaN
        assert!(!result.get(2).unwrap()); // 3.0 is not NaN
    }

    #[test]
    fn test_isinf() {
        let x = Array::from_data(vec![1.0, f64::INFINITY, f64::NEG_INFINITY], vec![3]);

        let result = isinf(&x).unwrap();

        assert_eq!(result.size(), 3);
        assert!(!result.get(0).unwrap()); // 1.0 is not infinite
        assert!(result.get(1).unwrap()); // INFINITY is infinite
        assert!(result.get(2).unwrap()); // NEG_INFINITY is infinite
    }

    #[test]
    fn test_isfinite() {
        let x = Array::from_data(vec![1.0, f64::NAN, f64::INFINITY], vec![3]);

        let result = isfinite(&x).unwrap();

        assert_eq!(result.size(), 3);
        assert!(result.get(0).unwrap()); // 1.0 is finite
        assert!(!result.get(1).unwrap()); // NaN is not finite
        assert!(!result.get(2).unwrap()); // INFINITY is not finite
    }

    #[test]
    fn test_isneginf() {
        let x = Array::from_data(vec![1.0, f64::NEG_INFINITY, f64::INFINITY], vec![3]);

        let result = isneginf(&x).unwrap();

        assert_eq!(result.size(), 3);
        assert!(!result.get(0).unwrap()); // 1.0 is not negative infinity
        assert!(result.get(1).unwrap()); // NEG_INFINITY is negative infinity
        assert!(!result.get(2).unwrap()); // INFINITY is not negative infinity
    }

    #[test]
    fn test_isposinf() {
        let x = Array::from_data(vec![1.0, f64::INFINITY, f64::NEG_INFINITY], vec![3]);

        let result = isposinf(&x).unwrap();

        assert_eq!(result.size(), 3);
        assert!(!result.get(0).unwrap()); // 1.0 is not positive infinity
        assert!(result.get(1).unwrap()); // INFINITY is positive infinity
        assert!(!result.get(2).unwrap()); // NEG_INFINITY is not positive infinity
    }

    #[test]
    fn test_float_check_edge_cases() {
        // Test with complex numbers containing NaN/Inf
        let x = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(f64::NAN, 2.0),
                num_complex::Complex64::new(1.0, f64::INFINITY),
            ],
            vec![3],
        );

        let isnan_result = isnan(&x).unwrap();
        let isinf_result = isinf(&x).unwrap();
        let isfinite_result = isfinite(&x).unwrap();

        assert!(!isnan_result.get(0).unwrap()); // (1.0, 2.0) is not NaN
        assert!(isnan_result.get(1).unwrap()); // (NaN, 2.0) has NaN in real part
        assert!(!isnan_result.get(2).unwrap()); // (1.0, Inf) is not NaN

        assert!(!isinf_result.get(0).unwrap()); // (1.0, 2.0) is not infinite
        assert!(!isinf_result.get(1).unwrap()); // (NaN, 2.0) - NaN is not Inf
        assert!(isinf_result.get(2).unwrap()); // (1.0, Inf) has infinite imag part

        assert!(isfinite_result.get(0).unwrap()); // (1.0, 2.0) is finite
        assert!(!isfinite_result.get(1).unwrap()); // (NaN, 2.0) is not finite (NaN)
        assert!(!isfinite_result.get(2).unwrap()); // (1.0, Inf) is not finite (Inf)
    }

    #[test]
    fn test_angle() {
        // Test with complex numbers on the unit circle
        let z = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 0.0),  // angle = 0
                num_complex::Complex64::new(0.0, 1.0),  // angle = π/2
                num_complex::Complex64::new(-1.0, 0.0), // angle = π
                num_complex::Complex64::new(0.0, -1.0), // angle = -π/2
                num_complex::Complex64::new(1.0, 1.0),  // angle = π/4
            ],
            vec![5],
        );

        let result = angle(&z).unwrap();

        assert!((result.get(0).unwrap() - 0.0).abs() < 1e-10);
        assert!((result.get(1).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((result.get(2).unwrap() - std::f64::consts::PI).abs() < 1e-10);
        assert!((result.get(3).unwrap() - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-10);
        assert!((result.get(4).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }

    #[test]
    fn test_angle32() {
        // Test with Complex32 numbers
        let z = Array::from_data(
            vec![
                num_complex::Complex32::new(1.0, 0.0),
                num_complex::Complex32::new(0.0, 1.0),
                num_complex::Complex32::new(-1.0, 0.0),
            ],
            vec![3],
        );

        let result = angle32(&z).unwrap();

        assert!((result.get(0).unwrap() - 0.0).abs() < 1e-6);
        assert!((result.get(1).unwrap() - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((result.get(2).unwrap() - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_real_imag() {
        // Test real part extraction
        let z = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(-3.0, 4.5),
                num_complex::Complex64::new(0.0, -1.0),
            ],
            vec![3],
        );

        let real_result = real(&z).unwrap();
        let imag_result = imag(&z).unwrap();

        assert_eq!(real_result.get(0).unwrap(), &1.0);
        assert_eq!(real_result.get(1).unwrap(), &(-3.0));
        assert_eq!(real_result.get(2).unwrap(), &0.0);

        assert_eq!(imag_result.get(0).unwrap(), &2.0);
        assert_eq!(imag_result.get(1).unwrap(), &4.5);
        assert_eq!(imag_result.get(2).unwrap(), &(-1.0));
    }

    #[test]
    fn test_real32_imag32() {
        // Test Complex32 versions
        let z = Array::from_data(
            vec![
                num_complex::Complex32::new(1.5, 2.5),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
            vec![2],
        );

        let real_result = real32(&z).unwrap();
        let imag_result = imag32(&z).unwrap();

        assert_eq!(real_result.get(0).unwrap(), &1.5);
        assert_eq!(real_result.get(1).unwrap(), &(-3.0));

        assert_eq!(imag_result.get(0).unwrap(), &2.5);
        assert_eq!(imag_result.get(1).unwrap(), &4.0);
    }

    #[test]
    fn test_conj() {
        // Test complex conjugate
        let z = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(-3.0, 4.5),
                num_complex::Complex64::new(0.0, -1.0),
            ],
            vec![3],
        );

        let result = conj(&z).unwrap();

        assert_eq!(
            result.get(0).unwrap(),
            &num_complex::Complex64::new(1.0, -2.0)
        );
        assert_eq!(
            result.get(1).unwrap(),
            &num_complex::Complex64::new(-3.0, -4.5)
        );
        assert_eq!(
            result.get(2).unwrap(),
            &num_complex::Complex64::new(0.0, 1.0)
        );
    }

    #[test]
    fn test_conj32() {
        // Test complex conjugate for Complex32
        let z = Array::from_data(
            vec![
                num_complex::Complex32::new(1.0, 2.0),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
            vec![2],
        );

        let result = conj32(&z).unwrap();

        assert_eq!(
            result.get(0).unwrap(),
            &num_complex::Complex32::new(1.0, -2.0)
        );
        assert_eq!(
            result.get(1).unwrap(),
            &num_complex::Complex32::new(-3.0, -4.0)
        );
    }

    #[test]
    fn test_conjugate() {
        // Test that conjugate is an alias for conj
        let z = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(-3.0, 4.5),
            ],
            vec![2],
        );

        let conj_result = conj(&z).unwrap();
        let conjugate_result = conjugate(&z).unwrap();

        assert_eq!(
            conj_result.get(0).unwrap(),
            conjugate_result.get(0).unwrap()
        );
        assert_eq!(
            conj_result.get(1).unwrap(),
            conjugate_result.get(1).unwrap()
        );
    }

    #[test]
    fn test_real_if_close() {
        // Test real_if_close with negligible imaginary part
        let z = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 0.0),     // already real
                num_complex::Complex64::new(2.0, 1e-12),   // negligible imag
                num_complex::Complex64::new(3.0, 0.5),     // significant imag
                num_complex::Complex64::new(-1.0, -1e-11), // negligible imag
            ],
            vec![4],
        );

        let result = real_if_close(&z, Some(1e-10)).unwrap();

        // Should have zero imaginary part for first, second, and fourth elements
        assert_eq!(
            result.get(0).unwrap(),
            &num_complex::Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            result.get(1).unwrap(),
            &num_complex::Complex64::new(2.0, 0.0)
        );
        assert_eq!(
            result.get(2).unwrap(),
            &num_complex::Complex64::new(3.0, 0.5)
        ); // unchanged
        assert_eq!(
            result.get(3).unwrap(),
            &num_complex::Complex64::new(-1.0, 0.0)
        );
    }

    #[test]
    fn test_real_if_close32() {
        // Test real_if_close32 with Complex32
        let z = Array::from_data(
            vec![
                num_complex::Complex32::new(1.0, 0.0),
                num_complex::Complex32::new(2.0, 1e-7),
                num_complex::Complex32::new(3.0, 0.5),
            ],
            vec![3],
        );

        let result = real_if_close32(&z, Some(1e-6)).unwrap();

        assert_eq!(
            result.get(0).unwrap(),
            &num_complex::Complex32::new(1.0, 0.0)
        );
        assert_eq!(
            result.get(1).unwrap(),
            &num_complex::Complex32::new(2.0, 0.0)
        );
        assert_eq!(
            result.get(2).unwrap(),
            &num_complex::Complex32::new(3.0, 0.5)
        ); // unchanged
    }

    #[test]
    fn test_complex_utilities_2d_array() {
        // Test with 2D complex arrays
        let z = Array::from_data(
            vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(3.0, 4.0),
                num_complex::Complex64::new(5.0, 6.0),
                num_complex::Complex64::new(7.0, 8.0),
            ],
            vec![2, 2],
        );

        let real_result = real(&z).unwrap();
        let imag_result = imag(&z).unwrap();
        let conj_result = conj(&z).unwrap();

        assert_eq!(real_result.shape(), &vec![2, 2]);
        assert_eq!(imag_result.shape(), &vec![2, 2]);
        assert_eq!(conj_result.shape(), &vec![2, 2]);

        assert_eq!(real_result.get(0).unwrap(), &1.0);
        assert_eq!(imag_result.get(0).unwrap(), &2.0);
        assert_eq!(
            conj_result.get(0).unwrap(),
            &num_complex::Complex64::new(1.0, -2.0)
        );

        assert_eq!(real_result.get(3).unwrap(), &7.0);
        assert_eq!(imag_result.get(3).unwrap(), &8.0);
        assert_eq!(
            conj_result.get(3).unwrap(),
            &num_complex::Complex64::new(7.0, -8.0)
        );
    }
}

// Sign and absolute value functions

/// Compute sign of each element (-1, 0, or 1)
pub fn sign<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("sign") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "sign",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Test if sign bit is set in each element (true for negative numbers, -0.0, and NaN)
pub fn signbit<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("signbit") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "signbit",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Copy sign of x2 to x1
pub fn copysign<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("copysign") {
        let mut output = Array::from_data(vec![T::default(); x1.size()], x1.shape().to_vec());
        ufunc.execute(&[x1, x2], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "copysign",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute absolute value element-wise
pub fn absolute<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("absolute") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "absolute",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Compute absolute value element-wise (alias for absolute)
pub fn abs<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    absolute(x)
}

/// Compute absolute value for float types
pub fn fabs<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if let Some(ufunc) = get_ufunc("fabs") {
        let mut output = Array::from_data(vec![T::default(); x.size()], x.shape().to_vec());
        ufunc.execute(&[x], &mut [&mut output], None)?;
        Ok(output)
    } else {
        Err(NumPyError::ufunc_error(
            "fabs",
            "Ufunc not found".to_string(),
        ))
    }
}

/// Unwrap phase angles by changing absolute jumps greater than discont to their 2π complement
///
/// This transforms phase angles to eliminate large jumps, which is critical for
/// signal processing applications, particularly when working with phase data from FFTs.
///
/// # Arguments
/// * `p` - Input array of phase angles
/// * `discont` - Maximum discontinuity between values (default: π)
/// * `axis` - Axis along which to unwrap (default: -1, last axis)
/// * `period` - Period of the input (default: 2π)
///
/// # Returns
/// Unwrapped phase angles
///
/// # Example
/// ```ignore
/// use numpy::{array, unwrap};
/// let phases = array![0.0, 6.0, 0.0, 6.0];
/// let unwrapped = unwrap(&phases.view(), None, None, -1).unwrap();
/// // Results in: [0.0, 6.0, 6.283..., 12.283...]
/// ```
///
/// # Notes
/// This is a non-generic implementation for f64 to avoid circular dependencies
/// with Float and FromPrimitive traits. For other types, use the generic version
/// when the trait dependency issue is resolved.
pub fn unwrap(
    p: &ArrayView<f64>,
    discont: Option<f64>,
    axis: Option<isize>,
    period: Option<f64>,
) -> Result<Array<f64>> {
    use std::f64::consts::PI;

    let discont_val = discont.unwrap_or(PI);
    let period_val = period.unwrap_or(2.0 * PI);
    let axis_val = axis.unwrap_or(-1);

    // Handle axis parameter (normalize negative indices)
    let ndim = p.ndim();
    let normalized_axis = if axis_val < 0 {
        ndim as isize + axis_val
    } else {
        axis_val
    };

    if ndim == 0 || normalized_axis < 0 || normalized_axis >= ndim as isize {
        return Err(NumPyError::value_error(
            &format!("axis {} is out of bounds for array of dimension {}", axis_val, ndim),
            "unwrap",
        ));
    }

    let axis_usize = normalized_axis as usize;

    // Get the data and shape
    let input_data = p.data();
    let input_shape = p.shape();

    // For 1D arrays, simple case
    if ndim == 1 {
        let mut result = Vec::with_capacity(input_data.len());
        if input_data.is_empty() {
            return Ok(Array::from_vec(result));
        }

        result.push(input_data[0]);

        for i in 1..input_data.len() {
            let prev = result[i - 1];
            let curr = input_data[i];
            let diff = curr - prev;

            // Check for discontinuity and correct
            let diff_corrected = (diff + period_val / 2.0).rem_euclid(period_val) - period_val / 2.0;
            
            // Determine how many periods to correct
            let mut cum_correction = 0.0;
            if diff > discont_val {
                cum_correction = -period_val * ((diff - discont_val) / period_val).ceil();
            } else if diff < -discont_val {
                cum_correction = period_val * ((-diff - discont_val) / period_val).ceil();
            }

            result.push(curr + cum_correction);
        }

        return Ok(Array::from_vec(result));
    }

    // For multi-dimensional arrays, process along the specified axis
    let mut result = input_data.to_vec();

    // Calculate stride and dimensions
    let axis_length = input_shape[axis_usize];
    
    // Calculate total elements and number of slices along the axis
    let total_elements: usize = input_shape.iter().product();
    let num_slices = total_elements / axis_length;

    // Calculate stride in the input data for moving along axis
    let mut stride = 1usize;
    for i in (axis_usize + 1)..ndim {
        stride *= input_shape[i];
    }

    // Process each 1D slice along the axis
    for slice_idx in 0..num_slices {
        let base_offset = (slice_idx / stride) * stride * axis_length + (slice_idx % stride);

        for i in 1..axis_length {
            let curr_idx = base_offset + i * stride;
            let prev_idx = base_offset + (i - 1) * stride;

            let prev = result[prev_idx];
            let curr = input_data[curr_idx];
            let diff = curr - prev;

            // Check for discontinuity
            if diff > discont_val {
                let correction = period_val * ((diff - discont_val) / period_val).ceil();
                result[curr_idx] = curr - correction;
            } else if diff < -discont_val {
                let correction = period_val * ((-diff - discont_val) / period_val).ceil();
                result[curr_idx] = curr + correction;
            } else {
                result[curr_idx] = curr;
            }
        }
    }

    Ok(Array::from_data(result, input_shape.to_vec()))
}

/// Unwrap phase angles (f32 version)
///
/// See the f64 version documentation for details. This is provided for
/// consistency with NumPy's API which supports multiple float types.
pub fn unwrap_f32(
    p: &ArrayView<f32>,
    discont: Option<f32>,
    axis: Option<isize>,
    period: Option<f32>,
) -> Result<Array<f32>> {
    use std::f32::consts::PI;

    let discont_val = discont.unwrap_or(PI);
    let period_val = period.unwrap_or(2.0 * PI);
    
    // Convert to f64, process, then convert back
    let p_f64: Vec<f64> = p.data().iter().map(|&x| x as f64).collect();
    let shape_f64 = p.shape().to_vec();
    let strides = p.strides().to_vec();
    
    // Create ArrayView for f64
    let array_view_f64 = ArrayView::from_data(&p_f64, &shape_f64, &strides);
    
    let result_f64 = unwrap(
        &array_view_f64, 
        Some(discont_val as f64), 
        axis.map(|a| a as isize), 
        Some(period_val as f64)
    )?;

    // Convert back to f32
    let result_f32: Vec<f32> = result_f64.data().iter().map(|&x| x as f32).collect();
    Ok(Array::from_vec(result_f32))
}
