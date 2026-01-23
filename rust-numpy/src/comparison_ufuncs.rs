use crate::array::Array;
use crate::broadcasting::broadcast_arrays;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use crate::ufunc::Ufunc;
use std::marker::PhantomData;

pub struct ComparisonUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T, &T) -> bool + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> ComparisonUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T, &T) -> bool + Send + Sync,
{
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for ComparisonUfunc<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(&T, &T) -> bool + Send + Sync,
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
        &[
            DtypeKind::Integer,
            DtypeKind::Unsigned,
            DtypeKind::Float,
            DtypeKind::Complex,
        ]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 2 && input_types.iter().all(|&t| t == std::any::type_name::<T>())
    }

    fn input_dtypes(&self) -> Vec<Dtype> {
        vec![Dtype::from_type::<T>(), Dtype::from_type::<T>()]
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
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<bool>) };

        let broadcast_shape = output.shape();

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, broadcast_shape)?)
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

pub struct LogicalUnaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> bool + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> LogicalUnaryUfunc<T, F>
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

impl<T, F> Ufunc for LogicalUnaryUfunc<T, F>
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
        &[
            DtypeKind::Integer,
            DtypeKind::Unsigned,
            DtypeKind::Float,
            DtypeKind::Complex,
        ]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 1 && input_types[0] == std::any::type_name::<T>()
    }

    fn input_dtypes(&self) -> Vec<Dtype> {
        vec![Dtype::from_type::<T>()]
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

        let input = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<bool>) };

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

pub struct ExtremaUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T, &T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> ExtremaUfunc<T, F>
where
    T: Clone + 'static + Send + Sync,
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

impl<T, F> Ufunc for ExtremaUfunc<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
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
        &[
            DtypeKind::Integer,
            DtypeKind::Unsigned,
            DtypeKind::Float,
            DtypeKind::Complex,
        ]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 2 && input_types.iter().all(|&t| t == std::any::type_name::<T>())
    }

    fn input_dtypes(&self) -> Vec<Dtype> {
        vec![Dtype::from_type::<T>(), Dtype::from_type::<T>()]
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

        let broadcast_shape = output.shape();

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, broadcast_shape)?)
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

pub trait ComparisonOps<T>: Send + Sync {
    fn greater(&self, other: &T) -> bool;
    fn less(&self, other: &T) -> bool;
    fn greater_equal(&self, other: &T) -> bool;
    fn less_equal(&self, other: &T) -> bool;
    fn equal(&self, other: &T) -> bool;
    fn not_equal(&self, other: &T) -> bool;
}

pub trait LogicalOps<T>: Send + Sync {
    fn logical_and(&self, other: &T) -> bool;
    fn logical_or(&self, other: &T) -> bool;
    fn logical_xor(&self, other: &T) -> bool;
}

pub trait LogicalUnaryOps<T>: Send + Sync {
    fn logical_not(&self) -> bool;
}

macro_rules! impl_comparison_ops_signed {
    ($($t:ty),*) => {
        $(
            impl ComparisonOps<$t> for $t {
                fn greater(self: &$t, other: &$t) -> bool { *self > *other }
                fn less(self: &$t, other: &$t) -> bool { *self < *other }
                fn greater_equal(self: &$t, other: &$t) -> bool { *self >= *other }
                fn less_equal(self: &$t, other: &$t) -> bool { *self <= *other }
                fn equal(self: &$t, other: &$t) -> bool { *self == *other }
                fn not_equal(self: &$t, other: &$t) -> bool { *self != *other }
            }

            impl LogicalOps<$t> for $t {
                fn logical_and(self: &$t, other: &$t) -> bool { *self != 0 && *other != 0 }
                fn logical_or(self: &$t, other: &$t) -> bool { *self != 0 || *other != 0 }
                fn logical_xor(self: &$t, other: &$t) -> bool { (*self != 0) != (*other != 0) }
            }

            impl LogicalUnaryOps<$t> for $t {
                fn logical_not(self: &$t) -> bool { *self == 0 }
            }
        )*
    }
}

macro_rules! impl_comparison_ops_unsigned {
    ($($t:ty),*) => {
        $(
            impl ComparisonOps<$t> for $t {
                fn greater(self: &$t, other: &$t) -> bool { *self > *other }
                fn less(self: &$t, other: &$t) -> bool { *self < *other }
                fn greater_equal(self: &$t, other: &$t) -> bool { *self >= *other }
                fn less_equal(self: &$t, other: &$t) -> bool { *self <= *other }
                fn equal(self: &$t, other: &$t) -> bool { *self == *other }
                fn not_equal(self: &$t, other: &$t) -> bool { *self != *other }
            }

            impl LogicalOps<$t> for $t {
                fn logical_and(self: &$t, other: &$t) -> bool { *self != 0 && *other != 0 }
                fn logical_or(self: &$t, other: &$t) -> bool { *self != 0 || *other != 0 }
                fn logical_xor(self: &$t, other: &$t) -> bool { (*self != 0) != (*other != 0) }
            }

            impl LogicalUnaryOps<$t> for $t {
                fn logical_not(self: &$t) -> bool { *self == 0 }
            }
        )*
    }
}

macro_rules! impl_comparison_ops_float {
    ($($t:ty),*) => {
        $(
            impl ComparisonOps<$t> for $t {
                fn greater(self: &$t, other: &$t) -> bool {
                    if self.is_nan() || other.is_nan() { return false; }
                    *self > *other
                }
                fn less(self: &$t, other: &$t) -> bool {
                    if self.is_nan() || other.is_nan() { return false; }
                    *self < *other
                }
                fn greater_equal(self: &$t, other: &$t) -> bool {
                    if self.is_nan() || other.is_nan() { return false; }
                    *self >= *other
                }
                fn less_equal(self: &$t, other: &$t) -> bool {
                    if self.is_nan() || other.is_nan() { return false; }
                    *self <= *other
                }
                fn equal(self: &$t, other: &$t) -> bool {
                    if self.is_nan() && other.is_nan() { return false; }
                    if self.is_nan() || other.is_nan() { return false; }
                    *self == *other
                }
                fn not_equal(self: &$t, other: &$t) -> bool {
                    if self.is_nan() && other.is_nan() { return true; }
                    if self.is_nan() || other.is_nan() { return true; }
                    *self != *other
                }
            }

            impl LogicalOps<$t> for $t {
                fn logical_and(self: &$t, other: &$t) -> bool { *self != 0.0 && *other != 0.0 }
                fn logical_or(self: &$t, other: &$t) -> bool { *self != 0.0 || *other != 0.0 }
                fn logical_xor(self: &$t, other: &$t) -> bool { (*self != 0.0) != (*other != 0.0) }
            }

            impl LogicalUnaryOps<$t> for $t {
                fn logical_not(self: &$t) -> bool { *self == 0.0 }
            }
        )*
    }
}

impl_comparison_ops_signed!(i64, i32, i16, i8);
impl_comparison_ops_unsigned!(u64, u32, u16, u8);
impl_comparison_ops_float!(f64, f32);

impl<T> ComparisonOps<num_complex::Complex<T>> for num_complex::Complex<T>
where
    T: PartialOrd + num_traits::Float + std::marker::Send + std::marker::Sync,
{
    fn greater(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        self.norm() > other.norm()
    }
    fn less(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        self.norm() < other.norm()
    }
    fn greater_equal(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        self.norm() >= other.norm()
    }
    fn less_equal(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        self.norm() <= other.norm()
    }
    fn equal(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        self.re == other.re && self.im == other.im
    }
    fn not_equal(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        self.re != other.re || self.im != other.im
    }
}

impl<T> LogicalOps<num_complex::Complex<T>> for num_complex::Complex<T>
where
    T: num_traits::Float + std::marker::Send + std::marker::Sync,
{
    fn logical_and(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        (self.norm() != T::zero()) && (other.norm() != T::zero())
    }
    fn logical_or(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        (self.norm() != T::zero()) || (other.norm() != T::zero())
    }
    fn logical_xor(self: &num_complex::Complex<T>, other: &num_complex::Complex<T>) -> bool {
        (self.norm() != T::zero()) != (other.norm() != T::zero())
    }
}

impl<T> LogicalUnaryOps<num_complex::Complex<T>> for num_complex::Complex<T>
where
    T: num_traits::Float + std::marker::Send + std::marker::Sync,
{
    fn logical_not(self: &num_complex::Complex<T>) -> bool {
        self.norm() == T::zero()
    }
}

// ==================== TOLERANCE-BASED COMPARISON FUNCTIONS ====================

/// Check if two arrays are element-wise equal within tolerance
///
/// Returns a boolean array where each element is True if the corresponding
/// elements of the input arrays are close within the specified tolerance.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance (default: 1e-05)
/// * `atol` - Absolute tolerance (default: 1e-08)
/// * `equal_nan` - If True, consider NaN values as equal (default: false)
///
/// # Formula
/// `|a - b| <= atol + rtol * |b|`
///
/// # Returns
/// Boolean array indicating which elements are close
pub fn isclose<T>(
    a: &Array<T>,
    b: &Array<T>,
    rtol: Option<f64>,
    atol: Option<f64>,
    equal_nan: Option<bool>,
) -> Result<Array<bool>>
where
    T: Clone + Default + Into<f64> + 'static + Send + Sync,
{
    let rtol = rtol.unwrap_or(1e-05);
    let atol = atol.unwrap_or(1e-08);
    let equal_nan = equal_nan.unwrap_or(false);

    // Broadcast arrays if needed
    let broadcasted = broadcast_arrays(&[a, b])?;
    let a_broadcast = &broadcasted[0];
    let b_broadcast = &broadcasted[1];

    let size = a_broadcast.size();
    let mut result = vec![false; size];

    for i in 0..size {
        let a_val = a_broadcast.get_linear(i).unwrap();
        let b_val = b_broadcast.get_linear(i).unwrap();

        let a_f64: f64 = a_val.clone().into();
        let b_f64: f64 = b_val.clone().into();

        // Handle NaN comparisons
        if a_f64.is_nan() || b_f64.is_nan() {
            result[i] = equal_nan && a_f64.is_nan() && b_f64.is_nan();
            continue;
        }

        // Handle infinity comparisons
        if a_f64.is_infinite() || b_f64.is_infinite() {
            result[i] = a_f64 == b_f64;
            continue;
        }

        // Standard tolerance comparison
        let diff = (a_f64 - b_f64).abs();
        let tolerance = atol + rtol * b_f64.abs();
        result[i] = diff <= tolerance;
    }

    Ok(Array::from_vec(result)
        .reshape(a_broadcast.shape())
        .unwrap())
}

/// Check if two arrays are element-wise equal within tolerance
///
/// Returns True if all elements of the two arrays are equal within
/// the specified tolerance.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance (default: 1e-05)
/// * `atol` - Absolute tolerance (default: 1e-08)
/// * `equal_nan` - If True, consider NaN values as equal (default: false)
///
/// # Returns
/// True if all elements are close, False otherwise
pub fn allclose<T>(
    a: &Array<T>,
    b: &Array<T>,
    rtol: Option<f64>,
    atol: Option<f64>,
    equal_nan: Option<bool>,
) -> Result<bool>
where
    T: Clone + Default + Into<f64> + 'static + Send + Sync,
{
    let close_array = isclose(a, b, rtol, atol, equal_nan)?;

    // Check if all elements are true
    for i in 0..close_array.size() {
        if let Some(&val) = close_array.get_linear(i) {
            if !val {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Check if two arrays are exactly equal
///
/// Returns True if arrays have the same shape and all elements are equal.
/// This is a strict comparison that does not use tolerance.
///
/// # Arguments
/// * `a1` - First array
/// * `a2` - Second array
///
/// # Returns
/// True if arrays are equal, False otherwise
pub fn array_equal<T>(a1: &Array<T>, a2: &Array<T>) -> bool
where
    T: Clone + PartialEq + 'static,
{
    // Check shape equality
    if a1.shape() != a2.shape() {
        return false;
    }

    // Check element equality
    for i in 0..a1.size() {
        let val1 = a1.get_linear(i);
        let val2 = a2.get_linear(i);

        match (val1, val2) {
            (Some(v1), Some(v2)) => {
                if v1 != v2 {
                    return false;
                }
            }
            _ => return false,
        }
    }

    true
}

/// Check if two arrays are element-wise equivalent (broadcasting allowed)
///
/// Returns True if arrays are element-wise equal after broadcasting.
/// Unlike array_equal, this function allows broadcasting.
///
/// # Arguments
/// * `a1` - First array
/// * `a2` - Second array
///
/// # Returns
/// True if arrays are equivalent, False otherwise
pub fn array_equiv<T>(a1: &Array<T>, a2: &Array<T>) -> bool
where
    T: Clone + PartialEq + Default + 'static,
{
    // Try to broadcast arrays
    let broadcasted = match broadcast_arrays(&[a1, a2]) {
        Ok(arrs) => arrs,
        Err(_) => return false,
    };

    let a1_broadcast = &broadcasted[0];
    let a2_broadcast = &broadcasted[1];

    // Check element equality
    for i in 0..a1_broadcast.size() {
        let val1 = a1_broadcast.get_linear(i);
        let val2 = a2_broadcast.get_linear(i);

        match (val1, val2) {
            (Some(v1), Some(v2)) => {
                if v1 != v2 {
                    return false;
                }
            }
            _ => return false,
        }
    }

    true
}

// ==================== PUBLIC EXPORTS ====================

/// Re-export comparison functions for public use
pub mod exports {
    pub use super::{allclose, array_equal, array_equiv, isclose};
}
