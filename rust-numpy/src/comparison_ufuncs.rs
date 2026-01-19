use crate::array::Array;
use crate::broadcasting::{broadcast_arrays, compute_broadcast_shape};

use crate::dtype::DtypeKind;
use crate::error::{NumPyError, Result};
// use crate::error::{NumPyError, Result}; // Removed duplicate
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

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
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

        let shape0 = input0.shape();
        let shape1 = input1.shape();
        let broadcast_shape = compute_broadcast_shape(shape0, shape1);

        let broadcasted = broadcast_arrays(&[input0, input1])?;

        let arr0 = &broadcasted[0];
        let arr1 = &broadcasted[1];

        for i in 0..broadcast_shape.iter().product::<usize>() {
            if let (Some(a), Some(b)) = (arr0.get(i), arr1.get(i)) {
                let result = (self.operation)(a, b);
                output.set(i, result)?;
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

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
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

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
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

        let shape0 = input0.shape();
        let shape1 = input1.shape();
        let broadcast_shape = compute_broadcast_shape(shape0, shape1);

        let broadcasted = broadcast_arrays(&[input0, input1])?;

        let arr0 = &broadcasted[0];
        let arr1 = &broadcasted[1];

        for i in 0..broadcast_shape.iter().product::<usize>() {
            if let (Some(a), Some(b)) = (arr0.get(i), arr1.get(i)) {
                let result = (self.operation)(a, b);
                output.set(i, result)?;
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
