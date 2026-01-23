//! Bitwise operations module for Rust NumPy
//!
//! Provides complete bitwise and logical operations with full NumPy compatibility.
//! Supports all integer dtypes with proper broadcasting, type safety, and performance optimizations.

use crate::array::Array;
use crate::broadcasting::broadcast_arrays;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use crate::ufunc::{Ufunc, UfuncRegistry};
use crate::UfuncEngine;
use std::marker::PhantomData;

/// Bitwise operations trait for integer types
pub trait BitwiseOps: Send + Sync + Sized {
    /// Bitwise AND operation
    fn bitwise_and(&self, other: &Self) -> Self;
    /// Bitwise OR operation
    fn bitwise_or(&self, other: &Self) -> Self;
    /// Bitwise XOR operation
    fn bitwise_xor(&self, other: &Self) -> Self;
    /// Bitwise NOT operation (one's complement)
    fn bitwise_not(&self) -> Self;
    /// Left shift operation with bounds checking
    fn left_shift(&self, shift: u32) -> Result<Self>;
    /// Right shift operation (arithmetic for signed, logical for unsigned)
    fn right_shift(&self, shift: u32) -> Result<Self>;
}

/// Macro to implement BitwiseOps for signed integer types
macro_rules! impl_bitwise_ops_signed {
    ($($t:ty),*) => {
        $(
            impl BitwiseOps for $t {
                fn bitwise_and(&self, other: &Self) -> Self {
                    self & other
                }

                fn bitwise_or(&self, other: &Self) -> Self {
                    self | other
                }

                fn bitwise_xor(&self, other: &Self) -> Self {
                    self ^ other
                }

                fn bitwise_not(&self) -> Self {
                    !self
                }

                fn left_shift(&self, shift: u32) -> Result<Self> {
                    let bit_width = std::mem::size_of::<$t>() as u32 * 8;
                    if shift >= bit_width {
                        return Err(NumPyError::invalid_value(
                            format!("Shift amount {} must be less than {}", shift, bit_width)
                        ));
                    }
                    Ok(self.wrapping_shl(shift))
                }

                fn right_shift(&self, shift: u32) -> Result<Self> {
                    let bit_width = std::mem::size_of::<$t>() as u32 * 8;
                    if shift >= bit_width {
                        return Err(NumPyError::invalid_value(
                            format!("Shift amount {} must be less than {}", shift, bit_width)
                        ));
                    }
                    Ok(self.wrapping_shr(shift))
                }
            }
        )*
    }
}

/// Macro to implement BitwiseOps for unsigned integer types
macro_rules! impl_bitwise_ops_unsigned {
    ($($t:ty),*) => {
        $(
            impl BitwiseOps for $t {
                fn bitwise_and(&self, other: &Self) -> Self {
                    self & other
                }

                fn bitwise_or(&self, other: &Self) -> Self {
                    self | other
                }

                fn bitwise_xor(&self, other: &Self) -> Self {
                    self ^ other
                }

                fn bitwise_not(&self) -> Self {
                    !self
                }

                fn left_shift(&self, shift: u32) -> Result<Self> {
                    let bit_width = std::mem::size_of::<$t>() as u32 * 8;
                    if shift >= bit_width {
                        return Err(NumPyError::invalid_value(
                            format!("Shift amount {} must be less than {}", shift, bit_width)
                        ));
                    }
                    Ok(self.wrapping_shl(shift))
                }

                fn right_shift(&self, shift: u32) -> Result<Self> {
                    let bit_width = std::mem::size_of::<$t>() as u32 * 8;
                    if shift >= bit_width {
                        return Err(NumPyError::invalid_value(
                            format!("Shift amount {} must be less than {}", shift, bit_width)
                        ));
                    }
                    Ok(self.wrapping_shr(shift))
                }
            }
        )*
    }
}

// Implement BitwiseOps for all integer types
impl_bitwise_ops_signed!(i8, i16, i32, i64);
impl_bitwise_ops_unsigned!(u8, u16, u32, u64);

/// Binary bitwise ufunc for operations like AND, OR, XOR
pub struct BitwiseBinaryUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + 'static,
    F: Fn(&T, &T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> BitwiseBinaryUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + 'static,
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

impl<T, F> Ufunc for BitwiseBinaryUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + Send + Sync + 'static,
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
        &[DtypeKind::Integer, DtypeKind::Unsigned]
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

/// Unary bitwise ufunc for operations like NOT
pub struct BitwiseUnaryUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + 'static,
    F: Fn(&T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> BitwiseUnaryUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + 'static,
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

impl<T, F> Ufunc for BitwiseUnaryUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + Send + Sync + 'static,
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
        &[DtypeKind::Integer, DtypeKind::Unsigned]
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
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        if !self.supported_dtypes().contains(&input.dtype().kind()) {
            return Err(NumPyError::dtype_error(format!(
                "Bitwise operations only support integer types, got {:?}",
                input.dtype().kind()
            )));
        }

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

/// Shift operation ufunc with validation
pub struct BitwiseShiftUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + 'static,
    F: Fn(&T, u32) -> Result<T> + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> BitwiseShiftUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + 'static,
    F: Fn(&T, u32) -> Result<T> + Send + Sync,
{
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }

    fn convert_shift_to_u32(&self, shift_val: &T) -> Result<u32> {
        // Handle different integer types for shift conversion
        // For signed types: ensure non-negative, then convert
        // For unsigned types: direct conversion
        match std::any::type_name::<T>() {
            "i8" => {
                let val = unsafe { std::mem::transmute::<&T, &i8>(shift_val) };
                if *val < 0 {
                    return Err(NumPyError::value_error(
                        "Shift amount must be non-negative".to_string(),
                        "int".to_string(),
                    ));
                }
                Ok(*val as u32)
            }
            "i16" => {
                let val = unsafe { std::mem::transmute::<&T, &i16>(shift_val) };
                if *val < 0 {
                    return Err(NumPyError::value_error(
                        "Shift amount must be non-negative".to_string(),
                        "int".to_string(),
                    ));
                }
                Ok(*val as u32)
            }
            "i32" => {
                let val = unsafe { std::mem::transmute::<&T, &i32>(shift_val) };
                if *val < 0 {
                    return Err(NumPyError::value_error(
                        "Shift amount must be non-negative".to_string(),
                        "int".to_string(),
                    ));
                }
                Ok(*val as u32)
            }
            "i64" => {
                let val = unsafe { std::mem::transmute::<&T, &i64>(shift_val) };
                if *val < 0 {
                    return Err(NumPyError::value_error(
                        "Shift amount must be non-negative".to_string(),
                        "int".to_string(),
                    ));
                }
                Ok(*val as u32)
            }
            "u8" => {
                let val = unsafe { std::mem::transmute::<&T, &u8>(shift_val) };
                Ok(*val as u32)
            }
            "u16" => {
                let val = unsafe { std::mem::transmute::<&T, &u16>(shift_val) };
                Ok(*val as u32)
            }
            "u32" => {
                let val = unsafe { std::mem::transmute::<&T, &u32>(shift_val) };
                Ok(*val)
            }
            "u64" => {
                let val = unsafe { std::mem::transmute::<&T, &u64>(shift_val) };
                if *val > u32::MAX as u64 {
                    return Err(NumPyError::invalid_value(format!(
                        "Shift amount {} exceeds u32::MAX",
                        val
                    )));
                }
                Ok(*val as u32)
            }
            _ => Err(NumPyError::dtype_error(format!(
                "Unsupported type for shift operation: {}",
                std::any::type_name::<T>()
            ))),
        }
    }
}

impl<T, F> Ufunc for BitwiseShiftUfunc<T, F>
where
    T: Clone + Default + BitwiseOps + Send + Sync + 'static,
    F: Fn(&T, u32) -> Result<T> + Send + Sync,
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
        &[DtypeKind::Integer, DtypeKind::Unsigned]
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
                if let (Some(a), Some(shift_val)) = (arr0.get(i), arr1.get(i)) {
                    // Convert shift amount to u32, handling different integer types
                    let shift_u32 = self.convert_shift_to_u32(shift_val)?;

                    let result = (self.operation)(a, shift_u32)?;
                    output.set(i, result)?;
                }
            }
        }

        Ok(())
    }
}

// Helper enum for type-based conversions
#[allow(dead_code)]
enum ShiftValue<T> {
    Signed(T),
    Unsigned(T),
}

/// Enhanced logical operations returning boolean arrays
pub struct EnhancedLogicalUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(&T, &T) -> bool + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> EnhancedLogicalUfunc<T, F>
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

impl<T, F> Ufunc for EnhancedLogicalUfunc<T, F>
where
    T: Clone + Default + Send + Sync + 'static,
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
            DtypeKind::Bool,
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

// Public API functions

/// Element-wise bitwise AND operation
///
/// Computes the bitwise AND of two arrays element-wise.
/// Only supports integer dtypes.
///
/// # Arguments
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
/// Result containing array with bitwise AND results
///
/// # Example
/// ```rust
/// use numpy::{array, bitwise_and};
/// let a = array![5, 3, 7];  // 101, 011, 111
/// let b = array![2, 6, 1];  // 010, 110, 001
/// let result = bitwise_and(&a, &b).unwrap();
/// // result == [0, 2, 1]    // 000, 010, 001
/// ```
pub fn bitwise_and<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_binary("bitwise_and", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Element-wise bitwise OR operation
///
/// Computes the bitwise OR of two arrays element-wise.
/// Only supports integer dtypes.
pub fn bitwise_or<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_binary("bitwise_or", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Element-wise bitwise XOR operation
///
/// Computes the bitwise XOR of two arrays element-wise.
/// Only supports integer dtypes.
pub fn bitwise_xor<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_binary("bitwise_xor", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Element-wise bitwise NOT operation
///
/// Computes the bitwise NOT (one's complement) of the input array element-wise.
/// Only supports integer dtypes.
pub fn bitwise_not<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_unary("bitwise_not", x, None, crate::dtype::Casting::Safe)
}

/// Alias for bitwise_not for NumPy compatibility
pub fn invert<T>(x: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    bitwise_not(x)
}

/// Element-wise left shift operation
///
/// Shifts the bits of the first input array to the left by the number of bits
/// specified in the second input array. Only supports integer dtypes.
///
/// Shift amounts must be non-negative and less than the bit width of the type.
pub fn left_shift<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_binary("left_shift", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Element-wise right shift operation
///
/// Shifts the bits of the first input array to the right by the number of bits
/// specified in the second input array. Only supports integer dtypes.
///
/// For signed integers, performs arithmetic right shift (preserves sign bit).
/// For unsigned integers, performs logical right shift.
///
/// Shift amounts must be non-negative and less than the bit width of the type.
pub fn right_shift<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_binary("right_shift", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Enhanced element-wise logical AND operation
///
/// Computes the logical AND of two arrays element-wise, treating non-zero values as True.
/// Returns a boolean array.
pub fn logical_and<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialEq + Default + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_comparison("logical_and", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Enhanced element-wise logical OR operation
///
/// Computes the logical OR of two arrays element-wise, treating non-zero values as True.
/// Returns a boolean array.
pub fn logical_or<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialEq + Default + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_comparison("logical_or", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Enhanced element-wise logical XOR operation
///
/// Computes the logical XOR of two arrays element-wise, treating non-zero values as True.
/// Returns a boolean array.
pub fn logical_xor<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialEq + Default + 'static,
{
    let engine = UfuncEngine::new();
    engine.execute_comparison("logical_xor", x1, x2, None, crate::dtype::Casting::Safe)
}

/// Enhanced element-wise logical NOT operation
///
/// Computes the logical NOT of the input array element-wise, treating non-zero values as True.
/// Returns a boolean array.
pub fn logical_not<T>(x: &Array<T>) -> Result<Array<bool>>
where
    T: Clone + PartialEq + Default + 'static,
{
    let data: Vec<bool> = x.iter().map(|v| *v == T::default()).collect();
    Ok(Array::from_data(data, x.shape().to_vec()))
}

/// Register all bitwise ufuncs in the registry
pub fn register_bitwise_ufuncs(registry: &mut UfuncRegistry) {
    // Register bitwise operations for i8
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &i8, b: &i8| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &i8, b: &i8| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &i8, b: &i8| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new("bitwise_not", |a: &i8| {
        a.bitwise_not()
    })));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &i8, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &i8, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for i16
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &i16, b: &i16| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &i16, b: &i16| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &i16, b: &i16| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new(
        "bitwise_not",
        |a: &i16| a.bitwise_not(),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &i16, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &i16, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for i32
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &i32, b: &i32| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &i32, b: &i32| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &i32, b: &i32| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new(
        "bitwise_not",
        |a: &i32| a.bitwise_not(),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &i32, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &i32, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for i64
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &i64, b: &i64| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &i64, b: &i64| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &i64, b: &i64| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new(
        "bitwise_not",
        |a: &i64| a.bitwise_not(),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &i64, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &i64, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for u8
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &u8, b: &u8| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &u8, b: &u8| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &u8, b: &u8| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new("bitwise_not", |a: &u8| {
        a.bitwise_not()
    })));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &u8, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &u8, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for u16
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &u16, b: &u16| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &u16, b: &u16| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &u16, b: &u16| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new(
        "bitwise_not",
        |a: &u16| a.bitwise_not(),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &u16, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &u16, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for u32
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &u32, b: &u32| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &u32, b: &u32| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &u32, b: &u32| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new(
        "bitwise_not",
        |a: &u32| a.bitwise_not(),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &u32, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &u32, shift: u32| a.right_shift(shift),
    )));

    // Register bitwise operations for u64
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_and",
        |a: &u64, b: &u64| a.bitwise_and(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_or",
        |a: &u64, b: &u64| a.bitwise_or(b),
    )));
    registry.register(Box::new(BitwiseBinaryUfunc::new(
        "bitwise_xor",
        |a: &u64, b: &u64| a.bitwise_xor(b),
    )));
    registry.register(Box::new(BitwiseUnaryUfunc::new(
        "bitwise_not",
        |a: &u64| a.bitwise_not(),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "left_shift",
        |a: &u64, shift: u32| a.left_shift(shift),
    )));
    registry.register(Box::new(BitwiseShiftUfunc::new(
        "right_shift",
        |a: &u64, shift: u32| a.right_shift(shift),
    )));
}

/// Convenience methods for Array bitwise operations
impl<T> Array<T>
where
    T: Clone + Default + BitwiseOps + 'static,
{
    /// Element-wise bitwise AND
    pub fn bitwise_and(&self, other: &Array<T>) -> Result<Array<T>> {
        bitwise_and(self, other)
    }

    /// Element-wise bitwise OR
    pub fn bitwise_or(&self, other: &Array<T>) -> Result<Array<T>> {
        bitwise_or(self, other)
    }

    /// Element-wise bitwise XOR
    pub fn bitwise_xor(&self, other: &Array<T>) -> Result<Array<T>> {
        bitwise_xor(self, other)
    }

    /// Element-wise bitwise NOT
    pub fn bitwise_not(&self) -> Result<Array<T>> {
        bitwise_not(self)
    }

    /// Left shift operation
    pub fn left_shift(&self, shift: &Array<T>) -> Result<Array<T>> {
        left_shift(self, shift)
    }

    /// Right shift operation
    pub fn right_shift(&self, shift: &Array<T>) -> Result<Array<T>> {
        right_shift(self, shift)
    }
}

/// SIMD-optimized implementations (placeholder for future optimization)
#[cfg(feature = "simd")]
mod simd_optimized {
    use super::*;

    /// SIMD-optimized bitwise AND for aligned data
    pub fn bitwise_and_simd<T>(a: &[T], b: &[T], result: &mut [T])
    where
        T: Copy + BitwiseOps + std::fmt::Debug,
    {
        // Placeholder for SIMD implementation
        // Would use std::arch intrinsics for actual optimization
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            result[i] = ai.bitwise_and(bi);
        }
    }

    /// SIMD-optimized bitwise OR for aligned data
    pub fn bitwise_or_simd<T>(a: &[T], b: &[T], result: &mut [T])
    where
        T: Copy + BitwiseOps + std::fmt::Debug,
    {
        // Placeholder for SIMD implementation
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            result[i] = ai.bitwise_or(bi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array, Array};

    #[test]
    fn test_bitwise_and() -> Result<()> {
        let a = array![5u8, 3u8, 7u8]; // 101, 011, 111
        let b = array![2u8, 6u8, 1u8]; // 010, 110, 001
        let result = bitwise_and(&a, &b)?;
        assert_eq!(result.get(0), Some(&0u8)); // 101 & 010 = 000
        assert_eq!(result.get(1), Some(&2u8)); // 011 & 110 = 010
        assert_eq!(result.get(2), Some(&1u8)); // 111 & 001 = 001
        Ok(())
    }

    #[test]
    fn test_bitwise_or() -> Result<()> {
        let a = array![5u8, 3u8, 7u8]; // 101, 011, 111
        let b = array![2u8, 6u8, 1u8]; // 010, 110, 001
        let result = bitwise_or(&a, &b)?;
        assert_eq!(result.get(0), Some(&7u8)); // 111
        assert_eq!(result.get(1), Some(&7u8)); // 111
        assert_eq!(result.get(2), Some(&7u8)); // 111
        Ok(())
    }

    #[test]
    fn test_bitwise_xor() -> Result<()> {
        let a = array![5u8, 3u8, 7u8]; // 101, 011, 111
        let b = array![2u8, 6u8, 1u8]; // 010, 110, 001
        let result = bitwise_xor(&a, &b)?;
        assert_eq!(result.get(0), Some(&7u8)); // 101 ^ 010 = 111
        assert_eq!(result.get(1), Some(&5u8)); // 011 ^ 110 = 101
        assert_eq!(result.get(2), Some(&6u8)); // 111 ^ 001 = 110
        Ok(())
    }

    #[test]
    fn test_bitwise_not() -> Result<()> {
        let a = array![0u8, 1u8, 255u8];
        let result = bitwise_not(&a)?;
        assert_eq!(result.get(0), Some(&255u8));
        assert_eq!(result.get(1), Some(&254u8));
        assert_eq!(result.get(2), Some(&0u8));
        Ok(())
    }

    #[test]
    fn test_left_shift() -> Result<()> {
        let a = array![1u8, 2u8, 3u8];
        let shifts = array![1u8, 2u8, 3u8];
        let result = left_shift(&a, &shifts)?;
        assert_eq!(result.get(0), Some(&2u8)); // 1 << 1 = 2
        assert_eq!(result.get(1), Some(&8u8)); // 2 << 2 = 8
        assert_eq!(result.get(2), Some(&24u8)); // 3 << 3 = 24
        Ok(())
    }

    #[test]
    fn test_right_shift() -> Result<()> {
        let a = array![4u8, 8u8, 16u8];
        let shifts = array![1u8, 2u8, 3u8];
        let result = right_shift(&a, &shifts)?;
        assert_eq!(result.get(0), Some(&2u8)); // 4 >> 1 = 2
        assert_eq!(result.get(1), Some(&2u8)); // 8 >> 2 = 2
        assert_eq!(result.get(2), Some(&2u8)); // 16 >> 3 = 2
        Ok(())
    }

    #[test]
    fn test_signed_right_shift() -> Result<()> {
        let a = array![-4i8, -8i8];
        let shifts = array![1u8, 2u8];
        let shifts_i8 = Array::from_vec(shifts.to_vec().into_iter().map(|x| x as i8).collect());
        let result = right_shift(&a, &shifts_i8)?;
        assert_eq!(result.get(0), Some(&-2i8)); // Arithmetic shift preserves sign
        assert_eq!(result.get(1), Some(&-2i8));
        Ok(())
    }

    #[test]
    fn test_shift_bounds() {
        let a = array![1u8];
        let shifts = array![8u8]; // Invalid: shift >= bit_width
        let result = left_shift(&a, &shifts);
        assert!(result.is_err());
    }

    #[test]
    fn test_logical_and() -> Result<()> {
        let a = array![1, 0, 3];
        let b = array![0, 2, 4];
        let result = logical_and(&a, &b)?;
        assert_eq!(result.get(0), Some(&false));
        assert_eq!(result.get(1), Some(&false));
        assert_eq!(result.get(2), Some(&true));
        Ok(())
    }

    #[test]
    fn test_logical_or() -> Result<()> {
        let a = array![1, 0, 3];
        let b = array![0, 2, 4];
        let result = logical_or(&a, &b)?;
        assert_eq!(result.get(0), Some(&true));
        assert_eq!(result.get(1), Some(&true));
        assert_eq!(result.get(2), Some(&true));
        Ok(())
    }

    #[test]
    fn test_logical_xor() -> Result<()> {
        let a = array![1, 0, 3];
        let b = array![0, 2, 4];
        let result = logical_xor(&a, &b)?;
        assert_eq!(result.get(0), Some(&true));
        assert_eq!(result.get(1), Some(&true));
        assert_eq!(result.get(2), Some(&false));
        Ok(())
    }

    #[test]
    fn test_logical_not() -> Result<()> {
        let a = array![0, 1, 2];
        let result = logical_not(&a)?;
        assert_eq!(result.get(0), Some(&true));
        assert_eq!(result.get(1), Some(&false));
        assert_eq!(result.get(2), Some(&false));
        Ok(())
    }
}
