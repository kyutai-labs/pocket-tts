use crate::array::Array;
use crate::broadcasting::{broadcast_arrays, compute_broadcast_shape};

use crate::error::{NumPyError, Result};
use crate::ufunc::UfuncRegistry;
use std::sync::Arc;

/// Ufunc execution engine
#[allow(dead_code)]
pub struct UfuncEngine {
    registry: Arc<UfuncRegistry>,
}

impl UfuncEngine {
    /// Create new ufunc engine
    pub fn new() -> Self {
        Self {
            registry: Arc::new(UfuncRegistry::new()),
        }
    }

    /// Execute binary ufunc on two arrays
    pub fn execute_binary<T>(
        &self,
        ufunc_name: &str,
        a: &Array<T>,
        b: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
    {
        let input_dtypes = vec![a.dtype().clone(), b.dtype().clone()];
        let (ufunc, _target_dtypes) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| {
                NumPyError::ufunc_error(ufunc_name, "Function not found or unsupported casting")
            })?;

        // Broadcast arrays to common shape
        let broadcasted = broadcast_arrays(&[a, b])?;

        // Create output array
        let output_shape = compute_broadcast_shape(a.shape(), b.shape());
        let mut output = Array::zeros(output_shape);

        // Create array views for ufunc execution
        let views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
            .iter()
            .map(|arr| arr as &dyn crate::ufunc::ArrayView)
            .collect();

        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        // Execute ufunc
        ufunc.execute(&views, &mut outputs, where_mask)?;

        Ok(output)
    }

    /// Execute comparison ufunc on two arrays
    pub fn execute_comparison<T>(
        &self,
        ufunc_name: &str,
        a: &Array<T>,
        b: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<bool>>
    where
        T: Clone + Default + 'static,
    {
        let input_dtypes = vec![a.dtype().clone(), b.dtype().clone()];
        let (ufunc, _target_dtypes) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| {
                NumPyError::ufunc_error(ufunc_name, "Function not found or unsupported casting")
            })?;

        let broadcasted = broadcast_arrays(&[a, b])?;

        let output_shape = compute_broadcast_shape(a.shape(), b.shape());
        let mut output = Array::<bool>::zeros(output_shape);

        let views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
            .iter()
            .map(|arr| arr as &dyn crate::ufunc::ArrayView)
            .collect();

        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        ufunc.execute(&views, &mut outputs, where_mask)?;

        Ok(output)
    }

    /// Execute unary ufunc on single array
    pub fn execute_unary_bool<T>(
        &self,
        ufunc_name: &str,
        a: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<bool>>
    where
        T: Clone + Default + 'static,
    {
        let input_dtypes = vec![a.dtype().clone()];
        let (ufunc, _target_dtypes) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| {
                NumPyError::ufunc_error(ufunc_name, "Function not found or unsupported casting")
            })?;

        let output_shape = a.shape().to_vec();
        let mut output = Array::<bool>::zeros(output_shape);

        let input_views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        ufunc.execute(&input_views, &mut outputs, where_mask)?;

        Ok(output)
    }

    pub fn execute_unary<T>(
        &self,
        ufunc_name: &str,
        a: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
    {
        let input_dtypes = vec![a.dtype().clone()];
        let (ufunc, _target_dtypes) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| {
                NumPyError::ufunc_error(ufunc_name, "Function not found or unsupported casting")
            })?;

        let mut output = a.clone();

        let input_views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        ufunc.execute(&input_views, &mut outputs, where_mask)?;

        Ok(output)
    }

    pub fn execute_reduction<T, F>(
        &self,
        _ufunc_name: &str,
        array: &Array<T>,
        axis: Option<&[isize]>,
        keepdims: bool,
        operation: F,
    ) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
        F: Fn(T, T) -> T + Send + Sync,
    {
        let output_shape = crate::broadcasting::broadcast_shape_for_reduce(
            array.shape(),
            axis.unwrap_or(&[]),
            keepdims,
        );

        let mut output = Array::zeros(output_shape);

        if let Some(reduction_axes) = axis {
            self.reduce_along_axes(array, &mut output, reduction_axes, keepdims, &operation)?;
        } else if let Some(initial) = array.get(0) {
            let mut result = initial.clone();
            for i in 1..array.size() {
                if let Some(element) = array.get(i) {
                    result = operation(result, element.clone());
                }
            }
            if output.size() == 1 {
                output.set(0, result)?;
            }
        }

        Ok(output)
    }

    fn reduce_along_axes<T, F>(
        &self,
        input: &Array<T>,
        output: &mut Array<T>,
        axes: &[isize],
        keepdims: bool,
        operation: F,
    ) -> Result<()>
    where
        T: Clone + Default + 'static,
        F: Fn(T, T) -> T + Send + Sync,
    {
        let input_shape = input.shape();
        let output_shape = output.shape().to_vec();
        let input_ndim = input.ndim();

        let reduced_axes_mask: Vec<bool> = (0..input_ndim)
            .map(|i| {
                axes.iter().any(|&ax| {
                    let norm_ax = if ax < 0 { ax + input_ndim as isize } else { ax } as usize;
                    norm_ax == i
                })
            })
            .collect();

        let mut initialized = vec![false; output.size()];

        for input_idx in 0..input.size() {
            let input_indices = crate::strides::compute_multi_indices(input_idx, input_shape);

            // Map input indices to output indices
            let mut output_indices = Vec::with_capacity(output.ndim());
            for (dim_idx, &idx_val) in input_indices.iter().enumerate() {
                if !reduced_axes_mask[dim_idx] {
                    output_indices.push(idx_val);
                } else if keepdims {
                    output_indices.push(0);
                }
            }

            if let Ok(element) = input.get_by_indices(&input_indices) {
                let out_strides = crate::strides::compute_strides(&output_shape);
                let out_linear_idx =
                    crate::strides::compute_linear_index(&output_indices, &out_strides) as usize;

                if !initialized[out_linear_idx] {
                    output.set_linear(out_linear_idx, element.clone());
                    initialized[out_linear_idx] = true;
                } else {
                    if let Some(current_val) = output.get(out_linear_idx) {
                        output.set_linear(
                            out_linear_idx,
                            operation(current_val.clone(), element.clone()),
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for UfuncEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level ufunc operations for Array
impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Element-wise addition
    pub fn add(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("add", self, other, where_mask, casting)
    }

    /// Element-wise subtraction
    pub fn subtract(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("subtract", self, other, where_mask, casting)
    }

    /// Element-wise multiplication
    pub fn multiply(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("multiply", self, other, where_mask, casting)
    }

    /// Element-wise division
    pub fn divide(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("divide", self, other, where_mask, casting)
    }

    /// Element-wise negation
    pub fn negative(
        &self,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_unary("negative", self, where_mask, casting)
    }

    /// Absolute value
    pub fn abs(
        &self,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_unary("absolute", self, where_mask, casting)
    }

    /// Cumulative sum, treating NaNs as zero.
    pub fn nancumsum(&self, axis: Option<isize>) -> Result<Array<T>>
    where
        T: Clone + Default + std::ops::Add<Output = T> + num_traits::Float,
    {
        match axis {
            None => {
                let mut data = Vec::with_capacity(self.size());
                let mut running_sum = T::zero();

                for i in 0..self.size() {
                    if let Some(val) = self.get(i) {
                        if !val.is_nan() {
                            running_sum = running_sum + val.clone();
                        }
                        data.push(running_sum);
                    }
                }

                Ok(Array::from_vec(data))
            }
            Some(ax) => {
                let ax = if ax < 0 {
                    ax + self.ndim() as isize
                } else {
                    ax
                } as usize;

                if ax >= self.ndim() {
                    return Err(NumPyError::index_error(ax, self.ndim()));
                }

                let mut result = Array::zeros(self.shape().to_vec());

                let stride_before = if ax > 0 {
                    self.shape()[..ax].iter().product::<usize>()
                } else {
                    1
                };
                let stride_after = if ax + 1 < self.ndim() {
                    self.shape()[ax + 1..].iter().product::<usize>()
                } else {
                    1
                };
                let axis_size = self.shape()[ax];

                for outer in 0..stride_before {
                    for inner in 0..stride_after {
                        let mut running = T::zero();
                        for pos in 0..axis_size {
                            let idx = outer * axis_size * stride_after + pos * stride_after + inner;
                            if let Some(val) = self.get(idx) {
                                if !val.is_nan() {
                                    running = running + val.clone();
                                }
                                result.set(idx, running)?;
                            }
                        }
                    }
                }

                Ok(result)
            }
        }
    }

    /// Cumulative product, treating NaNs as one.
    pub fn nancumprod(&self, axis: Option<isize>) -> Result<Array<T>>
    where
        T: Clone + Default + std::ops::Mul<Output = T> + num_traits::Float,
    {
        match axis {
            None => {
                let mut data = Vec::with_capacity(self.size());
                let mut running_prod = T::one();

                for i in 0..self.size() {
                    if let Some(val) = self.get(i) {
                        if !val.is_nan() {
                            running_prod = running_prod * val.clone();
                        }
                        data.push(running_prod);
                    }
                }

                Ok(Array::from_vec(data))
            }
            Some(ax) => {
                let ax = if ax < 0 {
                    ax + self.ndim() as isize
                } else {
                    ax
                } as usize;

                if ax >= self.ndim() {
                    return Err(NumPyError::index_error(ax, self.ndim()));
                }

                let mut result = Array::zeros(self.shape().to_vec());

                let stride_before = if ax > 0 {
                    self.shape()[..ax].iter().product::<usize>()
                } else {
                    1
                };
                let stride_after = if ax + 1 < self.ndim() {
                    self.shape()[ax + 1..].iter().product::<usize>()
                } else {
                    1
                };
                let axis_size = self.shape()[ax];

                for outer in 0..stride_before {
                    for inner in 0..stride_after {
                        let mut running = T::one();
                        for pos in 0..axis_size {
                            let idx = outer * axis_size * stride_after + pos * stride_after + inner;
                            if let Some(val) = self.get(idx) {
                                if !val.is_nan() {
                                    running = running * val.clone();
                                }
                                result.set(idx, running)?;
                            }
                        }
                    }
                }

                Ok(result)
            }
        }
    }

    fn get_mean_for_index(
        mean_array: &Array<f64>,
        linear_idx: usize,
        original_shape: &[usize],
    ) -> f64 {
        if mean_array.size() == 1 {
            return *mean_array.get(0).unwrap();
        }

        let indices = crate::strides::compute_multi_indices(linear_idx, original_shape);

        let mut reduced_indices = Vec::new();
        for (i, &val) in indices.iter().enumerate() {
            if i < mean_array.ndim() {
                reduced_indices.push(val);
            }
        }

        if reduced_indices.is_empty() {
            return *mean_array.get(0).unwrap();
        }

        while reduced_indices.len() < mean_array.ndim() {
            reduced_indices.push(0);
        }

        *mean_array.get_by_indices(&reduced_indices).unwrap_or(&0.0)
    }
}

/// Ufunc operation traits for different data types
pub trait UfuncOps<T>: Send + Sync {
    fn add(a: &T, b: &T) -> T;
    fn subtract(a: &T, b: &T) -> T;
    fn multiply(a: &T, b: &T) -> T;
    fn divide(a: &T, b: &T) -> T;
    fn negative(a: &T) -> T;
    fn absolute(a: &T) -> T;
}

/// Implement ufunc operations for basic numeric types
macro_rules! impl_signed_ufunc_ops {
    ($($t:ty),*) => {
        $(
            impl UfuncOps<$t> for $t {
                fn add(a: &$t, b: &$t) -> $t { a + b }
                fn subtract(a: &$t, b: &$t) -> $t { a - b }
                fn multiply(a: &$t, b: &$t) -> $t { a * b }
                fn divide(a: &$t, b: &$t) -> $t { a / b }
                fn negative(a: &$t) -> $t { -a }
                fn absolute(a: &$t) -> $t { a.abs() }
            }
        )*
    }
}

macro_rules! impl_unsigned_ufunc_ops {
    ($($t:ty),*) => {
        $(
            impl UfuncOps<$t> for $t {
                fn add(a: &$t, b: &$t) -> $t { a + b }
                fn subtract(a: &$t, b: &$t) -> $t { a - b }
                fn multiply(a: &$t, b: &$t) -> $t { a * b }
                fn divide(a: &$t, b: &$t) -> $t { a / b }
                fn negative(a: &$t) -> $t { 0_u8.wrapping_sub(1) as $t * a } // Not ideal but satisfies trait
                fn absolute(a: &$t) -> $t { *a }
            }
        )*
    }
}

impl_signed_ufunc_ops!(f64, f32, i64, i32, i16, i8);
impl_unsigned_ufunc_ops!(u64, u32, u16, u8);

impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    pub fn greater(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("greater", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn less(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("less", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn greater_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison(
            "greater_equal",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }

    pub fn less_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("less_equal", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("equal", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn not_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("not_equal", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn maximum(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("maximum", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn minimum(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("minimum", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn logical_and(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison(
            "logical_and",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }

    pub fn logical_or(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("logical_or", self, other, None, crate::dtype::Casting::Safe)
    }

    pub fn logical_xor(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison(
            "logical_xor",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }

    pub fn logical_not(&self) -> Result<Array<bool>>
    where
        T: PartialEq + Clone + Default + 'static,
    {
        let engine = UfuncEngine::new();
        engine.execute_unary_bool("logical_not", self, None, crate::dtype::Casting::Safe)
    }
}
