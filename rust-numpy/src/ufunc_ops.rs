use crate::array::Array;
use crate::broadcasting::{broadcast_arrays, compute_broadcast_shape};
use crate::error::{NumPyError, Result};
use crate::ufunc::{UfuncRegistry, UFUNC_REGISTRY};
use std::sync::Arc;

pub struct UfuncEngine {
    registry: Arc<UfuncRegistry>,
}
impl UfuncEngine {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(UfuncRegistry::new()),
        }
    }
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
        let (ufunc, _) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Not found"))?;
        let broadcasted = broadcast_arrays(&[a, b])?;
        let mut output = Array::zeros(compute_broadcast_shape(a.shape(), b.shape()));
        let views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
            .iter()
            .map(|arr| arr as &dyn crate::ufunc::ArrayView)
            .collect();
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];
        ufunc.execute(&views, &mut outputs, where_mask)?;
        Ok(output)
    }
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
        let (ufunc, _) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Not found"))?;
        let broadcasted = broadcast_arrays(&[a, b])?;
        let mut output = Array::<bool>::zeros(compute_broadcast_shape(a.shape(), b.shape()));
        let views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
            .iter()
            .map(|arr| arr as &dyn crate::ufunc::ArrayView)
            .collect();
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];
        ufunc.execute(&views, &mut outputs, where_mask)?;
        Ok(output)
    }
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
        let (ufunc, _) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Not found"))?;
        let mut output = Array::<bool>::zeros(a.shape().to_vec());
        let views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];
        ufunc.execute(&views, &mut outputs, where_mask)?;
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
        let (ufunc, _) = self
            .registry
            .resolve_ufunc(ufunc_name, &input_dtypes, casting)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Not found"))?;
        let mut output = a.clone();
        let views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];
        ufunc.execute(&views, &mut outputs, where_mask)?;
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
        let mut output = Array::zeros(crate::broadcasting::broadcast_shape_for_reduce(
            array.shape(),
            axis.unwrap_or(&[]),
            keepdims,
        ));
        if let Some(axes) = axis {
            self.reduce_along_axes(array, &mut output, axes, keepdims, operation)?;
        } else if let Some(initial) = array.get(0) {
            let mut res = initial.clone();
            for i in 1..array.size() {
                if let Some(e) = array.get(i) {
                    res = operation(res, e.clone());
                }
            }
            if output.size() == 1 {
                output.set(0, res)?;
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
            let input_indices = crate::strides::compute_multi_indices(input_idx, input.shape());
            let mut output_indices = Vec::with_capacity(output.ndim());
            for (dim_idx, &idx_val) in input_indices.iter().enumerate() {
                if !reduced_axes_mask[dim_idx] {
                    output_indices.push(idx_val);
                } else if keepdims {
                    output_indices.push(0);
                }
            }
            if let Ok(element) = input.get_by_indices(&input_indices) {
                let out_linear_idx = crate::strides::compute_linear_index(
                    &output_indices,
                    &crate::strides::compute_strides(&output.shape()),
                ) as usize;
                if !initialized[out_linear_idx] {
                    output.set_linear(out_linear_idx, element.clone());
                    initialized[out_linear_idx] = true;
                } else if let Some(curr) = output.get(out_linear_idx) {
                    output.set_linear(out_linear_idx, operation(curr.clone(), element.clone()));
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

impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    pub fn add(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        UfuncEngine::new().execute_binary("add", self, other, where_mask, casting)
    }
    pub fn subtract(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        UfuncEngine::new().execute_binary("subtract", self, other, where_mask, casting)
    }
    pub fn multiply(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        UfuncEngine::new().execute_binary("multiply", self, other, where_mask, casting)
    }
    pub fn divide(
        &self,
        other: &Array<T>,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        UfuncEngine::new().execute_binary("divide", self, other, where_mask, casting)
    }
    pub fn negative(
        &self,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        UfuncEngine::new().execute_unary("negative", self, where_mask, casting)
    }
    pub fn abs(
        &self,
        where_mask: Option<&Array<bool>>,
        casting: crate::dtype::Casting,
    ) -> Result<Array<T>> {
        UfuncEngine::new().execute_unary("absolute", self, where_mask, casting)
    }
    pub fn sum(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: std::ops::Add<Output = T>,
    {
        UfuncEngine::new().execute_reduction("sum", self, axis, keepdims, |a, b| a + b)
    }
    pub fn prod(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: std::ops::Mul<Output = T>,
    {
        UfuncEngine::new().execute_reduction("prod", self, axis, keepdims, |a, b| a * b)
    }
    pub fn min(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: PartialOrd,
    {
        UfuncEngine::new().execute_reduction(
            "min",
            self,
            axis,
            keepdims,
            |a, b| if a < b { a } else { b },
        )
    }
    pub fn max(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: PartialOrd,
    {
        UfuncEngine::new().execute_reduction(
            "max",
            self,
            axis,
            keepdims,
            |a, b| if a > b { a } else { b },
        )
    }
    pub fn mean(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<f64>>
    where
        T: Into<f64> + Clone + std::ops::Add<Output = T>,
    {
        let sum_res = self.sum(axis, keepdims)?;
        let count = self.count_reduced_elements(axis, keepdims)?;
        let mut result = Array::<f64>::zeros(sum_res.shape().to_vec());
        for i in 0..result.size() {
            if let (Some(s), Some(&c)) = (sum_res.get(i), count.get(i)) {
                result.set(i, s.clone().into() / c as f64)?;
            }
        }
        Ok(result)
    }
    pub fn count_reduced_elements(
        &self,
        axis: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<i64>> {
        let output_shape = crate::broadcasting::broadcast_shape_for_reduce(
            self.shape(),
            axis.unwrap_or(&[]),
            keepdims,
        );
        let count = if let Some(axes) = axis {
            let mut c = 1i64;
            for &ax in axes {
                let ax = if ax < 0 {
                    ax + self.ndim() as isize
                } else {
                    ax
                } as usize;
                if ax < self.ndim() {
                    c *= self.shape()[ax] as i64;
                }
            }
            c
        } else {
            self.size() as i64
        };
        Ok(Array::full(output_shape, count))
    }
    pub fn greater(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "greater",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn less(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "less",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "equal",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn not_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "not_equal",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn greater_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "greater_equal",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn less_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "less_equal",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn logical_and(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "logical_and",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn logical_or(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
            "logical_or",
            self,
            other,
            None,
            crate::dtype::Casting::Safe,
        )
    }
    pub fn logical_xor(&self, other: &Array<T>) -> Result<Array<bool>> {
        UfuncEngine::new().execute_comparison(
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
        UfuncEngine::new().execute_unary_bool(
            "logical_not",
            self,
            None,
            crate::dtype::Casting::Safe,
        )
    }
}
