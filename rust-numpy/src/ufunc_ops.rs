use crate::array::Array;
use crate::broadcasting::{broadcast_arrays, compute_broadcast_shape};

use crate::error::{NumPyError, Result};
use crate::ufunc::{get_ufunc, get_ufunc_typed, get_ufunc_typed_binary, UfuncRegistry};
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
    ) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
    {
        let ufunc = get_ufunc_typed_binary::<T>(ufunc_name)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Function not found"))?;

        // Check if ufunc supports the dtype
        if !ufunc.supports_dtypes(&[a.dtype(), b.dtype()]) {
            return Err(NumPyError::ufunc_error(
                ufunc_name,
                format!(
                    "Unsupported dtype combination: {:?} and {:?}",
                    a.dtype(),
                    b.dtype()
                ),
            ));
        }

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
        ufunc.execute(&views, &mut outputs)?;

        Ok(output)
    }

    /// Execute unary ufunc on single array
    pub fn execute_unary<T>(&self, ufunc_name: &str, a: &Array<T>) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
    {
        let ufunc = get_ufunc_typed::<T>(ufunc_name)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Function not found"))?;

        if !ufunc.supports_dtypes(&[a.dtype()]) {
            return Err(NumPyError::ufunc_error(
                ufunc_name,
                format!("Unsupported dtype: {:?}", a.dtype()),
            ));
        }

        let mut output = a.clone();

        let input_views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        ufunc.execute(&input_views, &mut outputs)?;

        Ok(output)
    }

    pub fn execute_comparison<T>(
        &self,
        ufunc_name: &str,
        a: &Array<T>,
        b: &Array<T>,
    ) -> Result<Array<bool>>
    where
        T: Clone + Default + 'static,
    {
        let ufunc = get_ufunc_typed_binary::<T>(ufunc_name)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Function not found"))?;

        if !ufunc.supports_dtypes(&[a.dtype(), b.dtype()]) {
            return Err(NumPyError::ufunc_error(
                ufunc_name,
                format!(
                    "Unsupported dtype combination: {:?} and {:?}",
                    a.dtype(),
                    b.dtype()
                ),
            ));
        }

        let broadcasted = broadcast_arrays(&[a, b])?;

        let output_shape = compute_broadcast_shape(a.shape(), b.shape());
        let mut output = Array::<bool>::zeros(output_shape);

        let views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
            .iter()
            .map(|arr| arr as &dyn crate::ufunc::ArrayView)
            .collect();

        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        ufunc.execute(&views, &mut outputs)?;

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
            self.reduce_along_axes(array, &mut output, reduction_axes, &operation)?;
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

        // Initialize output with default or first elements if necessary
        // In many cases, the caller might have initialized it, but let's ensure it's handled.
        // For simplicity in this O(N) pass, we'll use a tracker to know if it's the first element for that output slot.
        let mut initialized = vec![false; output.size()];

        for input_idx in 0..input.size() {
            let input_indices = crate::strides::compute_multi_indices(input_idx, input_shape);

            // Map input indices to output indices
            let mut output_indices = Vec::with_capacity(output.ndim());
            for (dim_idx, &idx_val) in input_indices.iter().enumerate() {
                if !reduced_axes_mask[dim_idx] {
                    output_indices.push(idx_val);
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
    pub fn add(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("add", self, other)
    }

    /// Element-wise subtraction
    pub fn subtract(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("subtract", self, other)
    }

    /// Element-wise multiplication
    pub fn multiply(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("multiply", self, other)
    }

    /// Element-wise division
    pub fn divide(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("divide", self, other)
    }

    /// Element-wise negation
    pub fn negative(&self) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_unary("negative", self)
    }

    /// Absolute value
    pub fn abs(&self) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_unary("absolute", self)
    }

    /// Sum of elements
    pub fn sum(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: std::ops::Add<Output = T>,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction("sum", self, axis, keepdims, |a, b| a + b)
    }

    pub fn product(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: std::ops::Mul<Output = T>,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction("product", self, axis, keepdims, |a, b| a * b)
    }

    pub fn count_reduced_elements(
        &self,
        axis: Option<&[isize]>,
        keepdims: bool,
    ) -> Result<Array<i64>> {
        use crate::Array;

        let output_shape = crate::broadcasting::broadcast_shape_for_reduce(
            self.shape(),
            axis.unwrap_or(&[]),
            keepdims,
        );

        let count = if let Some(reduction_axes) = axis {
            let mut count_val = 1i64;
            for &ax in reduction_axes {
                let ax = if ax < 0 {
                    ax + self.ndim() as isize
                } else {
                    ax
                } as usize;

                if ax < self.ndim() {
                    count_val *= self.shape()[ax] as i64;
                }
            }
            count_val
        } else {
            self.size() as i64
        };

        Ok(Array::full(output_shape, count))
    }

    /// Minimum of elements
    pub fn min(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: PartialOrd,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction(
            "min",
            self,
            axis,
            keepdims,
            |a, b| if a < b { a } else { b },
        )
    }

    /// Maximum of elements
    pub fn max(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: PartialOrd,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction(
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
        let sum_result = self.sum(axis, keepdims)?;
        let count = self.count_reduced_elements(axis, keepdims)?;

        let mut result = Array::<f64>::zeros(sum_result.shape().to_vec());
        for i in 0..result.size() {
            if let (Some(sum_val), Some(&cnt)) = (sum_result.get(i), count.get(i)) {
                result.set(i, sum_val.clone().into() / cnt as f64)?;
            }
        }

        Ok(result)
    }

    pub fn var(&self, axis: Option<&[isize]>, keepdims: bool, _skipna: bool) -> Result<Array<f64>>
    where
        T: Into<f64> + Clone + std::ops::Add<Output = T>,
    {
        let mean_val = self.mean(axis, keepdims)?;

        let mut squared_deviations = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            if let (Some(val), mean_f64) = (
                self.get(i),
                Self::get_mean_for_index(&mean_val, i, self.shape()),
            ) {
                let diff = val.clone().into() - mean_f64;
                squared_deviations.push(diff * diff);
            }
        }

        let dev_array = Array::from_shape_vec(self.shape().to_vec(), squared_deviations);

        dev_array.mean(axis, keepdims)
    }

    pub fn std(&self, axis: Option<&[isize]>, keepdims: bool, skipna: bool) -> Result<Array<f64>>
    where
        T: Into<f64> + Clone + std::ops::Add<Output = T>,
    {
        let variance = self.var(axis, keepdims, skipna)?;

        let mut result = Array::<f64>::zeros(variance.shape().to_vec());
        for i in 0..variance.size() {
            if let Some(&var_val) = variance.get(i) {
                result.set(i, var_val.sqrt())?;
            }
        }

        Ok(result)
    }

    pub fn ptp(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: Clone + Default + PartialOrd + std::ops::Sub<Output = T>,
    {
        let max_val = self.max(axis, keepdims)?;
        let min_val = self.min(axis, keepdims)?;

        let mut result = Array::zeros(max_val.shape().to_vec());
        for i in 0..result.size() {
            if let (Some(max_elem), Some(min_elem)) = (max_val.get(i), min_val.get(i)) {
                result.set(i, max_elem.clone() - min_elem.clone())?;
            }
        }

        Ok(result)
    }

    pub fn argmin(&self, axis: Option<isize>) -> Result<Array<usize>>
    where
        T: PartialOrd + Clone,
    {
        match axis {
            None => {
                let mut min_idx = 0;
                let mut min_val = self.get(0);

                for i in 1..self.size() {
                    if let Some(val) = self.get(i) {
                        if let Some(current_min) = min_val {
                            if val < current_min {
                                min_idx = i;
                                min_val = Some(val);
                            }
                        }
                    }
                }

                Ok(Array::from_vec(vec![min_idx]))
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

                let output_shape: Vec<usize> = self
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &dim)| dim)
                    .collect();

                if output_shape.is_empty() {
                    return Err(NumPyError::invalid_operation("Cannot reduce all axes"));
                }

                let mut result = Array::zeros(output_shape.clone());

                for output_idx in 0..result.size() {
                    let output_indices =
                        crate::strides::compute_multi_indices(output_idx, &output_shape);

                    let mut min_pos = 0;
                    let mut min_val: Option<&T> = None;

                    for pos in 0..self.shape()[ax] {
                        let mut full_indices = output_indices.clone();
                        full_indices.insert(ax, pos);

                        if let Ok(val) = self.get_by_indices(&full_indices) {
                            if min_val.is_none() || val < min_val.unwrap() {
                                min_pos = pos;
                                min_val = Some(val);
                            }
                        }
                    }

                    result.set(output_idx, min_pos)?;
                }

                Ok(result)
            }
        }
    }

    pub fn argmax(&self, axis: Option<isize>) -> Result<Array<usize>>
    where
        T: PartialOrd + Clone,
    {
        match axis {
            None => {
                let mut max_idx = 0;
                let mut max_val = self.get(0);

                for i in 1..self.size() {
                    if let Some(val) = self.get(i) {
                        if let Some(current_max) = max_val {
                            if val > current_max {
                                max_idx = i;
                                max_val = Some(val);
                            }
                        }
                    }
                }

                Ok(Array::from_vec(vec![max_idx]))
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

                let output_shape: Vec<usize> = self
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &dim)| dim)
                    .collect();

                if output_shape.is_empty() {
                    return Err(NumPyError::invalid_operation("Cannot reduce all axes"));
                }

                let mut result = Array::zeros(output_shape.clone());

                for output_idx in 0..result.size() {
                    let output_indices =
                        crate::strides::compute_multi_indices(output_idx, &output_shape);

                    let mut max_pos = 0;
                    let mut max_val: Option<&T> = None;

                    for pos in 0..self.shape()[ax] {
                        let mut full_indices = output_indices.clone();
                        full_indices.insert(ax, pos);

                        if let Ok(val) = self.get_by_indices(&full_indices) {
                            if max_val.is_none() || val > max_val.unwrap() {
                                max_pos = pos;
                                max_val = Some(val);
                            }
                        }
                    }

                    result.set(output_idx, max_pos)?;
                }

                Ok(result)
            }
        }
    }

    pub fn all(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<bool>>
    where
        T: Clone + Default + Into<bool> + 'static,
    {
        match axis {
            None => {
                let result = (0..self.size())
                    .all(|i| self.get(i).map(|v| v.clone().into()).unwrap_or(false));
                Ok(Array::from_scalar(result, vec![]))
            }
            Some(axes) => {
                let engine = UfuncEngine::new();
                // We need to map T to bool first, or use a custom reduction
                // Since execute_reduction expects T -> T, let's just do it manually for now
                // to match the existing pattern or implement a specialized logical reduction.

                let output_shape =
                    crate::broadcasting::broadcast_shape_for_reduce(self.shape(), axes, keepdims);
                let mut output = Array::<bool>::full(output_shape, true);

                let input_shape = self.shape();
                let output_shape_vec = output.shape().to_vec();

                let normalized_axes: Vec<usize> = axes
                    .iter()
                    .map(|&ax| {
                        if ax < 0 {
                            (ax + input_shape.len() as isize) as usize
                        } else {
                            ax as usize
                        }
                    })
                    .collect();

                for out_idx in 0..output.size() {
                    let out_indices =
                        crate::strides::compute_multi_indices(out_idx, &output_shape_vec);
                    let mut current_all = true;

                    for in_idx in 0..self.size() {
                        let in_indices = crate::strides::compute_multi_indices(in_idx, input_shape);

                        let mut reduced_match = true;
                        for i in 0..input_shape.len() {
                            if !normalized_axes.contains(&i) {
                                let out_ax_idx = if keepdims {
                                    i
                                } else {
                                    let mut count = 0;
                                    for j in 0..i {
                                        if !normalized_axes.contains(&j) {
                                            count += 1;
                                        }
                                    }
                                    count
                                };
                                if in_indices[i] != out_indices[out_ax_idx] {
                                    reduced_match = false;
                                    break;
                                }
                            }
                        }

                        if reduced_match {
                            if let Some(val) = self.get(in_idx) {
                                if !val.clone().into() {
                                    current_all = false;
                                    break;
                                }
                            }
                        }
                    }
                    output.set_linear(out_idx, current_all);
                }
                Ok(output)
            }
        }
    }

    pub fn map<U, F>(&self, f: F) -> Array<U>
    where
        U: Clone + Default + 'static,
        F: Fn(&T) -> U,
    {
        let mut new_data = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            if let Some(val) = self.get(i) {
                new_data.push(f(val));
            }
        }
        Array::from_data(new_data, self.shape().to_vec())
    }

    pub fn any(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<bool>>
    where
        T: Clone + Default + Into<bool> + 'static,
    {
        match axis {
            None => {
                let result = (0..self.size())
                    .any(|i| self.get(i).map(|v| v.clone().into()).unwrap_or(false));
                Ok(Array::from_scalar(result, vec![]))
            }
            Some(axes) => {
                let engine = UfuncEngine::new();
                let output_shape =
                    crate::broadcasting::broadcast_shape_for_reduce(self.shape(), axes, keepdims);
                let mut output = Array::<bool>::full(output_shape, false);

                let input_shape = self.shape();
                let output_shape_vec = output.shape().to_vec();

                let normalized_axes: Vec<usize> = axes
                    .iter()
                    .map(|&ax| {
                        if ax < 0 {
                            (ax + input_shape.len() as isize) as usize
                        } else {
                            ax as usize
                        }
                    })
                    .collect();

                for out_idx in 0..output.size() {
                    let out_indices =
                        crate::strides::compute_multi_indices(out_idx, &output_shape_vec);
                    let mut current_any = false;

                    for in_idx in 0..self.size() {
                        let in_indices = crate::strides::compute_multi_indices(in_idx, input_shape);

                        let mut reduced_match = true;
                        for i in 0..input_shape.len() {
                            if !normalized_axes.contains(&i) {
                                let out_ax_idx = if keepdims {
                                    i
                                } else {
                                    let mut count = 0;
                                    for j in 0..i {
                                        if !normalized_axes.contains(&j) {
                                            count += 1;
                                        }
                                    }
                                    count
                                };
                                if in_indices[i] != out_indices[out_ax_idx] {
                                    reduced_match = false;
                                    break;
                                }
                            }
                        }

                        if reduced_match {
                            if let Some(val) = self.get(in_idx) {
                                if val.clone().into() {
                                    current_any = true;
                                    break;
                                }
                            }
                        }
                    }
                    output.set_linear(out_idx, current_any);
                }
                Ok(output)
            }
        }
    }

    pub fn cumsum(&self, axis: Option<isize>) -> Result<Array<T>>
    where
        T: Clone + Default + std::ops::Add<Output = T>,
    {
        match axis {
            None => {
                let mut data = Vec::with_capacity(self.size());
                let mut running_sum = T::default();

                for i in 0..self.size() {
                    if let Some(val) = self.get(i) {
                        running_sum = running_sum + val.clone();
                        data.push(running_sum.clone());
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
                        let mut running = T::default();
                        for pos in 0..axis_size {
                            let idx = outer * axis_size * stride_after + pos * stride_after + inner;
                            if let Some(val) = self.get(idx) {
                                running = running + val.clone();
                                result.set(idx, running.clone())?;
                            }
                        }
                    }
                }

                Ok(result)
            }
        }
    }

    pub fn cumprod(&self, axis: Option<isize>) -> Result<Array<T>>
    where
        T: Clone + Default + std::ops::Mul<Output = T>,
    {
        match axis {
            None => {
                let mut data = Vec::with_capacity(self.size());
                let mut running_prod = T::default();

                for i in 0..self.size() {
                    if let Some(val) = self.get(i) {
                        if i == 0 {
                            running_prod = val.clone();
                        } else {
                            running_prod = running_prod * val.clone();
                        }
                        data.push(running_prod.clone());
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
                        let mut running: Option<T> = None;
                        for pos in 0..axis_size {
                            let idx = outer * axis_size * stride_after + pos * stride_after + inner;
                            if let Some(val) = self.get(idx) {
                                match &mut running {
                                    None => {
                                        let v = val.clone();
                                        result.set(idx, v.clone())?;
                                        running = Some(v);
                                    }
                                    Some(r) => {
                                        *r = r.clone() * val.clone();
                                        result.set(idx, r.clone())?;
                                    }
                                }
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
        engine.execute_comparison("greater", self, other)
    }

    pub fn less(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("less", self, other)
    }

    pub fn greater_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("greater_equal", self, other)
    }

    pub fn less_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("less_equal", self, other)
    }

    pub fn equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("equal", self, other)
    }

    pub fn not_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("not_equal", self, other)
    }

    pub fn maximum(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("maximum", self, other)
    }

    pub fn minimum(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("minimum", self, other)
    }

    pub fn logical_and(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("logical_and", self, other)
    }

    pub fn logical_or(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("logical_or", self, other)
    }

    pub fn logical_xor(&self, other: &Array<T>) -> Result<Array<bool>> {
        let engine = UfuncEngine::new();
        engine.execute_comparison("logical_xor", self, other)
    }

    pub fn logical_not(&self) -> Result<Array<bool>>
    where
        T: PartialEq + Clone + Default + 'static,
    {
        let _ufunc = get_ufunc("logical_not")
            .ok_or_else(|| NumPyError::ufunc_error("logical_not", "Function not found"))?;

        let output_shape = self.shape().to_vec();
        let mut output = Array::<bool>::zeros(output_shape);

        for i in 0..self.size() {
            if let Some(val) = self.get(i) {
                output.set(i, val.clone() == T::default())?;
            }
        }

        Ok(output)
    }
}
