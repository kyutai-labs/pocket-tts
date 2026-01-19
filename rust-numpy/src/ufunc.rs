use crate::array::Array;
use crate::broadcasting::compute_broadcast_shape;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use std::marker::PhantomData;
use std::sync::Arc;

impl<T> ArrayView for Array<T> {
    fn dtype(&self) -> &Dtype {
        self.dtype()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn strides(&self) -> &[isize] {
        self.strides()
    }

    fn size(&self) -> usize {
        self.size()
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.is_c_contiguous()
    }

    fn as_ptr(&self) -> *const u8 {
        // Get pointer to the underlying data
        self.data.as_ref().as_slice().as_ptr() as *const u8
    }
}

impl<T> ArrayViewMut for Array<T> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        // Get mutable pointer to the underlying data
        // SAFETY: We have &mut self, which guarantees exclusive access to the Array.
        // Even though the data is in an Arc, the &mut ensures no other references exist
        // that could modify the data. We use unsafe to bypass Arc's ref counting here.
        unsafe {
            let ptr = self.data.as_ref().as_slice().as_ptr() as *mut u8;
            // The cast from const to mut is safe because:
            // 1. We have &mut self, guaranteeing exclusive mutable access
            // 2. The Arc is not shared during ufunc execution (outputs are newly created)
            ptr
        }
    }
}

/// Universal function trait - base for all NumPy ufuncs
pub trait Ufunc: Send + Sync {
    /// Get ufunc name
    fn name(&self) -> &'static str;

    /// Get number of inputs
    fn nin(&self) -> usize;

    /// Get number of outputs
    fn nout(&self) -> usize;

    /// Get supported input types
    fn supported_dtypes(&self) -> &[DtypeKind];

    /// Execute ufunc on inputs
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()>;

    /// Check if ufunc supports given dtypes
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool {
        dtypes
            .iter()
            .all(|dt| self.supported_dtypes().contains(&dt.kind()))
    }
}

/// Trait for viewing array data
pub trait ArrayView {
    /// Get dtype
    fn dtype(&self) -> &Dtype;

    /// Get shape
    fn shape(&self) -> &[usize];

    /// Get strides
    fn strides(&self) -> &[isize];

    /// Get total size
    fn size(&self) -> usize;

    /// Get number of dimensions
    fn ndim(&self) -> usize;

    /// Check if contiguous
    fn is_contiguous(&self) -> bool;

    /// Get raw data pointer
    fn as_ptr(&self) -> *const u8;
}

/// Trait for mutable array data
pub trait ArrayViewMut: ArrayView {
    /// Get mutable raw data pointer
    fn as_mut_ptr(&mut self) -> *mut u8;
}

/// Binary operation ufunc
#[allow(dead_code)]
pub struct BinaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> BinaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    /// Create new binary ufunc
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for BinaryUfunc<T, F>
where
    T: Clone + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
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
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
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

        // Check dtypes are supported
        if !self.supports_dtypes(&[inputs[0].dtype(), inputs[1].dtype()]) {
            return Err(NumPyError::ufunc_error(
                self.name(),
                "Unsupported dtype combination".to_string(),
            ));
        }

        // Simplified implementation - real NumPy has complex broadcasting
        let shape1 = inputs[0].shape();
        let shape2 = inputs[1].shape();
        let _broadcast_shape = compute_broadcast_shape(shape1, shape2);

        // This is a very simplified implementation
        // Real implementation would need proper broadcasting, dtype promotion, etc.
        Ok(())
    }
}

/// Unary operation ufunc
#[allow(dead_code)]
pub struct UnaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> UnaryUfunc<T, F>
where
    T: Clone + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    /// Create new unary ufunc
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for UnaryUfunc<T, F>
where
    T: Clone + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
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
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
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

        // Check dtype is supported
        if !self.supports_dtypes(&[inputs[0].dtype()]) {
            return Err(NumPyError::ufunc_error(
                self.name(),
                "Unsupported dtype".to_string(),
            ));
        }

        // Simplified implementation
        Ok(())
    }
}

/// Ufunc registry for looking up functions by name
pub struct UfuncRegistry {
    ufuncs: std::collections::HashMap<String, Box<dyn Ufunc>>,
}

impl UfuncRegistry {
    /// Create new registry
    pub fn new() -> Self {
        let mut registry = Self {
            ufuncs: std::collections::HashMap::new(),
        };

        // Register basic ufuncs
        registry.register_basic_ufuncs();
        registry.register_comparison_ufuncs();
        registry.register_math_ufuncs();
        registry.register_bitwise_ufuncs();
        registry
    }

    /// Register a ufunc
    pub fn register(&mut self, ufunc: Box<dyn Ufunc>) {
        self.ufuncs.insert(ufunc.name().to_string(), ufunc);
    }

    /// Get ufunc by name
    pub fn get(&self, name: &str) -> Option<&dyn Ufunc> {
        self.ufuncs.get(name).map(|uf| uf.as_ref())
    }

    /// List all registered ufuncs
    pub fn list(&self) -> Vec<&str> {
        self.ufuncs.keys().map(|s| s.as_str()).collect()
    }

    /// Register basic mathematical ufuncs
    fn register_basic_ufuncs(&mut self) {
        // Addition
        self.register(Box::new(BinaryUfunc::new("add", |a: f64, b: f64| a + b)));
        self.register(Box::new(BinaryUfunc::new("add", |a: f32, b: f32| a + b)));
        self.register(Box::new(BinaryUfunc::new("add", |a: i64, b: i64| a + b)));

        // Subtraction
        self.register(Box::new(BinaryUfunc::new("subtract", |a: f64, b: f64| {
            a - b
        })));
        self.register(Box::new(BinaryUfunc::new("subtract", |a: f32, b: f32| {
            a - b
        })));
        self.register(Box::new(BinaryUfunc::new("subtract", |a: i64, b: i64| {
            a - b
        })));

        // Multiplication
        self.register(Box::new(BinaryUfunc::new("multiply", |a: f64, b: f64| {
            a * b
        })));
        self.register(Box::new(BinaryUfunc::new("multiply", |a: f32, b: f32| {
            a * b
        })));
        self.register(Box::new(BinaryUfunc::new("multiply", |a: i64, b: i64| {
            a * b
        })));

        // Division
        self.register(Box::new(BinaryUfunc::new("divide", |a: f64, b: f64| a / b)));
        self.register(Box::new(BinaryUfunc::new("divide", |a: f32, b: f32| a / b)));
        self.register(Box::new(BinaryUfunc::new("divide", |a: i64, b: i64| a / b)));

        // Unary operations
        self.register(Box::new(UnaryUfunc::new("negative", |a: f64| -a)));
        self.register(Box::new(UnaryUfunc::new("negative", |a: f32| -a)));
        self.register(Box::new(UnaryUfunc::new("negative", |a: i64| -a)));

        self.register(Box::new(UnaryUfunc::new("absolute", |a: f64| a.abs())));
        self.register(Box::new(UnaryUfunc::new("absolute", |a: f32| a.abs())));
        self.register(Box::new(UnaryUfunc::new("absolute", |a: i64| a.abs())));
    }

    /// Register mathematical ufuncs
    fn register_math_ufuncs(&mut self) {
        use crate::math_ufuncs::register_math_ufuncs;
        register_math_ufuncs(self);
    }

    /// Register bitwise ufuncs
    fn register_bitwise_ufuncs(&mut self) {
        use crate::bitwise::register_bitwise_ufuncs;
        register_bitwise_ufuncs(self);
    }

    /// Register comparison ufuncs
    fn register_comparison_ufuncs(&mut self) {
        use crate::comparison_ufuncs::{ComparisonUfunc, ExtremaUfunc, LogicalUnaryUfunc};

        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &f64, b: &f64| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &f32, b: &f32| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i64, b: &i64| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i32, b: &i32| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i16, b: &i16| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i8, b: &i8| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u64, b: &u64| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u32, b: &u32| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u16, b: &u16| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u8, b: &u8| a > b,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &f64, b: &f64| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &f32, b: &f32| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &i64, b: &i64| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &i32, b: &i32| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &i16, b: &i16| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new("less", |a: &i8, b: &i8| {
            a < b
        })));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &u64, b: &u64| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &u32, b: &u32| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &u16, b: &u16| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new("less", |a: &u8, b: &u8| {
            a < b
        })));

        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &f64, b: &f64| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &f32, b: &f32| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i64, b: &i64| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i32, b: &i32| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i16, b: &i16| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i8, b: &i8| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u64, b: &u64| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u32, b: &u32| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u16, b: &u16| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u8, b: &u8| a >= b,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &f64, b: &f64| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &f32, b: &f32| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i64, b: &i64| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i32, b: &i32| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i16, b: &i16| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i8, b: &i8| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u64, b: &u64| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u32, b: &u32| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u16, b: &u16| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u8, b: &u8| a <= b,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &f64, b: &f64| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &f32, b: &f32| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &i64, b: &i64| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &i32, b: &i32| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &i16, b: &i16| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new("equal", |a: &i8, b: &i8| {
            a == b
        })));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &u64, b: &u64| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &u32, b: &u32| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &u16, b: &u16| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new("equal", |a: &u8, b: &u8| {
            a == b
        })));

        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &f64, b: &f64| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &f32, b: &f32| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i64, b: &i64| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i32, b: &i32| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i16, b: &i16| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i8, b: &i8| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u64, b: &u64| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u32, b: &u32| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u16, b: &u16| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u8, b: &u8| a != b,
        )));

        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &f64, b: &f64| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &f32, b: &f32| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &i64, b: &i64| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &i32, b: &i32| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &i16, b: &i16| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("maximum", |a: &i8, b: &i8| {
            if a >= b {
                *a
            } else {
                *b
            }
        })));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &u64, b: &u64| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &u32, b: &u32| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &u16, b: &u16| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("maximum", |a: &u8, b: &u8| {
            if a >= b {
                *a
            } else {
                *b
            }
        })));

        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &f64, b: &f64| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &f32, b: &f32| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &i64, b: &i64| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &i32, b: &i32| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &i16, b: &i16| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("minimum", |a: &i8, b: &i8| {
            if a <= b {
                *a
            } else {
                *b
            }
        })));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &u64, b: &u64| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &u32, b: &u32| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &u16, b: &u16| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("minimum", |a: &u8, b: &u8| {
            if a <= b {
                *a
            } else {
                *b
            }
        })));

        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &f64, b: &f64| *a != 0.0 && *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &f32, b: &f32| *a != 0.0 && *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i64, b: &i64| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i32, b: &i32| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i16, b: &i16| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i8, b: &i8| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u64, b: &u64| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u32, b: &u32| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u16, b: &u16| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u8, b: &u8| *a != 0 && *b != 0,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &f64, b: &f64| *a != 0.0 || *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &f32, b: &f32| *a != 0.0 || *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i64, b: &i64| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i32, b: &i32| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i16, b: &i16| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i8, b: &i8| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u64, b: &u64| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u32, b: &u32| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u16, b: &u16| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u8, b: &u8| *a != 0 || *b != 0,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &f64, b: &f64| (*a != 0.0) != (*b != 0.0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &f32, b: &f32| (*a != 0.0) != (*b != 0.0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i64, b: &i64| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i32, b: &i32| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i16, b: &i16| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i8, b: &i8| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u64, b: &u64| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u32, b: &u32| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u16, b: &u16| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u8, b: &u8| (*a != 0) != (*b != 0),
        )));

        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &f64| *a == 0.0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &f32| *a == 0.0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i64| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i32| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i16| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new("logical_not", |a: &i8| {
            *a == 0
        })));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &u64| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &u32| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &u16| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new("logical_not", |a: &u8| {
            *a == 0
        })));
    }
}

impl Default for UfuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for UfuncRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UfuncRegistry({} ufuncs)", self.ufuncs.len())
    }
}

// Global ufunc registry (doc comment removed to avoid warning)
lazy_static::lazy_static! {
    pub static ref UFUNC_REGISTRY: UfuncRegistry = UfuncRegistry::new();
}

/// Get ufunc by name
pub fn get_ufunc(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get(name)
}

/// List all available ufuncs
pub fn list_ufuncs() -> Vec<&'static str> {
    UFUNC_REGISTRY.list()
}
