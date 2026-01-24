use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use std::marker::PhantomData;

impl<T: 'static> ArrayView for Array<T> {
    fn dtype(&self) -> &Dtype {
        self.dtype()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn offset(&self) -> usize {
        self.offset
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
        self.data.as_ref().as_slice().as_ptr() as *const u8
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T: 'static> ArrayViewMut for Array<T> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_ref().as_slice().as_ptr() as *mut u8
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub trait ArrayView {
    fn dtype(&self) -> &Dtype;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn size(&self) -> usize;
    fn offset(&self) -> usize;
    fn ndim(&self) -> usize;
    fn is_contiguous(&self) -> bool;
    fn as_ptr(&self) -> *const u8;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait ArrayViewMut: ArrayView {
    fn as_mut_ptr(&mut self) -> *mut u8;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

pub trait Ufunc: Send + Sync {
    fn name(&self) -> &'static str;
    fn nin(&self) -> usize;
    fn nout(&self) -> usize;
    fn supported_dtypes(&self) -> &[DtypeKind];
    fn type_signature(&self) -> String;
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()>;
    fn get_strided_kernel(&self, _dtypes: &[Dtype]) -> Option<Box<dyn std::any::Any>> {
        None
    }
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool {
        dtypes
            .iter()
            .all(|dt| self.supported_dtypes().contains(&dt.kind()))
    }
    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool;
    fn input_dtypes(&self) -> Vec<Dtype>;
}

pub struct BinaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> BinaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
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
    T: Clone + Default + 'static + Send + Sync,
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
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        let input0 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let input1 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        if input0.is_c_contiguous()
            && input1.is_c_contiguous()
            && output.is_c_contiguous()
            && input0.shape() == input1.shape()
            && input0.shape() == output.shape()
        {
            let d0 = input0.data.as_slice();
            let d1 = input1.data.as_slice();
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, output.size())
            };
            for i in 0..output.size() {
                if mask
                    .as_ref()
                    .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
                {
                    out_slice[i] = (self.operation)(
                        d0[input0.offset + i].clone(),
                        d1[input1.offset + i].clone(),
                    );
                }
            }
            return Ok(());
        }

        let broadcasted = crate::broadcasting::broadcast_arrays(&[input0, input1])?;
        for i in 0..output.size() {
            if mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let (Some(a), Some(b)) = (broadcasted[0].get(i), broadcasted[1].get(i)) {
                    output.set(i, (self.operation)(a.clone(), b.clone()))?;
                }
            }
        }
        Ok(())
    }
}

pub struct UnaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> UnaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T) -> T + Send + Sync,
{
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
    T: Clone + Default + 'static + Send + Sync,
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
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        let input = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        if input.is_c_contiguous() && output.is_c_contiguous() && input.shape() == output.shape() {
            let in_slice = input.data.as_slice();
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, output.size())
            };
            for i in 0..output.size() {
                if mask
                    .as_ref()
                    .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
                {
                    out_slice[i] = (self.operation)(in_slice[input.offset + i].clone());
                }
            }
            return Ok(());
        }
        for i in 0..input.size() {
            if mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let Some(a) = input.get(i) {
                    output.set(i, (self.operation)(a.clone()))?;
                }
            }
        }
        Ok(())
    }
}

pub struct UfuncRegistry {
    ufuncs: std::collections::HashMap<String, Vec<Box<dyn Ufunc>>>,
}

impl UfuncRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            ufuncs: std::collections::HashMap::new(),
        };
        registry.register_basic_ufuncs();
        registry.register_comparison_ufuncs();
        registry.register_math_ufuncs();
        registry.register_bitwise_ufuncs();
        registry
    }
    pub fn register(&mut self, ufunc: Box<dyn Ufunc>) {
        let name = ufunc.name().to_string();
        self.ufuncs.entry(name).or_default().push(ufunc);
    }
    pub fn get(&self, name: &str) -> Option<&dyn Ufunc> {
        self.ufuncs
            .get(name)
            .and_then(|v| v.first())
            .map(|u| u.as_ref())
    }
    pub fn get_by_dtypes(&self, name: &str, input_types: &[&'static str]) -> Option<&dyn Ufunc> {
        self.ufuncs.get(name).and_then(|v| {
            v.iter()
                .find(|u| u.matches_concrete_types(input_types))
                .map(|u| u.as_ref())
        })
    }
    pub fn resolve_ufunc(
        &self,
        name: &str,
        input_dtypes: &[Dtype],
        casting: crate::dtype::Casting,
    ) -> Option<(&dyn Ufunc, Vec<Dtype>)> {
        if let Some(candidates) = self.ufuncs.get(name) {
            for ufunc in candidates {
                let target_dtypes = ufunc.input_dtypes();
                if target_dtypes.len() != input_dtypes.len() {
                    continue;
                }
                if input_dtypes
                    .iter()
                    .zip(target_dtypes.iter())
                    .all(|(in_dt, target_dt)| in_dt.can_cast(target_dt, casting))
                {
                    return Some((ufunc.as_ref(), target_dtypes));
                }
            }
        }
        None
    }
    pub fn list(&self) -> Vec<&str> {
        self.ufuncs.keys().map(|s| s.as_str()).collect()
    }
    fn register_basic_ufuncs(&mut self) {
        let types = vec!["add", "subtract", "multiply", "divide"];
        for t in types {
            self.register(Box::new(BinaryUfunc::new(
                t,
                match t {
                    "add" => |a: f64, b: f64| a + b,
                    "subtract" => |a, b| a - b,
                    "multiply" => |a, b| a * b,
                    _ => |a, b| a / b,
                },
            )));
            self.register(Box::new(BinaryUfunc::new(
                t,
                match t {
                    "add" => |a: f32, b: f32| a + b,
                    "subtract" => |a, b| a - b,
                    "multiply" => |a, b| a * b,
                    _ => |a, b| a / b,
                },
            )));
            self.register(Box::new(BinaryUfunc::new(
                t,
                match t {
                    "add" => |a: i64, b: i64| a + b,
                    "subtract" => |a, b| a - b,
                    "multiply" => |a, b| a * b,
                    _ => |a, b| a / b,
                },
            )));
            self.register(Box::new(BinaryUfunc::new(
                t,
                match t {
                    "add" => |a: i32, b: i32| a + b,
                    "subtract" => |a, b| a - b,
                    "multiply" => |a, b| a * b,
                    _ => |a, b| a / b,
                },
            )));
        }
        self.register(Box::new(UnaryUfunc::new("negative", |a: f64| -a)));
        self.register(Box::new(UnaryUfunc::new("absolute", |a: f64| a.abs())));
    }
    fn register_math_ufuncs(&mut self) {
        crate::math_ufuncs::register_math_ufuncs(self);
    }
    fn register_bitwise_ufuncs(&mut self) {
        crate::bitwise::register_bitwise_ufuncs(self);
    }
    fn register_comparison_ufuncs(&mut self) {
        use crate::comparison_ufuncs::ComparisonUfunc;
        let ops = vec![
            "greater",
            "less",
            "equal",
            "not_equal",
            "greater_equal",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
        ];
        for op in ops {
            self.register(Box::new(ComparisonUfunc::new(
                op,
                match op {
                    "greater" => |a: &f64, b: &f64| a > b,
                    "less" => |a: &f64, b: &f64| a < b,
                    "equal" => |a: &f64, b: &f64| a == b,
                    "not_equal" => |a: &f64, b: &f64| a != b,
                    "greater_equal" => |a: &f64, b: &f64| a >= b,
                    "less_equal" => |a: &f64, b: &f64| a <= b,
                    "logical_and" => |a: &f64, b: &f64| *a != 0.0 && *b != 0.0,
                    "logical_or" => |a: &f64, b: &f64| *a != 0.0 || *b != 0.0,
                    _ => |a: &f64, b: &f64| (*a != 0.0) != (*b != 0.0),
                },
            )));
            self.register(Box::new(ComparisonUfunc::new(
                op,
                match op {
                    "greater" => |a: &i32, b: &i32| a > b,
                    "less" => |a: &i32, b: &i32| a < b,
                    "equal" => |a: &i32, b: &i32| a == b,
                    "not_equal" => |a: &i32, b: &i32| a != b,
                    "greater_equal" => |a: &i32, b: &i32| a >= b,
                    "less_equal" => |a: &i32, b: &i32| a <= b,
                    "logical_and" => |a: &i32, b: &i32| *a != 0 && *b != 0,
                    "logical_or" => |a: &i32, b: &i32| *a != 0 || *b != 0,
                    _ => |a: &i32, b: &i32| (*a != 0) != (*b != 0),
                },
            )));
        }
        use crate::comparison_ufuncs::LogicalUnaryUfunc;
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &f64| *a == 0.0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i32| *a == 0,
        )));
    }
}

lazy_static::lazy_static! { pub static ref UFUNC_REGISTRY: UfuncRegistry = UfuncRegistry::new(); }
pub fn get_ufunc(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get(name)
}
pub fn get_ufunc_typed<T: 'static>(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get_by_dtypes(name, &[std::any::type_name::<T>()])
}
pub fn get_ufunc_typed_binary<T: 'static>(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get_by_dtypes(
        name,
        &[std::any::type_name::<T>(), std::any::type_name::<T>()],
    )
}
pub fn list_ufuncs() -> Vec<&'static str> {
    UFUNC_REGISTRY.list()
}

impl Default for UfuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl std::fmt::Debug for UfuncRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UfuncRegistry({} names)", self.ufuncs.len())
    }
}
