use crate::array::Array;
use crate::broadcasting::compute_broadcasted_strides;
use crate::error::Result;
use crate::strides::compute_broadcast_shape;
use crate::ufunc::ArrayView;

/// Iterator over array elements
pub struct ArrayIter<'a, T> {
    array: &'a Array<T>,
    indices: Vec<usize>,
    current_offset: isize,
    remaining: usize,
}

impl<'a, T> ArrayIter<'a, T> {
    pub fn new(array: &'a Array<T>) -> Self {
        let size = array.size();
        let ndim = array.shape().len();
        Self {
            array,
            indices: vec![0; ndim],
            current_offset: 0,
            remaining: size,
        }
    }
}

impl<'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // Get element at current physical offset
        let physical_idx = (self.array.offset as isize + self.current_offset) as usize;
        // Use direct data access via MemoryManager (which handles UnsafeCell)
        let item = self.array.data.get(physical_idx);

        // Advance counters
        self.remaining -= 1;
        if self.remaining > 0 {
            let shape = self.array.shape();
            let strides = self.array.strides();
            let ndim = shape.len();

            for i in (0..ndim).rev() {
                self.indices[i] += 1;
                self.current_offset += strides[i];

                if self.indices[i] < shape[i] {
                    // No carry, we are done
                    break;
                }

                // Carry over: reset this dimension and continue loop to next dimension
                self.indices[i] = 0;
                // Backtrack offset: we added strides[i] * shape[i] total.
                // We want to return to 0-index offset for this dimension.
                // So we remove stride * shape.
                self.current_offset -= strides[i] * (shape[i] as isize);
            }
        }

        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for ArrayIter<'a, T> {}

/// Multi-operand N-dimensional iterator for broadcasting support
pub struct NDIter<'a> {
    #[allow(dead_code)]
    operands: Vec<&'a dyn ArrayView>,
    shape: Vec<usize>,
    operand_strides: Vec<Vec<isize>>,
    indices: Vec<usize>,
    current_offsets: Vec<isize>,
    remaining: usize,
    started: bool,
}

impl<'a> NDIter<'a> {
    pub fn new(operands: Vec<&'a dyn ArrayView>) -> Result<Self> {
        if operands.is_empty() {
            return Err(crate::error::NumPyError::invalid_operation(
                "NDIter requires at least one operand",
            ));
        }

        // Compute common broadcast shape
        let mut broadcast_shape = operands[0].shape().to_vec();
        for op in operands.iter().skip(1) {
            broadcast_shape = compute_broadcast_shape(&broadcast_shape, op.shape());
        }

        let size: usize = broadcast_shape.iter().product();
        let ndim = broadcast_shape.len();

        // Compute broadcasted strides for each operand
        let mut operand_strides = Vec::with_capacity(operands.len());
        let mut initial_offsets = Vec::with_capacity(operands.len());
        for op in &operands {
            let strides = compute_broadcasted_strides(op.shape(), op.strides(), &broadcast_shape);
            operand_strides.push(strides);
            initial_offsets.push(op.offset() as isize);
        }

        Ok(Self {
            operands,
            shape: broadcast_shape,
            operand_strides,
            indices: vec![0; ndim],
            current_offsets: initial_offsets,
            remaining: size,
            started: false,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.remaining
    }
}

impl<'a> Iterator for NDIter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        if !self.started {
            self.started = true;
        } else {
            // Advance indices and offsets
            let ndim = self.shape.len();
            for i in (0..ndim).rev() {
                self.indices[i] += 1;

                // Update offsets for each operand
                for (op_idx, op_strides) in self.operand_strides.iter().enumerate() {
                    self.current_offsets[op_idx] += op_strides[i];
                }

                if self.indices[i] < self.shape[i] {
                    break;
                }

                // Carry over
                self.indices[i] = 0;
                for (op_idx, op_strides) in self.operand_strides.iter().enumerate() {
                    self.current_offsets[op_idx] -= op_strides[i] * (self.shape[i] as isize);
                }
            }
        }

        self.remaining -= 1;

        // Return current offsets as usize
        Some(self.current_offsets.iter().map(|&o| o as usize).collect())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for NDIter<'a> {}
