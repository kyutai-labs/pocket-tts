use crate::array::Array;

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

        // Get element at current offset
        // We trust our offset calculation is correct and within bounds of the storage
        // relative to array.offset
        let item = self.array.get(self.current_offset as usize);

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
