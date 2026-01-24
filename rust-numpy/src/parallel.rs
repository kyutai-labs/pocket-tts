use crate::array::Array;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Trait for parallel iteration over array elements
///
/// This trait provides parallel iterator capabilities for Arrays when the "rayon" feature is enabled.
/// Currently, optimized parallel iteration is supported for C-contiguous arrays.
#[cfg(feature = "rayon")]
pub trait ParArrayIter<'a, T>
where
    T: Sync + Send + 'a,
{
    /// Create a parallel iterator over the array elements
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numpy::Array;
    /// use numpy::parallel::ParArrayIter;
    /// use rayon::prelude::*;
    ///
    /// let arr = Array::from_vec(vec![1, 2, 3, 4]);
    /// let sum: i32 = arr.par_iter().sum();
    /// assert_eq!(sum, 10);
    /// ```
    fn par_iter(&'a self) -> rayon::slice::Iter<'a, T>;

    /// Create a parallel mutable iterator over the array elements
    fn par_iter_mut(&'a mut self) -> rayon::slice::IterMut<'a, T>;
}

#[cfg(feature = "rayon")]
impl<'a, T> ParArrayIter<'a, T> for Array<T>
where
    T: Sync + Send + 'a + Clone + Default + 'static,
{
    fn par_iter(&'a self) -> rayon::slice::Iter<'a, T> {
        // For now, we only support efficient slice-based parallel iteration
        // for contiguous arrays. In the future, we may add specific parallel
        // iterators for strided arrays.

        // We use the underlying data slice directly if valid
        if self.is_c_contiguous() {
            let start = self.offset;
            let end = start + self.size();
            // SAFETY: MemoryManager ensures data validity, and is_c_contiguous+size check ensures bounds
            let slice = &self.data.as_slice()[start..end];
            slice.par_iter()
        } else {
            // Fallback for non-contiguous arrays could be implemented here,
            // but for infrastructure v1, we'll panic or return an empty iter logic?
            // Returning just the available slice might be misleading if strides matter.
            // To be safe: Panic if not contiguous (mimicking early infrastructure strictness)
            // or better: Documentation says "optimized... for C-contiguous".
            // Let's panic with a clear message for now to encourage explicit layout handling
            panic!("Parallel iteration currently only supported for C-contiguous arrays. Use .as_standard_layout() first.");
        }
    }

    fn par_iter_mut(&'a mut self) -> rayon::slice::IterMut<'a, T> {
        if self.is_c_contiguous() {
            let start = self.offset;
            let end = start + self.size();
            // Need to handle safe mutable access. MemoryManager provides as_slice_mut.
            // But Array wraps it in Arc. If we have &mut Array, we can get mut access if we have unique ownership
            // or if MemoryManager allows internal mutability (UnsafeCell).
            // array.rs line 543 `index_mut` uses `Arc::make_mut`.

            // We need to ensure we have unique access to the data
            let data = Arc::make_mut(&mut self.data);
            let slice = &mut data.as_slice_mut()[start..end];
            slice.par_iter_mut()
        } else {
            panic!("Parallel mutable iteration currently only supported for C-contiguous arrays. Use .as_standard_layout() first.");
        }
    }
}

/// Utility module to expose Rayon prelude items
#[cfg(feature = "rayon")]
pub mod prelude {
    pub use super::ParArrayIter;
    pub use rayon::prelude::*;
}
