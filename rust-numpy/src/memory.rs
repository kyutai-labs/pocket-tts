// use crate::error::{NumPyError, Result};
// use std::sync::Arc;

/// Memory layout options for arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
}

/// Memory manager for efficient array data handling
pub struct MemoryManager<T> {
    data: Vec<T>,
    ref_count: std::sync::atomic::AtomicUsize,
}

impl<T> MemoryManager<T> {
    /// Create new memory manager from vector
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            ref_count: std::sync::atomic::AtomicUsize::new(1),
        }
    }

    /// Create new memory manager with capacity
    pub fn with_capacity(capacity: usize) -> Self
    where
        T: Clone + Default,
    {
        Self {
            data: vec![T::default(); capacity],
            ref_count: std::sync::atomic::AtomicUsize::new(1),
        }
    }

    /// Get data as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get data as mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get length of data
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get element at index (mutable)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Clone data (deep copy)
    pub fn clone_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }

    pub fn as_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }

    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Resize data
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(new_len, value);
    }

    /// Push element
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Extend with iterator
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter);
    }

    /// Clear data
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Increment reference count
    pub fn inc_ref(&self) {
        self.ref_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Decrement reference count and check if should deallocate
    pub fn dec_ref(&self) -> bool {
        self.ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
            == 1
    }
}

impl<T> Clone for MemoryManager<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            ref_count: std::sync::atomic::AtomicUsize::new(1),
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for MemoryManager<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryManager(len={})", self.data.len())
    }
}

/// Buffer for temporary operations
pub struct TempBuffer<T> {
    data: Vec<T>,
    capacity: usize,
}

impl<T> TempBuffer<T>
where
    T: Clone + Default,
{
    /// Create new temporary buffer
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            capacity: 0,
        }
    }

    /// Ensure buffer has capacity
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.capacity {
            self.data.resize(capacity, T::default());
            self.capacity = capacity;
        }
    }

    /// Get buffer as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.capacity]
    }

    /// Get buffer as mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data[..self.capacity]
    }

    /// Get current capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset buffer
    pub fn reset(&mut self) {
        self.capacity = 0;
        self.data.clear();
    }
}

impl<T> Default for TempBuffer<T>
where
    T: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pool for efficient allocation
pub struct MemoryPool<T> {
    available: Vec<Vec<T>>,
    max_size: usize,
}

impl<T> MemoryPool<T>
where
    T: Clone + Default,
{
    /// Create new memory pool
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Vec::new(),
            max_size,
        }
    }

    /// Get buffer from pool or allocate new one
    pub fn get_buffer(&mut self, size: usize) -> Vec<T> {
        if let Some(mut buf) = self.available.pop() {
            if buf.capacity() >= size {
                buf.resize(size, T::default());
                return buf;
            }
        }
        Vec::with_capacity(size)
    }

    /// Return buffer to pool
    pub fn return_buffer(&mut self, mut buf: Vec<T>) {
        if self.available.len() < self.max_size {
            buf.clear();
            self.available.push(buf);
        }
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        self.available.clear();
    }

    /// Get number of available buffers
    pub fn len(&self) -> usize {
        self.available.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }
}

impl<T> Default for MemoryPool<T>
where
    T: Clone + Default,
{
    fn default() -> Self {
        Self::new(10)
    }
}

/// Alignment utilities for SIMD operations
pub mod alignment {
    // use crate::error::NumPyError; // Unused - using crate::error::NumPyError directly in functions\n    // use std::mem; // Unused - using std::mem directly

    /// Get preferred alignment for type T
    pub fn preferred_alignment<T>() -> usize {
        std::mem::align_of::<T>()
    }

    /// Get SIMD alignment (typically 16, 32, or 64 bytes)
    pub fn simd_alignment() -> usize {
        // For common SIMD instruction sets
        #[cfg(target_arch = "x86_64")]
        {
            32 // AVX2 requires 32-byte alignment
        }
        #[cfg(target_arch = "aarch64")]
        {
            16 // NEON typically uses 16-byte alignment
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            8 // Default to 8-byte alignment
        }
    }

    /// Align pointer to specified alignment
    pub fn align_ptr<T>(ptr: *mut T, alignment: usize) -> *mut T {
        let addr = ptr as usize;
        let aligned = (addr + alignment - 1) & !(alignment - 1);
        aligned as *mut T
    }

    /// Check if pointer is aligned
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Allocate aligned memory
    pub fn alloc_aligned<T>(count: usize, alignment: usize) -> crate::error::Result<*mut T> {
        let layout =
            std::alloc::Layout::from_size_align(count * std::mem::size_of::<T>(), alignment)
                .map_err(|_| {
                    crate::error::NumPyError::memory_error(count * std::mem::size_of::<T>())
                })?;

        unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                return Err(crate::error::NumPyError::memory_error(
                    count * std::mem::size_of::<T>(),
                ));
            }
            Ok(ptr as *mut T)
        }
    }

    /// Deallocate aligned memory
    pub unsafe fn dealloc_aligned<T>(ptr: *mut T, count: usize, alignment: usize) {
        let layout =
            std::alloc::Layout::from_size_align(count * std::mem::size_of::<T>(), alignment)
                .unwrap_unchecked();
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}
