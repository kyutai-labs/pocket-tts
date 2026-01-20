use crate::array::Array;
use faer::{MatRef, MatMut};

/// Trait to bridge rust-numpy Array to faer MatRef/MatMut
pub trait ToFaer<'a, T> {
    fn as_faer(&'a self) -> MatRef<'a, T>;
    // fn as_faer_mut(&'a mut self) -> MatMut<'a, T>; // Deferred until mutable access pattern is clarified
}

impl<'a, T> ToFaer<'a, T> for Array<T> {
    fn as_faer(&'a self) -> MatRef<'a, T> {
        assert_eq!(self.ndim(), 2, "Array must be 2D for faer conversion");
        let nrows = self.shape()[0];
        let ncols = self.shape()[1];
        let row_stride = self.strides()[0];
        let col_stride = self.strides()[1];

        unsafe {
            MatRef::from_raw_parts(
                self.as_slice().as_ptr(),
                nrows,
                ncols,
                row_stride,
                col_stride,
            )
        }
    }
}
