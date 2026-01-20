use crate::array::Array;
use crate::error::NumPyError;

// Stubs for eigen
pub fn eig<T>(_a: &Array<T>) -> Result<Array<T>, NumPyError> {
    Err(NumPyError::not_implemented("eig not implemented"))
}
