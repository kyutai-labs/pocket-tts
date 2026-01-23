/// Mathematical constants matching NumPy
pub mod math {
    use std::f64;

    /// Euler's number
    pub const E: f64 = f64::consts::E;

    /// Pi
    pub const PI: f64 = f64::consts::PI;

    /// 2 * Pi
    pub const TAU: f64 = f64::consts::TAU;

    /// Infinity
    pub const INF: f64 = f64::INFINITY;

    /// Negative infinity
    pub const NEG_INF: f64 = f64::NEG_INFINITY;

    /// Not a Number
    pub const NAN: f64 = f64::NAN;
}

/// Type-specific constants
pub mod dtype {
    /// Maximum value for int8
    pub const INT8_MAX: i8 = i8::MAX;

    /// Minimum value for int8
    pub const INT8_MIN: i8 = i8::MIN;

    /// Maximum value for int16
    pub const INT16_MAX: i16 = i16::MAX;

    /// Minimum value for int16
    pub const INT16_MIN: i16 = i16::MIN;

    /// Maximum value for int32
    pub const INT32_MAX: i32 = i32::MAX;

    /// Minimum value for int32
    pub const INT32_MIN: i32 = i32::MIN;

    /// Maximum value for int64
    pub const INT64_MAX: i64 = i64::MAX;

    /// Minimum value for int64
    pub const INT64_MIN: i64 = i64::MIN;

    /// Maximum value for uint8
    pub const UINT8_MAX: u8 = u8::MAX;

    /// Maximum value for uint16
    pub const UINT16_MAX: u16 = u16::MAX;

    /// Maximum value for uint32
    pub const UINT32_MAX: u32 = u32::MAX;

    /// Maximum value for uint64
    pub const UINT64_MAX: u64 = u64::MAX;
}

/// Floating point special values
pub mod float {
    use std::f64;

    /// Smallest positive normal f64
    pub const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;

    /// Machine epsilon for f64
    pub const EPSILON: f64 = f64::EPSILON;

    /// Largest finite f64
    pub const MAX: f64 = f64::MAX;

    /// Smallest finite f64
    pub const MIN: f64 = f64::MIN;

    /// Difference between 1.0 and next representable f64
    pub const EPSILON_F32: f32 = f32::EPSILON;
}

/// Array creation limits
pub mod array {
    /// Maximum array dimensions
    pub const MAX_DIMS: usize = 32;

    /// Maximum array elements (practical limit)
    pub const MAX_ELEMENTS: usize = usize::MAX / 8;

    /// Maximum string length for string arrays
    pub const MAX_STRING_LENGTH: usize = 65535;

    /// Default alignment for arrays
    pub const DEFAULT_ALIGNMENT: usize = 8;
}

/// Rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even
    HalfToEven,
    /// Round to nearest, ties away from zero
    HalfAwayFromZero,
    /// Round towards positive infinity
    Up,
    /// Round towards negative infinity
    Down,
    /// Round towards zero
    TowardsZero,
}

/// Comparison kinds for ufuncs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonKind {
    /// Equal
    Equal,
    /// Not equal
    NotEqual,
    /// Less than
    Less,
    /// Less than or equal
    LessEqual,
    /// Greater than
    Greater,
    /// Greater than or equal
    GreaterEqual,
}

/// Reduction modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMode {
    /// Sum of elements
    Sum,
    /// Product of elements
    Product,
    /// Minimum of elements
    Min,
    /// Maximum of elements
    Max,
    /// Mean of elements
    Mean,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Any non-zero element
    Any,
    /// All non-zero elements
    All,
}

/// Order constants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// C-style (row-major) order
    C,
    /// Fortran-style (column-major) order
    F,
    /// Keep existing order
    K,
    /// Closest order in memory
    A,
}

/// Sort kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    /// Quicksort
    QuickSort,
    /// Heapsort
    HeapSort,
    /// Stable sort
    Stable,
}

/// Search modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Insert left
    Left,
    /// Insert right
    Right,
}

/// Clip modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipMode {
    /// Clip to min/max values
    Clip,
    /// Wrap around
    Wrap,
    /// Raise error on overflow
    Raise,
}

/// Utility functions for special values
pub mod utils {
    use super::math::{INF, NAN, NEG_INF};

    /// Check if value is finite
    pub fn is_finite(value: f64) -> bool {
        value.is_finite()
    }

    /// Check if value is infinite
    pub fn is_infinite(value: f64) -> bool {
        value.is_infinite()
    }

    /// Check if value is NaN
    pub fn is_nan(value: f64) -> bool {
        value.is_nan()
    }

    /// Check if value is not NaN
    pub fn is_not_nan(value: f64) -> bool {
        !value.is_nan()
    }

    /// Positive infinity
    pub fn positive_inf() -> f64 {
        INF
    }

    /// Negative infinity
    pub fn negative_inf() -> f64 {
        NEG_INF
    }

    /// Not a number
    pub fn nan() -> f64 {
        NAN
    }
}

pub use array::{DEFAULT_ALIGNMENT, MAX_DIMS, MAX_ELEMENTS};
pub use dtype::{
    INT16_MAX, INT16_MIN, INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN, INT8_MAX, INT8_MIN,
    UINT16_MAX, UINT32_MAX, UINT64_MAX, UINT8_MAX,
};
pub use float::{EPSILON, EPSILON_F32, MAX, MIN, MIN_POSITIVE};
/// Re-export commonly used constants
pub use math::{E, INF, NAN, NEG_INF, PI, TAU};

/// Indexing constants
pub mod index {
    /// A placeholder for adding a new axis in array indexing
    /// Equivalent to np.newaxis in NumPy
    ///
    /// # Examples
    /// ```ignore
    /// let a = array![1, 2, 3];
    /// // Use expand_dims function or pass newaxis to indexing operations
    /// let expanded = expand_dims(&a, 0)?;
    /// ```
    pub const NEWAXIS: isize = -1;
}

pub use index::NEWAXIS;
