use numpy::dtype::{Casting, Dtype};
use numpy::math_ufuncs::MathBinaryUfunc;
use numpy::ufunc::UfuncRegistry;

#[test]
fn test_exact_match_resolution() {
    let mut registry = UfuncRegistry::new();

    // Register f64 implementation
    registry.register(Box::new(MathBinaryUfunc::new(
        "add_exact",
        |x: &f64, y: &f64| x + y,
    )));

    let f64_dtype = Dtype::from_type::<f64>();

    let result = registry.resolve_ufunc(
        "add_exact",
        &[f64_dtype.clone(), f64_dtype.clone()],
        Casting::Safe,
    );

    assert!(result.is_some());
    let (ufunc, target_dtypes) = result.unwrap();
    assert_eq!(ufunc.name(), "add_exact");
    assert_eq!(target_dtypes, vec![f64_dtype.clone(), f64_dtype.clone()]);
}

#[test]
fn test_safe_casting_resolution() {
    let mut registry = UfuncRegistry::new();

    // Register f64 implementation ONLY
    registry.register(Box::new(MathBinaryUfunc::new(
        "add_safe",
        |x: &f64, y: &f64| x + y,
    )));

    let i32_dtype = Dtype::from_type::<i32>();
    let f64_dtype = Dtype::from_type::<f64>();

    // Try to resolve with i32 inputs (should cast to f64 safely)
    let result = registry.resolve_ufunc(
        "add_safe",
        &[i32_dtype.clone(), i32_dtype.clone()],
        Casting::Safe,
    );

    assert!(
        result.is_some(),
        "Should find add<f64> for i32 input via safe casting"
    );
    let (_, target_dtypes) = result.unwrap();
    assert_eq!(target_dtypes, vec![f64_dtype.clone(), f64_dtype.clone()]);
}

#[test]
fn test_unsafe_casting_failure() {
    let mut registry = UfuncRegistry::new();

    // Register i32 implementation ONLY
    registry.register(Box::new(MathBinaryUfunc::new(
        "add_unsafe",
        |x: &i32, y: &i32| x + y,
    )));

    let f64_dtype = Dtype::from_type::<f64>();

    // Try to resolve with f64 inputs (f64 -> i32 is unsafe)
    let result = registry.resolve_ufunc(
        "add_unsafe",
        &[f64_dtype.clone(), f64_dtype.clone()],
        Casting::Safe,
    );

    assert!(result.is_none(), "Should NOT resolve f64->i32 as Safe");
}

#[test]
fn test_resolution_priority_first_match() {
    let mut registry = UfuncRegistry::new();

    // Register f64 first
    registry.register(Box::new(MathBinaryUfunc::new(
        "add_prio",
        |x: &f64, y: &f64| x + y,
    )));
    // Register i32 second
    registry.register(Box::new(MathBinaryUfunc::new(
        "add_prio",
        |x: &i32, y: &i32| x + y,
    )));

    let i32_dtype = Dtype::from_type::<i32>();

    // Input i32 matches BOTH:
    // 1. Safe cast to f64 (First registered check? Or does it iterate?)
    // 2. Exact match i32
    // Current implementation: Iterates in order.
    // If f64 is first, and i32->f64 is Safe, it will pick f64!
    // This highlights we might want "Exact match" priority ideally, but for now strict order.

    let result = registry.resolve_ufunc(
        "add_prio",
        &[i32_dtype.clone(), i32_dtype.clone()],
        Casting::Safe,
    );

    assert!(result.is_some());
    let (_, target_dtypes) = result.unwrap();

    // Since f64 was registered first and i32->f64 is SAFE, it picks f64.
    // This asserts the CURRENT behavior (First Safe Match).
    assert_eq!(target_dtypes[0], Dtype::from_type::<f64>());
}

#[test]
fn test_resolution_no_match() {
    let mut registry = UfuncRegistry::new();
    registry.register(Box::new(MathBinaryUfunc::new(
        "add_none",
        |x: &f64, y: &f64| x + y,
    )));

    let complex_dtype = Dtype::from_type::<num_complex::Complex64>();

    // Complex -> Float is not safe?
    // Actually Complex -> Float is usually "SameKind" (downcast kind?) or Unsafe (imaginary lost).
    // Let's assume it's NOT Safe.

    let result = registry.resolve_ufunc(
        "add_none",
        &[complex_dtype.clone(), complex_dtype.clone()],
        Casting::Safe,
    );

    assert!(result.is_none());
}
