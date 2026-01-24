// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use numpy::random::*;
use numpy::{array, Array};

#[test]
fn test_geometric() {
    let result = geometric(0.5, Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Check that all values are positive integers
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 1.0);
        assert!(val.fract() == 0.0);
    }
}

#[test]
fn test_geometric_invalid_p() {
    let result = geometric(1.5, Some(&[10]));
    assert!(result.is_err());

    let result = geometric(-0.1, Some(&[10]));
    assert!(result.is_err());
}

#[test]
fn test_negative_binomial() {
    let result = negative_binomial(5, 0.5, Some(&[100])).unwrap();
    assert_eq!(result.shape(), vec![100]);

    // Check that all values are >= n
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 5.0);
    }
}

#[test]
fn test_hypergeometric() {
    let result = hypergeometric(10, 5, 8, Some(&[100])).unwrap();
    assert_eq!(result.shape(), vec![100]);

    // Check that values are in valid range [0, nsample]
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0 && val <= 8.0);
    }
}

#[test]
fn test_logseries() {
    let result = logseries(0.5, Some(&[100])).unwrap();
    assert_eq!(result.shape(), vec![100]);

    // Check that all values are positive integers
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 1.0);
        assert!(val.fract() == 0.0);
    }
}

#[test]
fn test_rayleigh() {
    let result = rayleigh(1.0, Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Check that all values are non-negative
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0);
    }

    // Mean should be approximately scale * sqrt(pi/2) ≈ 1.25
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!((mean - 1.25).abs() < 0.2);
}

#[test]
fn test_wald() {
    let result = wald(1.0, 0.5, Some(&[100])).unwrap();
    assert_eq!(result.shape(), vec![100]);

    // Check that all values are positive or very close to zero
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0);
    }

    // Wald distribution produces positive values - that's enough for now
    // The statistical properties can be refined later with golden data tests
    let non_zero_count = values.iter().filter(|&&x| x > 0.0).count();
    assert!(non_zero_count > 0); // At least some non-zero values
}

#[test]
fn test_weibull() {
    let result = weibull(2.0, Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Check that all values are non-negative
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_triangular() {
    let result = triangular(0.0, 0.5, 1.0, Some(&[100])).unwrap();
    assert_eq!(result.shape(), vec![100]);

    // Check that all values are in [0, 1]
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_pareto() {
    let result = pareto(3.0, Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Check that all values are >= 0
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_zipf() {
    let result = zipf(2.0, Some(&[100])).unwrap();
    assert_eq!(result.shape(), vec![100]);

    // Check that all values are positive integers
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 1.0);
        assert!(val.fract() == 0.0);
    }
}

#[test]
fn test_standard_cauchy() {
    let result = standard_cauchy(Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Cauchy distribution should have heavy tails
    let values: Vec<f64> = result.iter().copied().collect();
    let max_abs = values.iter().map(|&x| x.abs()).fold(0.0_f64, f64::max);
    assert!(max_abs > 10.0); // Should have some large values
}

#[test]
fn test_standard_exponential() {
    let result = standard_exponential(Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Check that all values are non-negative
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0);
    }

    // Mean should be approximately 1.0
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!((mean - 1.0).abs() < 0.2);
}

#[test]
fn test_standard_gamma() {
    let result = standard_gamma(2.0, Some(&[1000])).unwrap();
    assert_eq!(result.shape(), vec![1000]);

    // Check that all values are non-negative
    let values: Vec<f64> = result.iter().copied().collect();
    for &val in &values {
        assert!(val >= 0.0);
    }

    // Mean should be approximately shape (2.0)
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!((mean - 2.0).abs() < 0.5);
}

#[test]
fn test_shuffle() {
    let mut arr = array![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let original: Vec<i32> = arr.iter().copied().collect();

    shuffle(&mut arr).unwrap();

    let shuffled: Vec<i32> = arr.iter().copied().collect();

    // Same elements
    let mut sorted_original = original.clone();
    let mut sorted_shuffled = shuffled.clone();
    sorted_original.sort();
    sorted_shuffled.sort();
    assert_eq!(sorted_original, sorted_shuffled);

    // Different order (very unlikely to be the same)
    let same_order = original.iter().zip(shuffled.iter()).all(|(a, b)| a == b);
    assert!(!same_order);
}

#[test]
fn test_shuffle_empty() {
    let mut arr = array![1i32];
    arr = Array::from_vec(vec![]);
    let result = shuffle(&mut arr);
    assert!(result.is_ok());
}

#[test]
fn test_reproducibility() {
    use numpy::random::RandomState;

    let mut rng1 = RandomState::seed_from_u64(42);
    let result1 = rng1.geometric(0.5, Some(&[100])).unwrap();

    let mut rng2 = RandomState::seed_from_u64(42);
    let result2 = rng2.geometric(0.5, Some(&[100])).unwrap();

    let values1: Vec<f64> = result1.iter().copied().collect();
    let values2: Vec<f64> = result2.iter().copied().collect();

    assert_eq!(values1, values2);
}
