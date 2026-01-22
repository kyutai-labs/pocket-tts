// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Random number generation utils

pub mod random_state;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use num_traits::NumCast;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::uniform::SampleUniform;
pub use random_state::{RandomGenerator, RandomState};
use std::cell::RefCell;

thread_local! {
    static DEFAULT_RNG: RefCell<RandomState> = RefCell::new(RandomState::new());
}

// --- Global Functions (Proxies) ---

pub fn random<T: Clone + Default + NumCast + 'static>(
    shape: &[usize],
    dtype: Dtype,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().random(shape, dtype))
}

pub fn randint<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().randint(low, high, shape))
}

pub fn uniform<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().uniform(low, high, shape))
}

pub fn normal<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    mean: T,
    std: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().normal(mean, std, shape))
}

pub fn standard_normal<T: Clone + From<f64> + PartialOrd + SampleUniform + Default + 'static>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    normal(T::from(0.0), T::from(1.0), shape)
}

pub fn permutation(n: usize) -> Result<Array<usize>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().permutation(n))
}

pub fn choice<T: Clone + Default + 'static>(
    a: &Array<T>,
    size: usize,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().choice(a, size))
}

pub fn sample<T: Clone + Default + 'static>(
    a: &Array<T>,
    k: usize,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().sample(a, k))
}

pub struct SeedSequence {
    seed: u64,
    position: u64,
}

impl SeedSequence {
    pub fn new(seed: u64) -> Self {
        Self { seed, position: 0 }
    }

    pub fn next(&mut self) -> u64 {
        use rand::RngCore;
        let mut rng = StdRng::seed_from_u64(self.seed + self.position);
        let next_seed = rng.next_u64();
        self.position += 1;
        next_seed
    }

    pub fn spawn(&mut self) -> RandomState {
        RandomState::seed_from_u64(self.next())
    }
}

pub fn seed(seed: u64) {
    DEFAULT_RNG.with(|rng| {
        *rng.borrow_mut() = RandomState::seed_from_u64(seed);
    });
}

pub fn get_state() -> u64 {
    use rand::RngCore;
    DEFAULT_RNG.with(|rng| rng.borrow_mut().rng.next_u64())
}

pub fn set_state(state: u64) {
    DEFAULT_RNG.with(|rng| {
        *rng.borrow_mut() = RandomState::seed_from_u64(state);
    });
}

pub fn binomial<T>(n: isize, p: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().binomial(n, p, size))
}

pub fn poisson<T>(lam: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().poisson(lam, size))
}

pub fn exponential<T>(scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().exponential(scale, size))
}

pub fn gamma<T>(shape: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().gamma(shape, scale, size))
}

pub fn beta<T>(a: T, b: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().beta(a, b, size))
}

pub fn chisquare<T>(df: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().chisquare(df, size))
}

pub fn gumbel<T>(loc: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().gumbel(loc, scale, size))
}

pub fn laplace<T>(loc: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().laplace(loc, scale, size))
}

pub fn lognormal<T>(mean: T, sigma: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().lognormal(mean, sigma, size))
}

pub fn multinomial<T>(
    n: isize,
    pvals: &Array<T>,
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().multinomial(n, pvals, size))
}

pub fn dirichlet<T>(alpha: &Array<T>, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().dirichlet(alpha, size))
}

pub fn rand<T>(d0: usize, d1: Option<usize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().rand(d0, d1))
}

pub fn randn<T>(d0: usize, d1: Option<usize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    DEFAULT_RNG.with(|rng| rng.borrow_mut().randn(d0, d1))
}

pub fn random_integers<T>(
    low: T,
    high: Option<T>,
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + PartialOrd + SampleUniform + Default + NumCast + 'static,
{
    let high_val = high.unwrap_or(NumCast::from(100).unwrap_or_default());
    let shape = size.unwrap_or(&[1]);
    randint(low, high_val, shape)
}

pub fn random_sample<T>(size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone
        + Into<f64>
        + From<f64>
        + PartialOrd
        + rand_distr::uniform::SampleUniform
        + Default
        + 'static,
{
    let shape = size.unwrap_or(&[1]);
    uniform(T::from(0.0), T::from(1.0), shape)
}

pub fn ranf<T>(size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone
        + Into<f64>
        + From<f64>
        + rand_distr::uniform::SampleUniform
        + Default
        + PartialOrd
        + 'static,
{
    random_sample(size)
}

pub fn legacy_sample<T>(size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone
        + Into<f64>
        + From<f64>
        + rand_distr::uniform::SampleUniform
        + Default
        + PartialOrd
        + 'static,
{
    random_sample(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_generation() {
        let arr = random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), vec![2, 3]);
    }

    #[test]
    fn test_randint() {
        let arr = randint(0, 10, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), vec![2, 2]);
    }

    #[test]
    fn test_permutation() {
        let perm = permutation(5).unwrap();
        assert_eq!(perm.shape(), vec![5]);
        let mut sorted = perm.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_state_struct() {
        let mut rs = RandomState::new();
        let arr = rs.normal(0.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), vec![2, 2]);

        let mut rs_seeded = RandomState::seed_from_u64(42);
        let arr1 = rs_seeded.uniform::<f64>(0.0, 1.0, &[5]).unwrap();

        let mut rs_seeded_again = RandomState::seed_from_u64(42);
        let arr2 = rs_seeded_again.uniform::<f64>(0.0, 1.0, &[5]).unwrap();

        // Check reproducibility
        for i in 0..5 {
            let val1 = *arr1.get_linear(i).unwrap();
            let val2 = *arr2.get_linear(i).unwrap();
            assert!((val1 - val2).abs() < 1e-10);
        }
    }
}
