// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Random number generation
//!
//! This module provides a complete implementation of NumPy's random functionality,
//! including various distributions, generators, and sampling methods.

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use rand::distributions::Distribution;
use rand::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Beta, Binomial, ChiSquared, Exp, Gamma, Gumbel, LogNormal, Normal, Poisson};
use std::cell::RefCell;

/// Random number generator interface
pub trait RandomGenerator: Rng + Send + Sync + 'static {
    /// Generate random numbers with given shape
    fn random<T>(&mut self, shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static;

    /// Generate integers in range
    fn randint<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static;

    /// Generate uniform random numbers
    fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static;

    /// Generate normal random numbers
    fn normal<T>(&mut self, mean: T, std: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static;
}

/// Default random number generator using StdRng
pub struct DefaultGenerator {
    rng: StdRng,
}

impl DefaultGenerator {
    /// Create new generator with random seed
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Create generator with specific seed
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl rand::RngCore for DefaultGenerator {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.rng.try_fill_bytes(dest)
    }
}

impl RandomGenerator for DefaultGenerator {
    fn random<T>(&mut self, shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        match dtype {
            Dtype::Float32 { .. } => {
                for _ in 0..size {
                    let val: f32 = self.rng.gen();
                    data.push(unsafe { std::mem::transmute_copy(&val) });
                }
            }
            Dtype::Float64 { .. } => {
                for _ in 0..size {
                    let val: f64 = self.rng.gen();
                    data.push(unsafe { std::mem::transmute_copy(&val) });
                }
            }
            Dtype::Int32 { .. } => {
                for _ in 0..size {
                    let val: i32 = self.rng.gen();
                    data.push(unsafe { std::mem::transmute_copy(&val) });
                }
            }
            Dtype::Int64 { .. } => {
                for _ in 0..size {
                    let val: i64 = self.rng.gen();
                    data.push(unsafe { std::mem::transmute_copy(&val) });
                }
            }
            Dtype::Bool => {
                for _ in 0..size {
                    let val: bool = self.rng.gen();
                    data.push(unsafe { std::mem::transmute_copy(&val) });
                }
            }
            _ => {
                return Err(NumPyError::not_implemented(
                    "Random generation for this dtype not yet implemented",
                ))
            }
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    fn randint<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.rng.gen_range(low.clone()..high.clone()));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.rng.gen_range(low.clone()..high.clone()));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    fn normal<T>(&mut self, mean: T, std_dev: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.rng.gen_range(mean.clone()..std_dev.clone()));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }
}

// Thread-local random generator instance
thread_local! {
    static DEFAULT_RNG: RefCell<DefaultGenerator> = RefCell::new(DefaultGenerator::new());
}

/// Generate random array with default generator
pub fn random<T: Clone + Default + 'static>(
    shape: &[usize],
    dtype: Dtype,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().random(shape, dtype))
}

/// Generate random integers
pub fn randint<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().randint(low, high, shape))
}

/// Generate uniform random numbers
pub fn uniform<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().uniform(low, high, shape))
}

/// Generate normal random numbers
pub fn normal<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    mean: T,
    std: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().normal(mean, std, shape))
}

/// Generate standard normal random numbers (mean=0, std=1)
pub fn standard_normal<T: Clone + From<f64> + PartialOrd + SampleUniform + Default + 'static>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    normal(T::from(0.0), T::from(1.0), shape)
}

/// Random permutation of integers
pub fn permutation(n: usize) -> Result<Array<usize>, NumPyError> {
    let mut data: Vec<usize> = (0..n).collect();
    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        data.shuffle(&mut rng.rng);
    });
    Ok(Array::from_vec(data))
}

/// Random choice from array
pub fn choice<T: Clone + Default + 'static>(
    a: &Array<T>,
    size: usize,
) -> Result<Array<T>, NumPyError> {
    let n = a.size();
    if n == 0 {
        return Err(NumPyError::invalid_value("Cannot choose from empty array"));
    }

    let mut indices = Vec::with_capacity(size);
    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for _ in 0..size {
            indices.push(rng.rng.gen_range(0..n));
        }
    });

    let mut data = Vec::with_capacity(size);
    for idx in indices {
        data.push(a.get(idx).unwrap().clone());
    }

    Ok(Array::from_vec(data))
}

/// Random sample without replacement
pub fn sample<T: Clone + Default + 'static>(
    a: &Array<T>,
    k: usize,
) -> Result<Array<T>, NumPyError> {
    let n = a.size();
    if k > n {
        return Err(NumPyError::value_error(
            "Sample size cannot be larger than array size",
            "sample",
        ));
    }

    let mut indices: Vec<usize> = (0..n).collect();
    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        indices.shuffle(&mut rng.rng);
    });

    let mut data = Vec::with_capacity(k);
    for i in 0..k {
        data.push(a.get(indices[i]).unwrap().clone());
    }

    Ok(Array::from_vec(data))
}

/// Random seed control
pub struct SeedSequence {
    seed: u64,
    position: u64,
}

impl SeedSequence {
    /// Create new seed sequence
    pub fn new(seed: u64) -> Self {
        Self { seed, position: 0 }
    }

    /// Generate next seed in sequence
    pub fn next(&mut self) -> u64 {
        use rand::RngCore;
        let mut rng = StdRng::seed_from_u64(self.seed + self.position);
        let next_seed = rng.next_u64();
        self.position += 1;
        next_seed
    }

    /// Spawn new generator
    pub fn spawn(&mut self) -> DefaultGenerator {
        DefaultGenerator::seed_from_u64(self.next())
    }
}

/// Global seed management
pub fn seed(seed: u64) {
    DEFAULT_RNG.with(|rng| {
        *rng.borrow_mut() = DefaultGenerator::seed_from_u64(seed);
    });
}

/// Get random generator state
pub fn get_state() -> u64 {
    use rand::RngCore;
    DEFAULT_RNG.with(|rng| rng.borrow_mut().rng.next_u64())
}

/// Set random generator state
pub fn set_state(state: u64) {
    DEFAULT_RNG.with(|rng| {
        *rng.borrow_mut() = DefaultGenerator::seed_from_u64(state);
    });
}

/// Binomial distribution
///
/// Draw samples from a binomial distribution.
///
/// # Arguments
///
/// * `n` - Number of trials (must be >= 0)
/// * `p` - Probability of success (must be in [0, 1])
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn binomial<T>(n: isize, p: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    if n < 0 {
        return Err(NumPyError::invalid_value("n must be non-negative"));
    }
    let p_f64 = p.into();
    if p_f64 < 0.0 || p_f64 > 1.0 {
        return Err(NumPyError::value_error("p must be in [0, 1]", "binomial"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist = Binomial::new(n as u64, p_f64)
            .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng) as f64;
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Poisson distribution
///
/// Draw samples from a Poisson distribution.
///
/// # Arguments
///
/// * `lam` - Expected number of events (lambda, must be >= 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn poisson<T>(lam: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let lam_f64 = lam.into();
    if lam_f64 < 0.0 {
        return Err(NumPyError::invalid_value("lambda must be non-negative"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist = Poisson::new(lam_f64).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng) as f64;
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Exponential distribution
///
/// Draw samples from an exponential distribution.
///
/// # Arguments
///
/// * `scale` - Scale parameter (1/lambda, must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn exponential<T>(scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let scale_f64 = scale.into();
    if scale_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("scale must be positive"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist =
            Exp::new(1.0 / scale_f64).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample * scale_f64)); // Wait, Exp(lambda) has mean 1/lambda. numpy exponential(scale) has mean scale. So lambda = 1/scale.
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Gamma distribution
///
/// Draw samples from a gamma distribution.
///
/// # Arguments
///
/// * `shape` - Shape parameter (alpha, must be > 0)
/// * `scale` - Scale parameter (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn gamma<T>(shape: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let shape_f64 = shape.into();
    let scale_f64 = scale.into();

    if shape_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("shape must be positive"));
    }
    if scale_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("scale must be positive"));
    }

    let shape_arr = size.unwrap_or(&[1]);
    let total_size = shape_arr.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist = Gamma::new(shape_f64, scale_f64)
            .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape_arr.to_vec()))
}

/// Beta distribution
///
/// Draw samples from a beta distribution.
///
/// # Arguments
///
/// * `a` - Alpha parameter (must be > 0)
/// * `b` - Beta parameter (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn beta<T>(a: T, b: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let a_f64 = a.into();
    let b_f64 = b.into();

    if a_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("a must be positive"));
    }
    if b_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("b must be positive"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist =
            Beta::new(a_f64, b_f64).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Chi-square distribution
///
/// Draw samples from a chi-square distribution.
///
/// # Arguments
///
/// * `df` - Degrees of freedom (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn chisquare<T>(df: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let df_f64 = df.into();
    if df_f64 <= 0.0 {
        return Err(NumPyError::value_error(
            "degrees of freedom must be positive",
            "chisquare",
        ));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist =
            ChiSquared::new(df_f64).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Gumbel distribution
///
/// Draw samples from a Gumbel distribution.
///
/// # Arguments
///
/// * `loc` - Location parameter
/// * `scale` - Scale parameter (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn gumbel<T>(loc: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let loc_f64 = loc.into();
    let scale_f64 = scale.into();

    if scale_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("scale must be positive"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist = Gumbel::new(loc_f64, scale_f64)
            .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Laplace distribution
///
/// Draw samples from a Laplace distribution.
///
/// # Arguments
///
/// * `loc` - Location parameter
/// * `scale` - Scale parameter (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn laplace<T>(loc: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let loc_f64 = loc.into();
    let scale_f64 = scale.into();

    if scale_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("scale must be positive"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();

        for _ in 0..total_size {
            let u: f64 = rng.rng.gen::<f64>() - 0.5;
            let sample = loc_f64 - scale_f64 * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Log-normal distribution
///
/// Draw samples from a log-normal distribution.
///
/// # Arguments
///
/// * `mean` - Mean of the underlying normal distribution
/// * `sigma` - Standard deviation of the underlying normal distribution (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn lognormal<T>(mean: T, sigma: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let mean_f64 = mean.into();
    let sigma_f64 = sigma.into();

    if sigma_f64 <= 0.0 {
        return Err(NumPyError::invalid_value("sigma must be positive"));
    }

    let shape = size.unwrap_or(&[1]);
    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist = LogNormal::new(mean_f64, sigma_f64)
            .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape.to_vec()))
}

/// Multinomial distribution
///
/// Draw samples from a multinomial distribution.
///
/// # Arguments
///
/// * `n` - Number of trials (must be >= 0)
/// * `pvals` - Probabilities for each outcome (must sum to 1)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn multinomial<T>(
    n: isize,
    pvals: &Array<T>,
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    if n < 0 {
        return Err(NumPyError::invalid_value("n must be non-negative"));
    }

    // Convert pvals to f64 vector
    let pvals_f64: Vec<f64> = pvals.to_vec().into_iter().map(|x| x.into()).collect();

    // Check probabilities are valid
    if pvals_f64.iter().any(|&p| p < 0.0) {
        return Err(NumPyError::value_error(
            "probabilities must be non-negative",
            "multinomial",
        ));
    }

    let sum: f64 = pvals_f64.iter().sum();
    if (sum - 1.0).abs() > 1e-10 {
        return Err(NumPyError::invalid_value("probabilities must sum to 1"));
    }

    let output_shape = if let Some(size) = size {
        let mut shape = size.to_vec();
        shape.push(pvals_f64.len());
        shape
    } else {
        vec![pvals_f64.len()]
    };

    let total_size = output_shape.iter().product::<usize>() / pvals_f64.len();
    let mut data = Vec::with_capacity(output_shape.iter().product());

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();

        for _ in 0..total_size {
            let mut remaining = n as u64;
            let mut remaining_prob = 1.0;
            let mut results = vec![0u64; pvals_f64.len()];

            for (i, &p) in pvals_f64.iter().enumerate() {
                if i == pvals_f64.len() - 1 {
                    // Last outcome gets remaining
                    results[i] = remaining;
                } else {
                    if remaining == 0 || p == 0.0 {
                        results[i] = 0;
                        continue;
                    }

                    let adjusted_p = p / remaining_prob;
                    let dist = Binomial::new(remaining, adjusted_p)
                        .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

                    results[i] = dist.sample(&mut rng.rng);
                    remaining -= results[i];
                    remaining_prob -= p;
                }
            }

            for &result in &results {
                data.push(T::from(result as f64));
            }
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, output_shape))
}

/// Dirichlet distribution
///
/// Draw samples from a Dirichlet distribution.
///
/// # Arguments
///
/// * `alpha` - Concentration parameters (must be > 0)
/// * `size` - Optional shape for the output array
///
/// # Returns
///
/// Array of drawn samples
pub fn dirichlet<T>(alpha: &Array<T>, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let alpha_f64: Vec<f64> = alpha.to_vec().into_iter().map(|x| x.into()).collect();

    // Check concentration parameters are valid
    if alpha_f64.iter().any(|&a| a <= 0.0) {
        return Err(NumPyError::invalid_value(
            "alpha parameters must be positive",
        ));
    }

    let k = alpha_f64.len();
    let output_shape = if let Some(size) = size {
        let mut shape = size.to_vec();
        shape.push(k);
        shape
    } else {
        vec![k]
    };

    let total_size = output_shape.iter().product::<usize>() / k;
    let mut data = Vec::with_capacity(output_shape.iter().product());

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();

        for _ in 0..total_size {
            let mut samples = Vec::with_capacity(k);
            let mut sum = 0.0;

            // Generate gamma samples for each alpha parameter
            for &a in &alpha_f64 {
                let dist =
                    Gamma::new(a, 1.0).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;
                let sample = dist.sample(&mut rng.rng);
                samples.push(sample);
                sum += sample;
            }

            // Normalize to sum to 1
            for sample in samples {
                data.push(T::from(sample / sum));
            }
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, output_shape))
}

/// Legacy functions for NumPy compatibility

/// Generate random floats in the half-open interval [0.0, 1.0)
///
/// Legacy function equivalent to numpy.random.rand
pub fn rand<T>(d0: usize, d1: Option<usize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + PartialOrd + Default + 'static,
{
    let shape = if let Some(d1_val) = d1 {
        vec![d0, d1_val]
    } else {
        vec![d0]
    };

    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for _ in 0..total_size {
            let sample = rng.rng.gen::<f64>();
            data.push(T::from(sample));
        }
    });

    Ok(Array::from_data(data, shape))
}

/// Generate random floats from the standard normal distribution
///
/// Legacy function equivalent to numpy.random.randn
pub fn randn<T>(d0: usize, d1: Option<usize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + Default + 'static,
{
    let shape = if let Some(d1_val) = d1 {
        vec![d0, d1_val]
    } else {
        vec![d0]
    };

    let total_size = shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let dist = Normal::new(0.0, 1.0).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut rng.rng);
            data.push(T::from(sample));
        }
        Ok::<(), NumPyError>(())
    })?;

    Ok(Array::from_data(data, shape))
}

/// Generate random integers
///
/// Legacy function for compatibility
pub fn random_integers<T>(
    low: T,
    high: Option<T>,
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + PartialOrd + SampleUniform + Default + num_traits::NumCast + 'static,
{
    let high_val = high.unwrap_or(num_traits::cast::NumCast::from(100).unwrap()); // Default high = 100
    let shape = size.unwrap_or(&[1]);
    randint(low, high_val, shape)
}

/// Generate random floats in [0.0, 1.0)
///
/// Legacy function equivalent to random_sample
pub fn random_sample<T>(size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + PartialOrd + SampleUniform + Default + 'static,
{
    let shape = size.unwrap_or(&[1]);
    uniform(T::from(0.0), T::from(1.0), shape)
}

/// Generate random floats in [0.0, 1.0)
///
/// Legacy function equivalent to numpy.random.ranf
pub fn ranf<T>(size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + PartialOrd + SampleUniform + Default + 'static,
{
    random_sample(size)
}

/// Generate random samples
///
/// Legacy function for compatibility
pub fn legacy_sample<T>(size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Into<f64> + From<f64> + PartialOrd + SampleUniform + Default + 'static,
{
    random_sample(size)
}

/// Random distribution generators
pub struct RandomDist;

impl RandomDist {
    /// Exponential distribution
    pub fn exponential<T>(scale: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + From<f64> + Into<f64> + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        DEFAULT_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let scale_f64: f64 = scale.clone().into();
            let dist =
                Exp::new(1.0 / scale_f64).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;
            for _ in 0..size {
                let sample = dist.sample(&mut rng.rng);
                data.push(T::from(sample * scale_f64));
            }
            Ok::<(), NumPyError>(())
        })?;

        Ok(Array::from_data(data, shape.to_vec()))
    }

    /// Poisson distribution
    pub fn poisson<T>(lambda: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + From<f64> + Into<f64> + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let lambda_f64 = lambda.into();

        DEFAULT_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist =
                Poisson::new(lambda_f64).map_err(|e| NumPyError::invalid_value(&e.to_string()))?;
            for _ in 0..size {
                let sample = dist.sample(&mut rng.rng) as f64;
                data.push(T::from(sample));
            }
            Ok::<(), NumPyError>(())
        })?;

        Ok(Array::from_data(data, shape.to_vec()))
    }

    /// Binomial distribution
    pub fn binomial<T>(n: T, p: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + From<f64> + Into<f64> + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let n_f64 = n.into();
        let p_f64 = p.into();

        DEFAULT_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Binomial::new(n_f64 as u64, p_f64)
                .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;
            for _ in 0..size {
                let sample = dist.sample(&mut rng.rng) as f64;
                data.push(T::from(sample));
            }
            Ok::<(), NumPyError>(())
        })?;

        Ok(Array::from_data(data, shape.to_vec()))
    }

    /// Gamma distribution
    pub fn gamma<T>(shape_param: T, scale: T, array_shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + From<f64> + Into<f64> + Default + 'static,
    {
        let size = array_shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let shape_f64 = shape_param.into();
        let scale_f64 = scale.into();

        DEFAULT_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Gamma::new(shape_f64, scale_f64)
                .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;
            for _ in 0..size {
                let sample = dist.sample(&mut rng.rng);
                data.push(T::from(sample));
            }
            Ok::<(), NumPyError>(())
        })?; // Add error check
             // Note: The original code logic was missing ? or handling for with error.
             // Assuming with returns Result, we need to propagate it.

        Ok(Array::from_data(data, array_shape.to_vec()))
    }

    /// Beta distribution
    pub fn beta<T>(alpha: T, beta_param: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + From<f64> + Into<f64> + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let alpha_f64 = alpha.into();
        let beta_f64 = beta_param.into();

        DEFAULT_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = rand_distr::Beta::new(alpha_f64, beta_f64)
                .map_err(|e| NumPyError::invalid_value(&e.to_string()))?;
            for _ in 0..size {
                let sample = dist.sample(&mut rng.rng);
                data.push(T::from(sample));
            }
            Ok::<(), NumPyError>(())
        })?;

        Ok(Array::from_data(data, shape.to_vec()))
    }
}

/// Random number generation utilities
pub struct RandomUtils;

impl RandomUtils {
    /// Create random array with specific distribution
    pub fn generate<T>(
        &mut self,
        shape: &[usize],
        distribution: &str,
        _params: &[f64],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + From<f64> + Into<f64> + PartialOrd + SampleUniform + Default + 'static,
    {
        match distribution {
            "uniform" => uniform(T::from(0.0), T::from(1.0), shape),
            "normal" => normal(T::from(0.0), T::from(1.0), shape),
            "exponential" => RandomDist::exponential(T::from(1.0), shape),
            _ => Err(NumPyError::invalid_value("Unknown distribution")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_generation() {
        let arr: Array<f64> = random(&[2, 3], Dtype::Float64 { byteorder: None }).unwrap();
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
}
