use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use num_traits::NumCast;
use rand::distributions::{Distribution, Standard};
use rand::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Beta, Binomial, ChiSquared, Exp, Gamma, Gumbel, LogNormal, Normal, Poisson};

/// Random number generator interface
pub trait RandomGenerator: Rng + Send + Sync + 'static {
    /// Generate random numbers with given shape
    fn random<T>(&mut self, shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + NumCast + 'static;

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

/// RandomState exposes the NumPy random number generator interface
pub struct RandomState {
    pub(crate) rng: StdRng,
}

impl RandomState {
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

    // --- Distributions implementing logic previously in global functions ---

    pub fn binomial<T>(
        &mut self,
        n: isize,
        p: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        if n < 0 {
            return Err(NumPyError::invalid_value("n must be non-negative"));
        }
        let p_f64 = p.into();
        if !(0.0..=1.0).contains(&p_f64) {
            return Err(NumPyError::value_error("p must be in [0, 1]", "binomial"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        let dist =
            Binomial::new(n as u64, p_f64).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng) as f64;
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn poisson<T>(&mut self, lam: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
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

        let dist = Poisson::new(lam_f64).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn exponential<T>(
        &mut self,
        scale: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
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

        let dist =
            Exp::new(1.0 / scale_f64).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample * scale_f64));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn gamma<T>(
        &mut self,
        shape_param: T,
        scale: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let shape_f64 = shape_param.into();
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

        let dist = Gamma::new(shape_f64, scale_f64)
            .map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape_arr.to_vec()))
    }

    pub fn beta<T>(&mut self, a: T, b: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
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

        let dist = Beta::new(a_f64, b_f64).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn chisquare<T>(&mut self, df: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
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

        let dist = ChiSquared::new(df_f64).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn gumbel<T>(
        &mut self,
        loc: T,
        scale: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
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

        let dist = Gumbel::new(loc_f64, scale_f64)
            .map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn laplace<T>(
        &mut self,
        loc: T,
        scale: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
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

        for _ in 0..total_size {
            let u: f64 = self.rng.gen::<f64>() - 0.5;
            let sample = loc_f64 - scale_f64 * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn lognormal<T>(
        &mut self,
        mean: T,
        sigma: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
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

        let dist = LogNormal::new(mean_f64, sigma_f64)
            .map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn multinomial<T>(
        &mut self,
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

        let pvals_f64: Vec<f64> = pvals.to_vec().into_iter().map(|x| x.into()).collect();

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

        for _ in 0..total_size {
            let mut remaining = n as u64;
            let mut remaining_prob = 1.0;
            let mut results = vec![0u64; pvals_f64.len()];

            for (i, &p) in pvals_f64.iter().enumerate() {
                if i == pvals_f64.len() - 1 {
                    results[i] = remaining;
                } else {
                    if remaining == 0 || p == 0.0 {
                        results[i] = 0;
                        continue;
                    }

                    let adjusted_p = p / remaining_prob;
                    let dist = Binomial::new(remaining, adjusted_p)
                        .map_err(|e| NumPyError::invalid_value(e.to_string()))?;

                    results[i] = dist.sample(&mut self.rng);
                    remaining -= results[i];
                    remaining_prob -= p;
                }
            }

            for &result in &results {
                data.push(T::from(result as f64));
            }
        }

        Ok(Array::from_data(data, output_shape))
    }

    pub fn dirichlet<T>(
        &mut self,
        alpha: &Array<T>,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let alpha_f64: Vec<f64> = alpha.to_vec().into_iter().map(|x| x.into()).collect();

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

        for _ in 0..total_size {
            let mut samples = Vec::with_capacity(k);
            let mut sum = 0.0;

            for &a in &alpha_f64 {
                let dist =
                    Gamma::new(a, 1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
                let sample = dist.sample(&mut self.rng);
                samples.push(sample);
                sum += sample;
            }

            for sample in samples {
                data.push(T::from(sample / sum));
            }
        }

        Ok(Array::from_data(data, output_shape))
    }

    pub fn geometric<T>(&mut self, p: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let p_f64 = p.into();
        if !(0.0..=1.0).contains(&p_f64) {
            return Err(NumPyError::value_error("p must be in [0, 1]", "geometric"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = (1.0 - u).ln() / (1.0 - p_f64).ln();
            let count = sample.ceil() as u64;
            data.push(T::from(count as f64));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn negative_binomial<T>(
        &mut self,
        n: isize,
        p: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        if n < 0 {
            return Err(NumPyError::invalid_value("n must be non-negative"));
        }
        let p_f64 = p.into();
        if !(0.0..=1.0).contains(&p_f64) {
            return Err(NumPyError::value_error("p must be in [0, 1]", "negative_binomial"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let geom_p = 1.0 - p_f64;
            let u: f64 = self.rng.gen();
            let sample = (1.0 - u).ln() / geom_p.ln();
            let geom_count = sample.ceil() as u64;
            let result = geom_count + n as u64 - 1;
            data.push(T::from(result as f64));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn hypergeometric<T>(
        &mut self,
        ngood: isize,
        nbad: isize,
        nsample: isize,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        if ngood < 0 || nbad < 0 || nsample < 0 {
            return Err(NumPyError::invalid_value("parameters must be non-negative"));
        }

        let total = ngood + nbad;
        if nsample > total {
            return Err(NumPyError::invalid_value("nsample must not exceed population size"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        let mut population: Vec<bool> = (0..ngood).map(|_| true).collect();
        population.extend((0..nbad).map(|_| false));

        for _ in 0..total_size {
            let mut sample_pop = population.clone();
            sample_pop.shuffle(&mut self.rng);
            let good_count = sample_pop.iter().take(nsample as usize).filter(|&&x| x).count();
            data.push(T::from(good_count as f64));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn logseries<T>(&mut self, p: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let p_f64 = p.into();
        if !(0.0..1.0).contains(&p_f64) {
            return Err(NumPyError::value_error("p must be in (0, 1)", "logseries"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let q = 1.0 - p_f64;
            let sample = (1.0 - q.powf(1.0 - u)) / (1.0 - q);
            let count = sample.log(1.0 / q).floor() as u64 + 1;
            data.push(T::from(count as f64));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn rayleigh<T>(&mut self, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
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

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = scale_f64 * (-2.0 * u.ln()).sqrt();
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn wald<T>(&mut self, mean: T, scale: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let mean_f64 = mean.into();
        let scale_f64 = scale.into();

        if mean_f64 <= 0.0 {
            return Err(NumPyError::invalid_value("mean must be positive"));
        }
        if scale_f64 <= 0.0 {
            return Err(NumPyError::invalid_value("scale must be positive"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            // Use inverse Gaussian transformation
            let y: f64 = Normal::new(0.0, 1.0)
                .map_err(|e| NumPyError::invalid_value(e.to_string()))?
                .sample(&mut self.rng);
            let y2 = y * y;

            let mu_y = mean_f64 * y;
            let x = mean_f64 + (mu_y * mu_y) / (2.0 * scale_f64) - (mu_y / (2.0 * scale_f64)) * ((4.0 * scale_f64) + mu_y).sqrt();

            let u: f64 = self.rng.gen();
            let sample = if u <= mean_f64 / (mean_f64 + x) {
                x.max(0.0) // Ensure non-negative
            } else {
                let mu2_div_x = (mean_f64 * mean_f64) / x.max(f64::EPSILON);
                if mu2_div_x.is_finite() && mu2_div_x > 0.0 {
                    mu2_div_x
                } else {
                    mean_f64 // Fallback to mean if computation fails
                }
            };
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn weibull<T>(&mut self, a: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let a_f64 = a.into();
        if a_f64 <= 0.0 {
            return Err(NumPyError::invalid_value("shape parameter must be positive"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = (-u.ln()).powf(1.0 / a_f64);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn triangular<T>(
        &mut self,
        left: T,
        mode: T,
        right: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let left_f64 = left.into();
        let mode_f64 = mode.into();
        let right_f64 = right.into();

        if left_f64 >= right_f64 {
            return Err(NumPyError::invalid_value("left must be less than right"));
        }
        if mode_f64 < left_f64 || mode_f64 > right_f64 {
            return Err(NumPyError::invalid_value("mode must be between left and right"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = if u <= (mode_f64 - left_f64) / (right_f64 - left_f64) {
                left_f64 + (u * (right_f64 - left_f64) * (mode_f64 - left_f64)).sqrt()
            } else {
                right_f64 - ((1.0 - u) * (right_f64 - left_f64) * (right_f64 - mode_f64)).sqrt()
            };
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn pareto<T>(&mut self, a: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let a_f64 = a.into();
        if a_f64 <= 0.0 {
            return Err(NumPyError::invalid_value("shape parameter must be positive"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = (1.0 - u).powf(-1.0 / a_f64) - 1.0;
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn zipf<T>(&mut self, a: T, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let a_f64 = a.into();
        if a_f64 <= 1.0 {
            return Err(NumPyError::invalid_value("exponent must be greater than 1"));
        }

        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = (1.0 - u).powf(-1.0 / (a_f64 - 1.0)).floor() as u64;
            data.push(T::from(sample as f64));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_cauchy<T>(&mut self, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            let u: f64 = self.rng.gen();
            let sample = (std::f64::consts::PI * (u - 0.5)).tan();
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_exponential<T>(&mut self, size: Option<&[usize]>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let shape = size.unwrap_or(&[1]);
        let total_size = shape.iter().product();
        let mut data = Vec::with_capacity(total_size);

        let dist = Exp::new(1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_gamma<T>(
        &mut self,
        shape: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Into<f64> + From<f64> + Default + 'static,
    {
        let shape_f64 = shape.into();
        if shape_f64 <= 0.0 {
            return Err(NumPyError::invalid_value("shape must be positive"));
        }

        let shape_arr = size.unwrap_or(&[1]);
        let total_size = shape_arr.iter().product();
        let mut data = Vec::with_capacity(total_size);

        let dist = Gamma::new(shape_f64, 1.0)
            .map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape_arr.to_vec()))
    }

    pub fn shuffle<T: Clone + Default + 'static>(
        &mut self,
        arr: &mut Array<T>,
    ) -> Result<(), NumPyError> {
        let data = arr.data.as_slice_mut();
        data.shuffle(&mut self.rng);
        Ok(())
    }

    // --- Legacy methods moved here ---

    pub fn rand<T>(&mut self, d0: usize, d1: Option<usize>) -> Result<Array<T>, NumPyError>
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

        for _ in 0..total_size {
            let sample = self.rng.gen::<f64>();
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape))
    }

    pub fn randn<T>(&mut self, d0: usize, d1: Option<usize>) -> Result<Array<T>, NumPyError>
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

        let dist = Normal::new(0.0, 1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;

        for _ in 0..total_size {
            let sample = dist.sample(&mut self.rng);
            data.push(T::from(sample));
        }

        Ok(Array::from_data(data, shape))
    }

    pub fn permutation(&mut self, n: usize) -> Result<Array<usize>, NumPyError> {
        let mut data: Vec<usize> = (0..n).collect();
        data.shuffle(&mut self.rng);
        Ok(Array::from_vec(data))
    }

    pub fn choice<T: Clone + Default + 'static>(
        &mut self,
        a: &Array<T>,
        size: usize,
    ) -> Result<Array<T>, NumPyError> {
        let n = a.size();
        if n == 0 {
            return Err(NumPyError::invalid_value("Cannot choose from empty array"));
        }

        let mut indices = Vec::with_capacity(size);
        for _ in 0..size {
            indices.push(self.rng.gen_range(0..n));
        }

        let mut data = Vec::with_capacity(size);
        for idx in indices {
            data.push(a.get(idx).unwrap().clone());
        }

        Ok(Array::from_vec(data))
    }

    pub fn sample<T: Clone + Default + 'static>(
        &mut self,
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
        indices.shuffle(&mut self.rng);

        let mut data = Vec::with_capacity(k);
        for i in 0..k {
            data.push(a.get(indices[i]).unwrap().clone());
        }

        Ok(Array::from_vec(data))
    }
}

impl rand::RngCore for RandomState {
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

impl RandomGenerator for RandomState {
    fn random<T>(&mut self, shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + NumCast + 'static,
    {
        let size = shape.iter().product();

        match dtype {
            Dtype::Float32 { .. } => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: f32 = self.rng.sample(Standard);
                    data.push(NumCast::from(val).unwrap_or_default());
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            Dtype::Float64 { .. } => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: f64 = self.rng.sample(Standard);
                    data.push(NumCast::from(val).unwrap_or_default());
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            Dtype::Int32 { .. } => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: i32 = self.rng.gen();
                    data.push(NumCast::from(val).unwrap_or_default());
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            Dtype::Int64 { .. } => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: i64 = self.rng.gen();
                    data.push(NumCast::from(val).unwrap_or_default());
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            Dtype::Bool => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: bool = self.rng.gen();
                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
                        let t_val: T = unsafe { std::mem::transmute_copy(&val) };
                        data.push(t_val);
                    } else {
                        let num = if val { 1 } else { 0 };
                        data.push(NumCast::from(num).unwrap_or_default());
                    }
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            _ => Err(NumPyError::not_implemented(
                "Random generation for this dtype not yet implemented",
            )),
        }
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

    fn normal<T>(&mut self, mean: T, std: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.rng.gen_range(mean.clone()..std.clone()));
        }

        Ok(Array::from_data(data, shape.to_vec()))
    }
}
