// Copyright 2024 The NumPyRS Authors.

use super::{Polynomial, PolynomialBase};
use crate::error::NumPyError;
use ndarray::Array1;
use num_traits::{Float, Num, One, Zero};

/// Generalized Laguerre polynomials
#[derive(Debug, Clone)]
pub struct Laguerre<T> {
    coeffs: Array1<T>,
    domain: [T; 2],
    window: [T; 2],
}

impl<T> Laguerre<T>
where
    T: Float
        + Num
        + std::fmt::Debug
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    pub fn new(coeffs: &Array1<T>) -> Result<Self, NumPyError> {
        if coeffs.len() == 0 {
            return Err(NumPyError::invalid_value(
                "Laguerre coefficients cannot be empty",
            ));
        }

        let coeffs = coeffs.to_owned();
        let domain = [T::zero(), T::one()];
        let window = [T::zero(), T::one()];

        Ok(Self {
            coeffs,
            domain,
            window,
        })
    }

    pub fn coeffs(&self) -> &Array1<T> {
        &self.coeffs
    }

    pub fn domain(&self) -> [T; 2] {
        self.domain
    }

    pub fn window(&self) -> [T; 2] {
        self.window
    }

    pub fn from_polynomial(poly: &Polynomial<T>) -> Result<Self, NumPyError> {
        let lag_coeffs = polynomial_to_laguerre(poly.coeffs())?;
        Ok(Self {
            coeffs: lag_coeffs,
            domain: poly.domain(),
            window: poly.window(),
        })
    }
}

impl<T> PolynomialBase<T> for Laguerre<T>
where
    T: Float
        + Num
        + std::fmt::Debug
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    fn coeffs(&self) -> &Array1<T> {
        &self.coeffs
    }

    fn eval(&self, x: &Array1<T>) -> Result<Array1<T>, NumPyError> {
        let mut result = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            let xi_scaled = (xi - self.window[0]) / (self.window[1] - self.window[0]);
            let value = laguerre_eval_recursive(&self.coeffs, xi_scaled);
            result[i] = value;
        }

        Ok(result)
    }

    fn deriv(&self, m: usize) -> Result<Self, NumPyError> {
        if m == 0 {
            return Ok(Self {
                coeffs: self.coeffs.clone(),
                domain: self.domain,
                window: self.window,
            });
        }

        let deriv_coeffs = laguerre_derivative_coeffs(&self.coeffs, m)?;
        Ok(Self {
            coeffs: deriv_coeffs,
            domain: self.domain,
            window: self.window,
        })
    }

    fn integ(&self, m: usize, k: Option<T>) -> Result<Self, NumPyError> {
        let integ_coeffs = laguerre_integral_coeffs(&self.coeffs, m, k)?;
        Ok(Self {
            coeffs: integ_coeffs,
            domain: self.domain,
            window: self.window,
        })
    }

    fn domain(&self) -> [T; 2] {
        self.domain
    }

    fn window(&self) -> [T; 2] {
        self.window
    }

    fn to_polynomial(&self) -> Result<Polynomial<T>, NumPyError> {
        let poly_coeffs = laguerre_to_polynomial(&self.coeffs)?;
        Polynomial::new(&poly_coeffs)
    }
}

fn laguerre_eval_recursive<T>(coeffs: &Array1<T>, x: T) -> T
where
    T: Float
        + Num
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::ops::DivAssign
        + 'static,
{
    if coeffs.len() == 0 {
        return T::zero();
    }

    let mut result = T::zero();
    let mut n_factorial = T::one();

    for (n, &coeff) in coeffs.iter().enumerate() {
        let mut lag_n = T::one();
        if n > 0 {
            let mut binomial_sum = T::zero();
            for k in 0..=n {
                let binomial = binomial_coefficient::<T>(n, k);
                let sign = num_traits::cast::NumCast::from((-1.0_f64).powi(k as i32)).unwrap();
                let factorial_part =
                    num_traits::cast::NumCast::from(factorial(n) / factorial(k)).unwrap();
                binomial_sum += binomial * sign * factorial_part * x.powi(k as i32);
            }
            lag_n = binomial_sum / T::from(factorial(n)).unwrap();
        }
        result += coeff * lag_n;
        n_factorial *= T::from(n + 1).unwrap();
    }

    result
}

fn polynomial_to_laguerre<T>(poly_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
where
    T: Float
        + Num
        + std::fmt::Debug
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    let n = poly_coeffs.len();
    let mut lag_coeffs = Array1::zeros(n);

    for k in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            if j <= k {
                let coeff = poly_coeffs[j];
                let binomial = binomial_coefficient(k, j);
                let factorial = T::from(factorial(k) / factorial(j)).unwrap();
                sum += coeff
                    * binomial
                    * factorial
                    * T::from((-1.0_f64).powi((k - j) as i32)).unwrap();
            }
        }
        lag_coeffs[k] = sum;
    }

    Ok(lag_coeffs)
}

fn laguerre_to_polynomial<T>(lag_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
where
    T: Float
        + Num
        + std::fmt::Debug
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    let n = lag_coeffs.len();
    let mut poly_coeffs = Array1::zeros(n);

    for k in 0..n {
        let coeff = lag_coeffs[k];

        for j in 0..=k {
            let binomial = binomial_coefficient(k, j);
            let sign = T::from((-1.0_f64).powi((k - j) as i32)).unwrap();
            let factorial = T::from(factorial(k) / factorial(j)).unwrap();
            poly_coeffs[j] += coeff * binomial * sign / factorial;
        }
    }

    Ok(poly_coeffs)
}

fn laguerre_derivative_coeffs<T>(coeffs: &Array1<T>, m: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static + std::ops::AddAssign + std::ops::MulAssign,
{
    if m == 0 {
        return Ok(coeffs.clone());
    }

    let n = coeffs.len();
    if m >= n {
        return Ok(Array1::from_vec(vec![T::zero()]));
    }

    let mut deriv_coeffs = Array1::zeros(n - m);

    for k in m..n {
        let mut sum = T::zero();
        for j in m..=k {
            let factor = laguerre_derivative_factor(k, j, m);
            sum += coeffs[j] * factor;
        }
        deriv_coeffs[k - m] = sum;
    }

    Ok(deriv_coeffs)
}

fn laguerre_derivative_factor<T>(k: usize, j: usize, m: usize) -> T
where
    T: Float + Num + std::ops::MulAssign,
{
    if m > k {
        return T::zero();
    }

    let mut factor = T::one();
    for r in 0..m {
        factor *= T::from(k - r).unwrap();
    }

    factor * T::from((-1.0_f64).powi(m as i32)).unwrap()
}

fn laguerre_integral_coeffs<T>(
    coeffs: &Array1<T>,
    m: usize,
    k: Option<T>,
) -> Result<Array1<T>, NumPyError>
where
    T: Float
        + Num
        + std::fmt::Debug
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    let n = coeffs.len();
    let mut integ_coeffs = Array1::zeros(n + m);

    // Zeros are already initialized
    if let Some(kval) = k {
        for i in 0..m {
            integ_coeffs[i] = kval;
        }
    }

    for k in 0..n {
        integ_coeffs[k + m] = coeffs[k];
    }

    Ok(integ_coeffs)
}

fn binomial_coefficient<T>(n: usize, k: usize) -> T
where
    T: Float + Num + 'static,
{
    if k > n {
        return T::zero();
    }

    if k == 0 || k == n {
        return T::one();
    }

    let mut result = T::one();
    for i in 1..=k {
        result = result * T::from(n - k + i).unwrap() / T::from(i).unwrap();
    }

    result
}

fn factorial(n: usize) -> usize {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
