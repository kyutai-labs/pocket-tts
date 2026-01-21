// Copyright 2024 The NumPyRS Authors.

use super::{Polynomial, PolynomialBase};
use crate::error::NumPyError;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, Num};

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
        + std::ops::SubAssign
        + std::ops::MulAssign
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
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    fn coeffs(&self) -> &Array1<T> {
        &self.coeffs
    }

    fn eval(&self, x: &Array1<T>) -> Result<Array1<T>, NumPyError> {
        let mut result = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            let xi_window = self.window[0]
                + (xi - self.domain[0]) * (self.window[1] - self.window[0])
                    / (self.domain[1] - self.domain[0]);
            let value = laguerre_eval_recursive(&self.coeffs, xi_window);
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
        + 'static
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
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
                let binomial: T = binomial_coefficient::<T>(n, k);
                let sign = T::from((-1.0_f64).powi(k as i32)).unwrap();
                let factorial_part = T::from(factorial(n) / factorial(k)).unwrap();
                let term: T = binomial * sign * factorial_part * x.powi(k as i32);
                binomial_sum += term;
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
        + std::ops::SubAssign
        + std::ops::MulAssign
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
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    let n = lag_coeffs.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let mut poly_coeffs = Array1::zeros(n);
    if n == 1 {
        poly_coeffs[0] = lag_coeffs[0];
        return Ok(poly_coeffs);
    }

    let mut c_prev = Array1::zeros(n);
    let mut c_curr = Array1::zeros(n);

    // L0 = 1
    c_prev[0] = T::one();
    poly_coeffs[0] += lag_coeffs[0] * c_prev[0];

    // L1 = 1 - x
    c_curr[0] = T::one();
    c_curr[1] = -T::one();
    poly_coeffs[0] += lag_coeffs[1] * c_curr[0];
    poly_coeffs[1] += lag_coeffs[1] * c_curr[1];

    for k in 1..(n - 1) {
        let mut c_next = Array1::zeros(n);
        let fk = T::from(k).unwrap();
        let fk1 = T::from(k + 1).unwrap();

        // L_{k+1} = ((2k+1-x) * L_k - k * L_{k-1}) / (k+1)
        for i in 0..n {
            c_next[i] += (T::from(2.0).unwrap() * fk + T::one()) * c_curr[i] / fk1;
        }
        for i in 1..n {
            c_next[i] -= c_curr[i - 1] / fk1; // -x * L_k / (k+1)
        }
        for i in 0..n {
            c_next[i] -= fk * c_prev[i] / fk1;
        }

        for i in 0..n {
            poly_coeffs[i] += lag_coeffs[k + 1] * c_next[i];
        }
        c_prev = c_curr.clone();
        c_curr = c_next;
    }

    Ok(poly_coeffs)
}

fn laguerre_derivative_coeffs<T>(coeffs: &Array1<T>, m: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float
        + Num
        + std::fmt::Debug
        + 'static
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
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
            let factor: T = laguerre_derivative_factor::<T>(k, j, m);
            sum += coeffs[j] * factor;
        }
        deriv_coeffs[k - m] = sum;
    }

    Ok(deriv_coeffs)
}

fn laguerre_derivative_factor<T>(k: usize, j: usize, m: usize) -> T
where
    T: Float
        + Num
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
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
        + std::ops::SubAssign
        + std::ops::MulAssign
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
    T: Float
        + Num
        + 'static
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
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
