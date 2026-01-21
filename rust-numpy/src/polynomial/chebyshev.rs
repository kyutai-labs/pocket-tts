// Copyright 2024 The NumPyRS Authors.

use super::{Polynomial, PolynomialBase};
use crate::error::NumPyError;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, Num};

/// Chebyshev polynomials of the first kind
#[derive(Debug, Clone)]
pub struct Chebyshev<T> {
    coeffs: Array1<T>,
    domain: [T; 2],
    window: [T; 2],
}

impl<T> Chebyshev<T>
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
                "Chebyshev coefficients cannot be empty",
            ));
        }

        let coeffs = coeffs.to_owned();
        let domain = [T::from(-1.0).unwrap(), T::one()];
        let window = [T::from(-1.0).unwrap(), T::one()];

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
        let cheb_coeffs = polynomial_to_chebyshev(poly.coeffs())?;
        Ok(Self {
            coeffs: cheb_coeffs,
            domain: poly.domain(),
            window: poly.window(),
        })
    }

    pub fn from_roots(roots: &Array1<T>) -> Result<Self, NumPyError> {
        let n = roots.len();
        let mut coeffs = Array1::zeros(n + 1);
        coeffs[n] = T::from(2.0_f64).unwrap().powi(-(n as i32 - 1));

        for (i, &root) in roots.iter().enumerate() {
            let mut temp_coeffs = coeffs.clone();
            for j in 0..=(n - i) {
                if j > 0 {
                    temp_coeffs[j - 1] -= root * coeffs[j];
                }
            }
            coeffs = temp_coeffs;
        }

        Self::new(&coeffs)
    }
}

impl<T> PolynomialBase<T> for Chebyshev<T>
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
            let value = chebyshev_eval_recursive(&self.coeffs, xi_window);
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

        let deriv_coeffs = chebyshev_derivative_coeffs(&self.coeffs, m)?;
        Ok(Self {
            coeffs: deriv_coeffs,
            domain: self.domain,
            window: self.window,
        })
    }

    fn integ(&self, m: usize, k: Option<T>) -> Result<Self, NumPyError> {
        let integ_coeffs = chebyshev_integral_coeffs(&self.coeffs, m, k)?;
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
        let poly_coeffs = chebyshev_to_polynomial(&self.coeffs)?;
        Polynomial::new(&poly_coeffs)
    }
}

fn chebyshev_eval_recursive<T>(coeffs: &Array1<T>, x: T) -> T
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

    let mut b2 = T::zero();
    let mut b1 = T::zero();

    for (i, &coeff) in coeffs.iter().rev().enumerate() {
        let temp = T::from(2.0).unwrap() * x * b1 - b2 + coeff;
        b2 = b1;
        b1 = temp;
    }

    b1 - x * b2
}

fn polynomial_to_chebyshev<T>(poly_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
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
    let mut cheb_coeffs = Array1::zeros(n);

    for k in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            if j <= k && (k - j) % 2 == 0 {
                let coeff = poly_coeffs[j];
                let binomial = binomial_coefficient(k, (k - j) / 2);
                sum += coeff * binomial * T::from(2.0_f64).unwrap().powi(j as i32);
            }
        }
        cheb_coeffs[k] = sum / T::from(2.0_f64).unwrap().powi(k as i32);
    }

    Ok(cheb_coeffs)
}

fn chebyshev_to_polynomial<T>(cheb_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
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
    let n = cheb_coeffs.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let mut poly_coeffs = Array1::zeros(n);
    if n == 1 {
        poly_coeffs[0] = cheb_coeffs[0];
        return Ok(poly_coeffs);
    }

    // Use recurrence relation: T_{n+1} = 2x * T_n - T_{n-1}
    // We maintain the power-basis coefficients of T_{k-1} and T_k
    let mut c_prev = Array1::zeros(n);
    let mut c_curr = Array1::zeros(n);

    // T0 = 1
    c_prev[0] = T::one();
    poly_coeffs[0] += cheb_coeffs[0] * c_prev[0];

    // T1 = x
    c_curr[1] = T::one();
    if n > 1 {
        poly_coeffs[0] += cheb_coeffs[1] * c_curr[0];
        poly_coeffs[1] += cheb_coeffs[1] * c_curr[1];
    }

    for k in 2..n {
        let mut c_next = Array1::zeros(n);
        // T_{k} = 2x * T_{k-1} - T_{k-2}
        // (2x * T_{k-1})[i] = 2 * T_{k-1}[i-1]
        for i in 1..n {
            c_next[i] += T::from(2.0).unwrap() * c_curr[i - 1];
        }
        for i in 0..n {
            c_next[i] -= c_prev[i];
        }

        for i in 0..n {
            poly_coeffs[i] += cheb_coeffs[k] * c_next[i];
        }
        c_prev = c_curr;
        c_curr = c_next;
    }

    Ok(poly_coeffs)
}

fn chebyshev_derivative_coeffs<T>(coeffs: &Array1<T>, m: usize) -> Result<Array1<T>, NumPyError>
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
        for j in (m..=k).step_by(2) {
            let factor = T::from(j).unwrap() * chebyshev_derivative_factor(j, m);
            sum += coeffs[k - j] * factor;
        }
        deriv_coeffs[k - m] = sum;
    }

    Ok(deriv_coeffs)
}

fn chebyshev_derivative_factor<T>(j: usize, m: usize) -> T
where
    T: Float
        + Num
        + 'static
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if m > j {
        return T::zero();
    }

    let mut factor = T::one();
    for r in 0..m {
        factor *= T::from(j * j - r * r).unwrap();
    }

    factor / T::from(2.0_f64).unwrap().powi(m as i32)
}

fn chebyshev_integral_coeffs<T>(
    coeffs: &Array1<T>,
    m: usize,
    k: Option<T>,
) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = coeffs.len();
    let mut integ_coeffs = Array1::zeros(n + m);

    if let Some(kval) = k {
        for i in 0..m {
            integ_coeffs[i] = kval;
        }
    }

    for k in 0..n {
        if k == 0 {
            integ_coeffs[k + m] = coeffs[k];
        } else {
            let factor = T::from(2.0_f64).unwrap() / T::from(k).unwrap();
            integ_coeffs[k + m] = coeffs[k] * factor;
        }
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
