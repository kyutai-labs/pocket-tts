// Copyright 2024 The NumPyRS Authors.

use super::{Polynomial, PolynomialBase};
use crate::error::NumPyError;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, Num, One, Zero};

/// Legendre polynomials
#[derive(Debug, Clone)]
pub struct Legendre<T> {
    coeffs: Array1<T>,
    domain: [T; 2],
    window: [T; 2],
}

impl<T> Legendre<T>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    pub fn new(coeffs: &Array1<T>) -> Result<Self, NumPyError> {
        if coeffs.len() == 0 {
            return Err(NumPyError::invalid_value(
                "Legendre coefficients cannot be empty",
                "legendre",
            ));
        }

        let coeffs = coeffs.to_owned();
        let domain = [T::one(), T::one()];
        let window = [T::one(), T::one()];

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
        let leg_coeffs = polynomial_to_legendre(poly.coeffs())?;
        Ok(Self {
            coeffs: leg_coeffs,
            domain: poly.domain(),
            window: poly.window(),
        })
    }
}

impl<T> PolynomialBase<T> for Legendre<T>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    fn coeffs(&self) -> &Array1<T> {
        &self.coeffs
    }

    fn eval(&self, x: &Array1<T>) -> Result<Array1<T>, NumPyError> {
        let mut result = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            let xi_scaled = (T::from(2.0).unwrap() * xi - (self.window[0] + self.window[1]))
                / (self.window[1] - self.window[0]);

            let value = legendre_eval_recursive(&self.coeffs, xi_scaled);
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

        let deriv_coeffs = legendre_derivative_coeffs(&self.coeffs, m)?;
        Ok(Self {
            coeffs: deriv_coeffs,
            domain: self.domain,
            window: self.window,
        })
    }

    fn integ(&self, m: usize, k: Option<T>) -> Result<Self, NumPyError> {
        let integ_coeffs = legendre_integral_coeffs(&self.coeffs, m, k)?;
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
        let poly_coeffs = legendre_to_polynomial(&self.coeffs)?;
        Polynomial::new(&poly_coeffs)
    }
}

fn legendre_eval_recursive<T>(coeffs: &Array1<T>, x: T) -> T
where
    T: Float + Num + std::fmt::Debug,
{
    if coeffs.len() == 0 {
        return T::zero();
    }

    let mut p_minus_2 = T::one();
    let mut p_minus_1 = x;
    let mut result = coeffs[0] * p_minus_2;

    if coeffs.len() > 1 {
        result += coeffs[1] * p_minus_1;
    }

    for n in 2..coeffs.len() {
        let p_n = (T::from(2.0_f64).unwrap() * T::from(n - 1).unwrap() * x * p_minus_1
            - T::from(n - 1).unwrap() * p_minus_2)
            / T::from(n).unwrap();
        result += coeffs[n] * p_n;
        p_minus_2 = p_minus_1;
        p_minus_1 = p_n;
    }

    result
}

fn polynomial_to_legendre<T>(poly_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = poly_coeffs.len();
    let mut leg_coeffs = Array1::zeros(n);

    for k in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            if j <= k && (k - j) % 2 == 0 {
                let coeff = poly_coeffs[j];
                let binomial1 = binomial_coefficient(k, (k - j) / 2);
                let binomial2 = binomial_coefficient(k + j, j);
                sum += coeff
                    * binomial1
                    * binomial2
                    * T::from((-1.0_f64).powi(((k - j) / 2) as i32)).unwrap();
            }
        }
        leg_coeffs[k] = sum / T::from(2.0_f64).unwrap().powi(k as i32);
    }

    Ok(leg_coeffs)
}

fn legendre_to_polynomial<T>(leg_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = leg_coeffs.len();
    let mut poly_coeffs = Array1::zeros(n);

    for k in 0..n {
        let coeff = leg_coeffs[k];

        for j in 0..=k {
            if (k - j) % 2 == 0 {
                let power = (k - j) / 2;
                let binomial1 = binomial_coefficient(k, j);
                let binomial2 = binomial_coefficient((k + j) / 2, power);
                poly_coeffs[j] +=
                    coeff * binomial1 * binomial2 * T::from((-1.0_f64).powi(power as i32)).unwrap();
            }
        }
    }

    Ok(poly_coeffs)
}

fn legendre_derivative_coeffs<T>(coeffs: &Array1<T>, m: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
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
            let factor = legendre_derivative_factor(j, m);
            sum += coeffs[k - j] * factor;
        }
        deriv_coeffs[k - m] = sum;
    }

    Ok(deriv_coeffs)
}

fn legendre_derivative_factor<T>(j: usize, m: usize) -> T
where
    T: Float + Num,
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

fn legendre_integral_coeffs<T>(
    coeffs: &Array1<T>,
    m: usize,
    k: Option<T>,
) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = coeffs.len();
    let mut integ_coeffs = Array1::zeros(n + m);

    for _ in 0..m {
        integ_coeffs.push(k.unwrap_or(T::zero()));
    }

    for k in 0..n {
        if k == 0 {
            integ_coeffs[k + m] = coeffs[k];
        } else {
            let factor = T::from(2.0_f64).unwrap() / T::from(k + 1).unwrap();
            integ_coeffs[k + m] = coeffs[k] * factor;
        }
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
