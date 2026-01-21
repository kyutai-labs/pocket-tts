// Copyright 2024 The NumPyRS Authors.

use super::{Polynomial, PolynomialBase};
use crate::error::NumPyError;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, Num};

/// HermiteE polynomials (probabilist's version)
#[derive(Debug, Clone)]
pub struct HermiteE<T> {
    coeffs: Array1<T>,
    domain: [T; 2],
    window: [T; 2],
}

impl<T> HermiteE<T>
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
                "HermiteE coefficients cannot be empty",
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
        let herme_coeffs = polynomial_to_hermite_e(poly.coeffs())?;
        Ok(Self {
            coeffs: herme_coeffs,
            domain: poly.domain(),
            window: poly.window(),
        })
    }

    pub fn from_hermite(hermite: &super::hermite::Hermite<T>) -> Result<Self, NumPyError>
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
        let herme_coeffs = hermite_to_hermite_e(hermite.coeffs())?;
        Ok(Self {
            coeffs: herme_coeffs,
            domain: hermite.domain(),
            window: hermite.window(),
        })
    }
}

impl<T> PolynomialBase<T> for HermiteE<T>
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
            let value = hermite_e_eval_recursive(&self.coeffs, xi_window);
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

        let deriv_coeffs = hermite_e_derivative_coeffs(&self.coeffs, m)?;
        Ok(Self {
            coeffs: deriv_coeffs,
            domain: self.domain,
            window: self.window,
        })
    }

    fn integ(&self, m: usize, k: Option<T>) -> Result<Self, NumPyError> {
        let integ_coeffs = hermite_e_integral_coeffs(&self.coeffs, m, k)?;
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
        let poly_coeffs = hermite_e_to_polynomial(&self.coeffs)?;
        Polynomial::new(&poly_coeffs)
    }
}

fn hermite_e_eval_recursive<T>(coeffs: &Array1<T>, x: T) -> T
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

    let mut he_minus_2 = T::one();
    let mut he_minus_1 = x;
    let mut result = coeffs[0] * he_minus_2;

    if coeffs.len() > 1 {
        result += coeffs[1] * he_minus_1;
    }

    for n in 2..coeffs.len() {
        let he_n = x * he_minus_1 - T::from(n - 1).unwrap() * he_minus_2;
        result += coeffs[n] * he_n;
        he_minus_2 = he_minus_1;
        he_minus_1 = he_n;
    }

    result
}

fn polynomial_to_hermite_e<T>(poly_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
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
    let mut herme_coeffs = Array1::zeros(n);

    for k in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            if j <= k && (k - j) % 2 == 0 {
                let coeff = poly_coeffs[j];
                let power = (k - j) / 2;
                let factorial = factorial(k) / factorial(power);
                let binomial = binomial_coefficient(k, power);
                sum += coeff * T::from(factorial).unwrap() * binomial;
            }
        }
        herme_coeffs[k] = sum;
    }

    Ok(herme_coeffs)
}

fn hermite_e_to_polynomial<T>(herme_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
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
    let n = herme_coeffs.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let mut poly_coeffs = Array1::zeros(n);
    if n == 1 {
        poly_coeffs[0] = herme_coeffs[0];
        return Ok(poly_coeffs);
    }

    let mut c_prev = Array1::zeros(n);
    let mut c_curr = Array1::zeros(n);

    // He0 = 1
    c_prev[0] = T::one();
    poly_coeffs[0] += herme_coeffs[0] * c_prev[0];

    // He1 = x
    c_curr[1] = T::one();
    poly_coeffs[0] += herme_coeffs[1] * c_curr[0];
    poly_coeffs[1] += herme_coeffs[1] * c_curr[1];

    for k in 1..(n - 1) {
        let mut c_next = Array1::zeros(n);
        let fk = T::from(k).unwrap();

        // He_{k+1} = x * He_k - k * He_{k-1}
        for i in 1..n {
            c_next[i] += c_curr[i - 1];
        }
        for i in 0..n {
            c_next[i] -= fk * c_prev[i];
        }

        for i in 0..n {
            poly_coeffs[i] += herme_coeffs[k + 1] * c_next[i];
        }
        c_prev = c_curr.clone();
        c_curr = c_next;
    }

    Ok(poly_coeffs)
}

fn hermite_to_hermite_e<T>(herm_coeffs: &Array1<T>) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = herm_coeffs.len();
    let mut herme_coeffs = Array1::zeros(n);

    for k in 0..n {
        let coeff = herm_coeffs[k];
        let factor = T::from(2.0_f64).unwrap().powi(-(k as i32) / 2);
        herme_coeffs[k] = coeff * factor;
    }

    Ok(herme_coeffs)
}

fn hermite_e_derivative_coeffs<T>(coeffs: &Array1<T>, m: usize) -> Result<Array1<T>, NumPyError>
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
            let factor = hermite_e_derivative_factor(j, m);
            sum += coeffs[k - j] * factor;
        }
        deriv_coeffs[k - m] = sum;
    }

    Ok(deriv_coeffs)
}

fn hermite_e_derivative_factor<T>(j: usize, m: usize) -> T
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
        factor *= T::from(j - r).unwrap();
    }

    factor
}

fn hermite_e_integral_coeffs<T>(
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
            let factor = T::one() / T::from(k + 1).unwrap();
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

fn factorial(n: usize) -> usize {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
