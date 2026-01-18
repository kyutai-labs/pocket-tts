// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Window functions for signal processing

use crate::error::NumPyError;
use ndarray::Array1;
use num_traits::{Float, Zero};

pub fn bartlett<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);

    for n in 0..M {
        let value = if M == 1 {
            T::one()
        } else {
            let n_float = T::from(n).unwrap();
            let m_float = T::from(M - 1).unwrap();
            T::from(2.0).unwrap() * n_float / m_float - T::one()
        };
        window[n] = T::one() - value.abs();
    }

    Ok(window)
}

pub fn blackman<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let four_pi = T::from(4.0 * std::f64::consts::PI).unwrap();
    let six_pi = T::from(6.0 * std::f64::consts::PI).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        let term1 = T::from(0.42).unwrap();
        let term2 = T::from(0.5).unwrap() * (two_pi * n_float / m_float).cos();
        let term3 = T::from(0.08).unwrap() * (four_pi * n_float / m_float).cos();

        window[n] = term1 - term2 + term3;
    }

    Ok(window)
}

pub fn hamming<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        let alpha = T::from(0.54).unwrap();
        let beta = T::from(0.46).unwrap();

        window[n] = alpha - beta * (two_pi * n_float / m_float).cos();
    }

    Ok(window)
}

pub fn hanning<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        window[n] = T::from(0.5).unwrap() * (T::one() - (two_pi * n_float / m_float).cos());
    }

    Ok(window)
}

pub fn kaiser<T>(M: usize, beta: T) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let i0_beta = bessel_i0(beta);

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        let alpha = m_float / T::from(2.0).unwrap();
        let arg = beta * (T::one() - ((n_float - alpha) / alpha).powi(2)).sqrt();

        let bessel_arg = if arg.is_sign_positive() {
            arg
        } else {
            T::zero()
        };
        window[n] = bessel_i0(bessel_arg) / i0_beta;
    }

    Ok(window)
}

pub fn gaussian<T>(M: usize, std: T) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    if std <= T::zero() {
        return Err(NumPyError::value_error("Standard deviation must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let m_float = T::from(M - 1).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let x =
            (n_float - m_float / T::from(2.0).unwrap()) / (std * m_float / T::from(2.0).unwrap());
        window[n] = (-x * x / T::from(2.0).unwrap()).exp();
    }

    Ok(window)
}

pub fn general_gaussian<T>(M: usize, std: T, exponent: T) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    if std <= T::zero() {
        return Err(NumPyError::value_error("Standard deviation must be positive", "window"));
    }

    if exponent <= T::zero() {
        return Err(NumPyError::value_error("Exponent must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let m_float = T::from(M - 1).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let x =
            (n_float - m_float / T::from(2.0).unwrap()) / (std * m_float / T::from(2.0).unwrap());
        window[n] = (-(x.abs()).powi(exponent.to_i32().unwrap())).exp();
    }

    Ok(window)
}

pub fn boxcar<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + Zero + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    Ok(Array1::from_elem(M, T::one()))
}

pub fn triang<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        window[n] = if M % 2 == 1 {
            T::one()
                - (T::from(2.0).unwrap() * (n_float - m_float / T::from(2.0).unwrap())).abs()
                    / m_float
        } else {
            T::one()
                - (T::from(2.0).unwrap()
                    * (n_float + T::from(0.5).unwrap() - m_float / T::from(2.0).unwrap()))
                .abs()
                    / m_float
        };
    }

    Ok(window)
}

pub fn parzen<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let m_float = T::from(M).unwrap();
    let half_m = m_float / T::from(2.0).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let x = (n_float - half_m + T::from(0.5).unwrap()) / half_m;
        let abs_x = x.abs();

        window[n] = if abs_x <= T::from(0.5).unwrap() {
            T::one() - T::from(6.0).unwrap() * abs_x.powi(2) * (T::one() - abs_x)
        } else if abs_x <= T::from(1.5).unwrap() {
            T::from(2.0).unwrap() * (T::one() - abs_x).powi(3)
        } else {
            T::zero()
        };
    }

    Ok(window)
}

pub fn bohman<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let m_float = T::from(M).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let x = T::from(2.0).unwrap() * n_float / m_float - T::one();
        let abs_x = x.abs();

        window[n] = if abs_x <= T::one() {
            let pi = T::from(std::f64::consts::PI).unwrap();
            let term1 = T::one() - abs_x;
            let term2 = (pi * abs_x).cos();
            let term3 = pi * abs_x * (pi * abs_x).sin();
            term1 * term2 + term3 / pi
        } else {
            T::zero()
        };
    }

    Ok(window)
}

pub fn blackmanharris<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let four_pi = T::from(4.0 * std::f64::consts::PI).unwrap();
    let six_pi = T::from(6.0 * std::f64::consts::PI).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        let term1 = T::from(0.35875).unwrap();
        let term2 = T::from(0.48829).unwrap() * (two_pi * n_float / m_float).cos();
        let term3 = T::from(0.14128).unwrap() * (four_pi * n_float / m_float).cos();
        let term4 = T::from(0.01168).unwrap() * (six_pi * n_float / m_float).cos();

        window[n] = term1 - term2 + term3 - term4;
    }

    Ok(window)
}

pub fn flattop<T>(M: usize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let four_pi = T::from(4.0 * std::f64::consts::PI).unwrap();
    let six_pi = T::from(6.0 * std::f64::consts::PI).unwrap();
    let eight_pi = T::from(8.0 * std::f64::consts::PI).unwrap();

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        let term1 = T::from(0.21557895).unwrap();
        let term2 = T::from(0.41663158).unwrap() * (two_pi * n_float / m_float).cos();
        let term3 = T::from(0.277263158).unwrap() * (four_pi * n_float / m_float).cos();
        let term4 = T::from(0.083578947).unwrap() * (six_pi * n_float / m_float).cos();
        let term5 = T::from(0.006947368).unwrap() * (eight_pi * n_float / m_float).cos();

        window[n] = term1 - term2 + term3 - term4 + term5;
    }

    Ok(window)
}

pub fn dpss<T>(M: usize, nw: isize, k: isize) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    if k < 0 || k >= nw.min((M / 2) as isize) {
        return Err(NumPyError::value_error("k must be in range [0, min(nw, M/2))", "window"));
    }

    let nw_float = T::from(nw as f64).unwrap();
    let mut window = Array1::zeros(M);

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M).unwrap();

        let x = T::from(std::f64::consts::PI).unwrap()
            * nw_float
            * (T::from(2.0).unwrap() * n_float + T::from(1.0).unwrap())
            / m_float;

        window[n] = (x.sin() / x).powi(2);
    }

    Ok(window)
}

pub fn chebwin<T>(M: usize, attenuation: T) -> Result<Array1<T>, NumPyError>
where
    T: Float + std::fmt::Debug + 'static,
{
    if M == 0 {
        return Err(NumPyError::value_error("Window length M must be positive", "window"));
    }

    if attenuation <= T::zero() {
        return Err(NumPyError::value_error("Attenuation must be positive", "window"));
    }

    let mut window = Array1::zeros(M);
    let beta = cosh_inverse(
        T::from(10.0)
            .unwrap()
            .powf(attenuation / T::from(20.0).unwrap()),
    );

    for n in 0..M {
        let n_float = T::from(n).unwrap();
        let m_float = T::from(M - 1).unwrap();

        let x = beta * (T::from(2.0).unwrap() * n_float / m_float - T::one()).cosh();
        let cheb_poly = chebyshev_polynomial(x, M);

        window[n] = cheb_poly / beta;
    }

    Ok(window)
}

fn bessel_i0<T>(x: T) -> T
where
    T: Float + std::fmt::Debug,
{
    let x_sq = x * x / T::from(14.0625).unwrap();

    if x.abs() < T::from(3.75).unwrap() {
        let mut sum = T::one();
        let mut term = T::one();

        for k in 1..=20 {
            term *= x_sq / T::from((k * k) as f64).unwrap();
            sum += term;
        }

        sum
    } else {
        let abs_x = x.abs();
        let mut sum = T::one();
        let mut term = T::one();

        for k in 1..=20 {
            term *= -T::from(14.0625).unwrap() / (T::from((k * k) as f64).unwrap() * x_sq);
            sum += term;
        }

        (abs_x / T::from(2.5).unwrap()).exp() / abs_x.sqrt() * sum
    }
}

fn cosh_inverse<T>(x: T) -> T
where
    T: Float + std::fmt::Debug,
{
    (x + (x * x - T::one()).sqrt()).ln()
}

fn chebyshev_polynomial<T>(x: T, n: usize) -> T
where
    T: Float + std::fmt::Debug,
{
    if n == 0 {
        return T::one();
    }
    if n == 1 {
        return x;
    }

    let mut t_minus_2 = T::one();
    let mut t_minus_1 = x;

    for _ in 2..=n {
        let t_current = T::from(2.0).unwrap() * x * t_minus_1 - t_minus_2;
        t_minus_2 = t_minus_1;
        t_minus_1 = t_current;
    }

    t_minus_1
}
