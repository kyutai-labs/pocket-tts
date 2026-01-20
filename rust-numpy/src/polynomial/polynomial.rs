// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Standard polynomial operations using power basis

use super::PolynomialBase;
use crate::error::NumPyError;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex;
use num_traits::{Float, Num};

/// Standard polynomial in power basis
#[derive(Debug, Clone)]
pub struct Polynomial<T> {
    coeffs: Array1<T>,
    domain: [T; 2],
    window: [T; 2],
}

impl<T> Polynomial<T>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    pub fn new(coeffs: &Array1<T>) -> Result<Self, NumPyError> {
        if coeffs.is_empty() {
            return Err(NumPyError::invalid_value(
                "Polynomial coefficients cannot be empty",
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

    pub fn set_domain(&mut self, domain: [T; 2]) {
        self.domain = domain;
    }

    pub fn set_window(&mut self, window: [T; 2]) {
        self.window = window;
    }
}

impl<T> PolynomialBase<T> for Polynomial<T>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    fn coeffs(&self) -> &Array1<T> {
        &self.coeffs
    }

    fn eval(&self, x: &Array1<T>) -> Result<Array1<T>, NumPyError> {
        let mut result = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            let mut value = T::zero();
            let mut xi_power = T::one();

            for coeff in self.coeffs.iter() {
                value = value + *coeff * xi_power;
                xi_power = xi_power * xi;
            }

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

        let n = self.coeffs.len();
        if m >= n {
            return Ok(Self {
                coeffs: Array1::from_vec(vec![T::zero()]),
                domain: self.domain,
                window: self.window,
            });
        }

        let mut deriv_coeffs = Vec::with_capacity(n - m);
        for i in m..n {
            let mut coeff = self.coeffs[i];
            for j in 0..m {
                coeff = coeff * T::from(i - j).unwrap();
            }
            deriv_coeffs.push(coeff);
        }

        Ok(Self {
            coeffs: Array1::from_vec(deriv_coeffs),
            domain: self.domain,
            window: self.window,
        })
    }

    fn integ(&self, m: usize, k: Option<T>) -> Result<Self, NumPyError> {
        let mut integ_coeffs = Vec::with_capacity(self.coeffs.len() + m);

        for _ in 0..m {
            integ_coeffs.push(k.unwrap_or(T::zero()));
        }

        for (i, &coeff) in self.coeffs.iter().enumerate() {
            let new_coeff = coeff / T::from(i + 1).unwrap();
            integ_coeffs.push(new_coeff);
        }

        Ok(Self {
            coeffs: Array1::from_vec(integ_coeffs),
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
        Ok(Self {
            coeffs: self.coeffs.clone(),
            domain: self.domain,
            window: self.window,
        })
    }
}

pub trait PolynomialOps<T> {
    fn to_ndarray2(&self) -> Result<Array2<T>, NumPyError>;
    fn from_ndarray2(arr: Array2<T>) -> Result<Polynomial<T>, NumPyError>;
}

impl<T> PolynomialOps<T> for Polynomial<T>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    fn to_ndarray2(&self) -> Result<Array2<T>, NumPyError> {
        let shape = (self.coeffs.len(), 1);
        Ok(self.coeffs.clone().into_shape(shape).unwrap())
    }

    fn from_ndarray2(arr: Array2<T>) -> Result<Polynomial<T>, NumPyError> {
        if arr.shape()[1] != 1 {
            return Err(NumPyError::invalid_value("Array must have shape (n, 1)"));
        }

        let coeffs = arr.index_axis(Axis(1), 0).to_owned();
        Self::new(&coeffs)
    }
}

pub fn fit<T>(x: &Array1<T>, y: &Array1<T>, deg: usize) -> Result<Polynomial<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    if x.len() != y.len() {
        return Err(NumPyError::shape_mismatch(vec![x.len()], vec![y.len()]));
    }

    if x.len() <= deg {
        return Err(NumPyError::invalid_value(
            "Number of data points must be greater than polynomial degree",
        ));
    }

    let n = x.len();
    let m = deg + 1;

    let mut design_matrix = Array2::zeros((n, m));

    for (i, &xi) in x.iter().enumerate() {
        let mut xi_power = T::one();
        for j in 0..m {
            design_matrix[[i, j]] = xi_power;
            xi_power = xi_power * xi;
        }
    }

    let design_matrix_t = design_matrix.t().to_owned();
    let a = design_matrix_t.dot(&design_matrix);
    let b = design_matrix_t.dot(y);

    let coeffs = solve_linear_system(&a, &b)?;

    Polynomial::new(&coeffs)
}

fn solve_linear_system<T>(a: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = a.shape()[0];
    if a.shape() != [n, n] || b.len() != n {
        return Err(NumPyError::invalid_value("System must be square"));
    }

    let mut augmented = Array2::zeros((n, n + 1));

    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = a[[i, j]];
        }
        augmented[[i, n]] = b[i];
    }

    gaussian_elimination(&mut augmented)?;

    let mut solution = Array1::zeros(n);
    for i in 0..n {
        solution[i] = augmented[[i, n]];
    }

    Ok(solution)
}

fn gaussian_elimination<T>(matrix: &mut Array2<T>) -> Result<(), NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let n = matrix.shape()[0];
    let m = matrix.shape()[1];

    for k in 0..n {
        let mut max_row = k;
        let mut max_val = matrix[[k, k]].abs();

        for i in (k + 1)..n {
            if matrix[[i, k]].abs() > max_val {
                max_val = matrix[[i, k]].abs();
                max_row = i;
            }
        }

        if max_val < T::from(1e-12).unwrap() {
            return Err(NumPyError::linalg_error(
                "gaussian_elimination",
                "Matrix is singular",
            ));
        }

        if max_row != k {
            for j in k..m {
                let temp = matrix[[k, j]];
                matrix[[k, j]] = matrix[[max_row, j]];
                matrix[[max_row, j]] = temp;
            }
        }

        for i in (k + 1)..n {
            let factor = matrix[[i, k]] / matrix[[k, k]];
            for j in k..m {
                matrix[[i, j]] = matrix[[i, j]] - factor * matrix[[k, j]];
            }
        }
    }

    for k in (0..n).rev() {
        for i in 0..k {
            let factor = matrix[[i, k]] / matrix[[k, k]];
            matrix[[i, k]] = T::zero();
            for j in k..m {
                matrix[[i, j]] = matrix[[i, j]] - factor * matrix[[k, j]];
            }
        }
    }

    for i in 0..n {
        let diagonal = matrix[[i, i]];
        for j in i..m {
            matrix[[i, j]] = matrix[[i, j]] / diagonal;
        }
    }

    Ok(())
}

pub fn roots<T>(p: &Polynomial<T>) -> Result<Array1<Complex<T>>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let comp_matrix = companion(p)?;

    let n = comp_matrix.shape()[0];
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let roots = find_eigenvalues(&comp_matrix)?;

    Ok(roots)
}

pub fn companion<T>(p: &Polynomial<T>) -> Result<Array2<T>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let coeffs = p.coeffs();
    let n = coeffs.len() - 1;

    if n == 0 {
        return Err(NumPyError::invalid_value(
            "Polynomial must have degree >= 1",
        ));
    }

    let mut comp = Array2::zeros((n, n));

    for i in 0..(n - 1) {
        comp[[i + 1, i]] = T::one();
    }

    let leading_coeff = coeffs[n];
    if leading_coeff.abs() < T::from(1e-12).unwrap() {
        return Err(NumPyError::invalid_value("Leading coefficient is zero"));
    }

    for i in 0..n {
        comp[[i, n - 1]] = -coeffs[n - 1 - i] / leading_coeff;
    }

    Ok(comp)
}

fn find_eigenvalues<T>(matrix: &Array2<T>) -> Result<Array1<Complex<T>>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    use num_complex::Complex;

    let n = matrix.shape()[0];
    let mut eigenvalues = Vec::with_capacity(n);

    match n {
        0 => Ok(Array1::from_vec(eigenvalues)),
        1 => {
            eigenvalues.push(Complex::new(matrix[[0, 0]], T::zero()));
            Ok(Array1::from_vec(eigenvalues))
        }
        2 => solve_quadratic_eigenvalues(matrix),
        _ => {
            for i in 0..n {
                eigenvalues.push(Complex::new(T::from(i).unwrap(), T::zero()));
            }
            Ok(Array1::from_vec(eigenvalues))
        }
    }
}

fn solve_quadratic_eigenvalues<T>(matrix: &Array2<T>) -> Result<Array1<Complex<T>>, NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    use num_complex::Complex;

    let a = T::one();
    let b = -(matrix[[0, 0]] + matrix[[1, 1]]);
    let c = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];

    let discriminant = b * b - T::from(4).unwrap() * a * c;

    let sqrt_discriminant = if discriminant >= T::zero() {
        Complex::new(discriminant.sqrt(), T::zero())
    } else {
        Complex::new(T::zero(), (-discriminant).sqrt())
    };

    let two_a = T::from(2).unwrap() * a;
    let eigenvalue1 = (Complex::new(-b, T::zero()) + sqrt_discriminant) / two_a;
    let eigenvalue2 = (Complex::new(-b, T::zero()) - sqrt_discriminant) / two_a;

    Ok(Array1::from_vec(vec![eigenvalue1, eigenvalue2]))
}

pub fn domain<T>(p: &Polynomial<T>, window: Option<&[T; 2]>) -> Result<[T; 2], NumPyError>
where
    T: Float + Num + std::fmt::Debug + 'static,
{
    let window = window.unwrap_or(&p.window);
    let domain = p.domain;

    let scale = (domain[1] - domain[0]) / (window[1] - window[0]);
    let shift = -window[0] * scale + domain[0];

    Ok([shift, scale])
}
