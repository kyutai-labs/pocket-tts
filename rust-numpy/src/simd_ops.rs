// SIMD-optimized operations for mathematical ufuncs
//
// This module provides SIMD-optimized implementations of common mathematical
// operations using architecture-specific intrinsics for maximum performance.

#[cfg(feature = "simd")]
use stdsimd::prelude::*;

/// SIMD chunk size for different architectures
#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdChunkSize {
    /// 256-bit vectors (AVX2): 4x f64, 8x f32
    Avx2,
    /// 512-bit vectors (AVX-512): 8x f64, 16x f32
    Avx512,
    /// 128-bit vectors (NEON/SSE): 2x f64, 4x f32
    Sse128,
    /// Scalar fallback
    Scalar,
}

#[cfg(feature = "simd")]
impl SimdChunkSize {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[cfg(target_feature = "avx2")]
    const fn new() -> Self {
        if is_x86_feature_detected!("avx512f") {
            SimdChunkSize::Avx512
        } else {
            SimdChunkSize::Avx2
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[cfg(not(target_feature = "avx2"))]
    const fn new() -> Self {
        SimdChunkSize::Sse128
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    const fn new() -> Self {
        SimdChunkSize::Sse128
    }

    #[inline]
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const fn new() -> Self {
        SimdChunkSize::Scalar
    }

    #[inline]
    pub fn chunk_size_f64(&self) -> usize {
        match self {
            SimdChunkSize::Avx2 => 4,
            SimdChunkSize::Avx512 => 8,
            SimdChunkSize::Sse128 => 2,
            SimdChunkSize::Scalar => 1,
        }
    }

    #[inline]
    pub fn chunk_size_f32(&self) -> usize {
        match self {
            SimdChunkSize::Avx2 => 8,
            SimdChunkSize::Avx512 => 16,
            SimdChunkSize::Sse128 => 4,
            SimdChunkSize::Scalar => 1,
        }
    }
}

/// Process array using SIMD-optimized sin function
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
pub fn simd_sin_f64(values: &[f64]) -> Vec<f64> {
    #[cfg(target_feature = "avx2")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_sin_f64_avx2(values);
        }
    }

    #[cfg(target_feature = "sse4.1")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return simd_sin_f64_sse(values);
        }
    }

    // Fallback to scalar
    values.iter().copied().map(|x| x.sin()).collect()
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn simd_sin_f64_avx2(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    // Process 4 values at a time using AVX2
    while i + 4 <= values.len() {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let s = _mm256_sin_pd(v);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), s);
        i += 4;
    }

    // Process remaining values
    while i < values.len() {
        result.push(values[i].sin());
        i += 1;
    }

    result.set_len(values.len());
    result
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "sse4.1")]
#[inline(always)]
unsafe fn simd_sin_f64_sse(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    // Process 2 values at a time using SSE
    while i + 2 <= values.len() {
        let v = _mm_loadu_pd(values.as_ptr().add(i));
        let s = _mm_sin_pd(v);
        _mm_storeu_pd(result.as_mut_ptr().add(i), s);
        i += 2;
    }

    // Process remaining values
    while i < values.len() {
        result.push(values[i].sin());
        i += 1;
    }

    result.set_len(values.len());
    result
}

/// Process array using SIMD-optimized cos function
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
pub fn simd_cos_f64(values: &[f64]) -> Vec<f64> {
    #[cfg(target_feature = "avx2")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_cos_f64_avx2(values);
        }
    }

    #[cfg(target_feature = "sse4.1")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return simd_cos_f64_sse(values);
        }
    }

    // Fallback to scalar
    values.iter().copied().map(|x| x.cos()).collect()
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn simd_cos_f64_avx2(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    while i + 4 <= values.len() {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let c = _mm256_cos_pd(v);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), c);
        i += 4;
    }

    while i < values.len() {
        result.push(values[i].cos());
        i += 1;
    }

    result.set_len(values.len());
    result
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "sse4.1")]
#[inline(always)]
unsafe fn simd_cos_f64_sse(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    while i + 2 <= values.len() {
        let v = _mm_loadu_pd(values.as_ptr().add(i));
        let c = _mm_cos_pd(v);
        _mm_storeu_pd(result.as_mut_ptr().add(i), c);
        i += 2;
    }

    while i < values.len() {
        result.push(values[i].cos());
        i += 1;
    }

    result.set_len(values.len());
    result
}

/// Process array using SIMD-optimized exp function
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
pub fn simd_exp_f64(values: &[f64]) -> Vec<f64> {
    #[cfg(target_feature = "avx2")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_exp_f64_avx2(values);
        }
    }

    // Fallback to scalar
    values.iter().copied().map(|x| x.exp()).collect()
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn simd_exp_f64_avx2(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    while i + 4 <= values.len() {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let e = _mm256_exp_pd(v);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), e);
        i += 4;
    }

    while i < values.len() {
        result.push(values[i].exp());
        i += 1;
    }

    result.set_len(values.len());
    result
}

/// Process array using SIMD-optimized log function
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
pub fn simd_log_f64(values: &[f64]) -> Vec<f64> {
    #[cfg(target_feature = "avx2")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_log_f64_avx2(values);
        }
    }

    // Fallback to scalar
    values.iter().copied().map(|x| x.ln()).collect()
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn simd_log_f64_avx2(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    while i + 4 <= values.len() {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let l = _mm256_log_pd(v);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), l);
        i += 4;
    }

    while i < values.len() {
        result.push(values[i].ln());
        i += 1;
    }

    result.set_len(values.len());
    result
}

/// Process array using SIMD-optimized sqrt function
#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
pub fn simd_sqrt_f64(values: &[f64]) -> Vec<f64> {
    #[cfg(target_feature = "avx2")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_sqrt_f64_avx2(values);
        }
    }

    #[cfg(target_feature = "sse4.1")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return simd_sqrt_f64_sse(values);
        }
    }

    // Fallback to scalar
    values.iter().copied().map(|x| x.sqrt()).collect()
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn simd_sqrt_f64_avx2(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    while i + 4 <= values.len() {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let s = _mm256_sqrt_pd(v);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), s);
        i += 4;
    }

    while i < values.len() {
        result.push(values[i].sqrt());
        i += 1;
    }

    result.set_len(values.len());
    result
}

#[cfg(feature = "simd")]
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "sse4.1")]
#[inline(always)]
unsafe fn simd_sqrt_f64_sse(values: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;

    while i + 2 <= values.len() {
        let v = _mm_loadu_pd(values.as_ptr().add(i));
        let s = _mm_sqrt_pd(v);
        _mm_storeu_pd(result.as_mut_ptr().add(i), s);
        i += 2;
    }

    while i < values.len() {
        result.push(values[i].sqrt());
        i += 1;
    }

    result.set_len(values.len());
    result
}

/// Scalar fallback for architectures without SIMD support
#[cfg(any(
    not(feature = "simd"),
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64")
))]
pub fn simd_sin_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.sin()).collect()
}

#[cfg(any(
    not(feature = "simd"),
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64")
))]
pub fn simd_cos_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.cos()).collect()
}

#[cfg(any(
    not(feature = "simd"),
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64")
))]
pub fn simd_exp_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.exp()).collect()
}

#[cfg(any(
    not(feature = "simd"),
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64")
))]
pub fn simd_log_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.ln()).collect()
}

#[cfg(any(
    not(feature = "simd"),
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64")
))]
pub fn simd_sqrt_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.sqrt()).collect()
}

/// Benchmark helper to compare SIMD vs scalar performance
#[cfg(feature = "simd")]
pub fn benchmark_simd_vs_scalar<T>(
    name: &str,
    scalar_fn: impl Fn(&[T]) -> Vec<T>,
    simd_fn: impl Fn(&[T]) -> Vec<T>,
) where
    T: Copy + std::fmt::Debug,
{
    use std::time::Instant;

    let test_data: Vec<T> = (0..10000)
        .map(|i| match std::any::TypeId::of::<T>() {
            std::any::TypeId::of::<f64>() => unsafe {
                std::mem::transmute_copy::<i64, f64>(i as i64)
            },
            _ => unsafe { std::mem::transmute_copy::<i32, T>(i as i32) },
        })
        .collect();

    let start = Instant::now();
    let _scalar_result = scalar_fn(&test_data);
    let scalar_time = start.elapsed();

    let start = Instant::now();
    let _simd_result = simd_fn(&test_data);
    let simd_time = start.elapsed();

    println!("Benchmark: {}", name);
    println!("  Scalar: {:?}", scalar_time);
    println!("  SIMD:   {:?}", simd_time);
    println!(
        "  Speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_sin() {
        let input = vec![
            0.0f64,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
        ];
        let result = simd_sin_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.sin()).abs() < 1e-10);
        assert!((result[1] - (std::f64::consts::PI / 4.0).sin()).abs() < 1e-10);
        assert!((result[2] - (std::f64::consts::PI / 2.0).sin()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_cos() {
        let input = vec![0.0f64, std::f64::consts::PI / 3.0, std::f64::consts::PI];
        let result = simd_cos_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.cos()).abs() < 1e-10);
        assert!((result[1] - (std::f64::consts::PI / 3.0).cos()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_exp() {
        let input = vec![0.0f64, 1.0, 2.0];
        let result = simd_exp_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.exp()).abs() < 1e-10);
        assert!((result[1] - 1.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_log() {
        let input = vec![1.0f64, std::f64::consts::E, 10.0];
        let result = simd_log_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 1.0_f64.ln()).abs() < 1e-10);
        assert!((result[1] - std::f64::consts::E.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_sqrt() {
        let input = vec![0.0f64, 1.0, 4.0, 9.0, 16.0];
        let result = simd_sqrt_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.sqrt()).abs() < 1e-10);
        assert!((result[2] - 4.0_f64.sqrt()).abs() < 1e-10);
    }
}
