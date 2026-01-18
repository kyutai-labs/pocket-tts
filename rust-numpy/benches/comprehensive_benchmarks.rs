use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_numpy::*;
use std::time::Instant;

fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");
    
    group.bench_function("zeros_100", |b| {
        b.iter(|| {
            let _ = Array::<f64>::zeros(vec![100]);
        });
    });
    
    group.bench_function("zeros_1000", |b| {
        b.iter(|| {
            let _ = Array::<f64>::zeros(vec![1000]);
        });
    });
    
    group.finish();
}

fn bench_array_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_ops");
    
    let arr = black_box(Array::<f64>::zeros(vec![1000]));
    
    group.bench_function("transpose", |b| {
        b.iter(|| {
            let _ = arr.transpose();
        });
    });
    
    group.bench_function("reshape", |b| {
        b.iter(|| {
            let _ = arr.reshape(&[100, 10]).unwrap();
        });
    });
    
    group.finish();
}

fn bench_math_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_ops");
    
    let arr = black_box(Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = arr.sin();
        });
    });
    
    group.bench_function("exp", |b| {
        b.iter(|| {
            let _ = arr.exp();
        });
    });
    
    group.bench_function("log", |b| {
        b.iter(|| {
            let _ = arr.log();
        });
    });
    
    group.finish();
}

/// Benchmark SIMD operations
#[cfg(feature = "simd")]
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd");
    
    let arr = black_box(Array::from_vec((0..10000).map(|i| i as f64).collect()));
    
    #[cfg(target_arch = "x86_64")]
    {
        group.bench_function("sin_simd", |b| {
            use crate::simd_ops;
            b.iter(|| {
                let _ = simd_ops::simd_sin_f64(&arr.to_vec());
            });
        });
        
        group.bench_function("cos_simd", |b| {
            use crate::simd_ops;
            b.iter(|| {
                let _ = simd_ops::simd_cos_f64(&arr.to_vec());
            });
        });
        
        group.bench_function("exp_simd", |b| {
            use crate::simd_ops;
            b.iter(|| {
                let _ = simd_ops::simd_exp_f64(&arr.to_vec());
            });
        });
        
        group.bench_function("log_simd", |b| {
            use crate::simd_ops;
            b.iter(|| {
                let _ = simd_ops::simd_log_f64(&arr.to_vec());
            });
        });
    }
    
    group.finish();
}

/// Benchmark parallel operations
#[cfg(feature = "rayon")]
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel");
    
    let arr = black_box(Array::from_vec((0..10000).map(|i| i as f64).collect()));
    
    group.bench_function("sum_sequential", |b| {
        b.iter(|| {
            let _ = arr.sum(None, false).unwrap();
        });
    });
    
    #[cfg(feature = "rayon")]
    group.bench_function("sum_parallel", |b| {
        use crate::parallel_ops;
        b.iter(|| {
            let _ = parallel_ops::parallel_sum(&arr).unwrap();
        });
    });
    
    group.bench_function("mean_sequential", |b| {
        b.iter(|| {
            let _ = arr.mean(None, false).unwrap();
        });
    });
    
    #[cfg(feature = "rayon")]
    group.bench_function("mean_parallel", |b| {
        use crate::parallel_ops;
        b.iter(|| {
            let _ = parallel_ops::parallel_mean(&arr).unwrap();
        });
    });
    
    group.finish();
}

/// Benchmark memory optimizations
fn bench_memory_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");
    
    let arr = black_box(Array::from_vec((0..10000).map(|i| i as f64).collect()));
    let scalar_arr = black_box(Array::from_vec(vec![1.0f64; 1000]));
    
    group.bench_function("to_vec_allocation", |b| {
        b.iter(|| {
            let _ = arr.to_vec();
        });
    });
    
    group.bench_function("broadcast_scalar_copy", |b| {
        use crate::broadcasting;
        b.iter(|| {
            let mut output = arr.clone();
            let _ = broadcasting::broadcast_to(&scalar_arr, &[10000]).unwrap();
        });
    });
    
    group.finish();
}

/// Benchmark advanced broadcasting
fn bench_advanced_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_broadcasting");
    
    let arr = black_box(Array::from_vec((0..1000).map(|i| i as f64).collect()));
    
    group.bench_function("repeat_axis_0", |b| {
        use crate::advanced_broadcast;
        b.iter(|| {
            let _ = advanced_broadcast::repeat(&arr, 5, Some(0)).unwrap();
        });
    });
    
    group.bench_function("tile_basic", |b| {
        use crate::advanced_broadcast;
        b.iter(|| {
            let _ = advanced_broadcast::tile(&arr, &[10, 10]).unwrap();
        });
    });
    
    group.bench_function("broadcast_to_large", |b| {
        use crate::advanced_broadcast;
        b.iter(|| {
            let _ = advanced_broadcast::broadcast_to_enhanced(&arr, &[1000, 10]).unwrap();
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_array_ops,
    bench_math_ops,
    bench_simd_operations,
    bench_parallel_operations,
    bench_memory_optimizations,
    bench_advanced_broadcasting
);

criterion_main!(benches);
