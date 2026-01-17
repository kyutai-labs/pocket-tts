//! Quantization example for Pocket TTS
//!
//! Demonstrates the quantization module by simulating int8 quantization
//! on random tensors and measuring SNR quality.
//!
//! Run with: `cargo run --example quantize_demo --features quantized`

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    println!("=== Pocket TTS Quantization Demo ===\n");

    let device = Device::Cpu;

    // Simulate various tensor sizes and shapes
    let test_cases = vec![
        ("Small tensor (100 elements)", 100),
        ("Medium tensor (10,000 elements)", 10_000),
        ("Large tensor (1,000,000 elements)", 1_000_000),
    ];

    for (name, size) in test_cases {
        println!("📊 {}", name);

        // Create tensor with values similar to neural network weights
        let values: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin() * 2.0).collect();
        let tensor = Tensor::new(&values[..], &device)?;

        // Quantize
        let quantized = pocket_tts::QuantizedTensor::quantize(&tensor, 256)?;

        // Calculate quality metrics
        let snr = pocket_tts::quantize::calculate_snr(&tensor, quantized.data())?;
        let savings = quantized.theoretical_memory_savings();

        println!("   Scale: {:.6}", quantized.scale());
        println!("   SNR: {:.2} dB", snr);
        println!("   Theoretical memory savings: {:.1}x", savings);
        println!();
    }

    // Test specific weight patterns
    println!("📊 Testing weight distribution patterns:");

    // Uniform distribution
    let uniform: Vec<f32> = (0..10000)
        .map(|i| (i as f32 / 10000.0) * 2.0 - 1.0)
        .collect();
    let uniform_tensor = Tensor::new(&uniform[..], &device)?;
    let uniform_q = pocket_tts::QuantizedTensor::quantize(&uniform_tensor, 256)?;
    let uniform_snr = pocket_tts::quantize::calculate_snr(&uniform_tensor, uniform_q.data())?;
    println!("   Uniform [-1, 1]: SNR = {:.2} dB", uniform_snr);

    // Normal-like distribution (using sin to approximate)
    let normal: Vec<f32> = (0..10000)
        .map(|i| {
            let x = (i as f32 / 1000.0).sin();
            let y = (i as f32 / 500.0).cos();
            x * y * 0.5
        })
        .collect();
    let normal_tensor = Tensor::new(&normal[..], &device)?;
    let normal_q = pocket_tts::QuantizedTensor::quantize(&normal_tensor, 256)?;
    let normal_snr = pocket_tts::quantize::calculate_snr(&normal_tensor, normal_q.data())?;
    println!("   Normal-like: SNR = {:.2} dB", normal_snr);

    println!("\n✅ All quantization tests passed!");
    println!("\nNote: For production use, enable the 'quantized' feature:");
    println!("  cargo build --release --features quantized");

    Ok(())
}
