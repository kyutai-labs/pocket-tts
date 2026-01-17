//! Quantization support for Pocket TTS
//!
//! This module provides quantization utilities for reduced memory footprint
//! and potentially faster inference on CPU.
//!
//! Note: Candle doesn't natively support int8 tensor operations, so we use
//! a simulated quantization approach that stores quantized values as f32 but
//! represents them using only 256 discrete levels (mimicking int8 range).
//!
//! For true int8 acceleration, use GGML/GGUF format weights with candle-quantized.

use anyhow::Result;
use candle_core::{DType, Tensor};
use std::collections::HashMap;

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Layers to skip quantization (keep in full precision)
    pub skip_layers: Vec<String>,
    /// Minimum tensor size to quantize (smaller tensors stay full precision)
    pub min_size: usize,
    /// Number of quantization levels (256 for int8-like behavior)
    pub num_levels: usize,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            skip_layers: vec![
                // Embeddings often benefit from staying in full precision
                "embed".to_string(),
                "lut".to_string(),
                // Final output projections
                "out_proj".to_string(),
                "eos_head".to_string(),
            ],
            min_size: 1024,  // Don't bother quantizing small tensors
            num_levels: 256, // int8-like
        }
    }
}

/// Quantized tensor wrapper that stores scale for dequantization
///
/// This uses simulated quantization - values are stored as f32 but discretized
/// to num_levels distinct values (256 for int8-equivalent).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (stored as f32 but with discrete values)
    pub data: Tensor,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: f32,
    /// Number of quantization levels used
    pub num_levels: usize,
}

impl QuantizedTensor {
    /// Quantize a tensor using symmetric per-tensor quantization
    ///
    /// This discretizes values to num_levels distinct values centered around 0,
    /// simulating int8 quantization behavior while using f32 storage.
    pub fn quantize(tensor: &Tensor, num_levels: usize) -> Result<Self> {
        // Convert to f32 if needed
        let tensor_f32 = tensor.to_dtype(DType::F32)?;

        // Find max absolute value for symmetric quantization
        let abs_max = tensor_f32.abs()?.max_all()?.to_scalar::<f32>()?;

        // Calculate scale (half the range for symmetric)
        let half_levels = (num_levels / 2) as f32;
        let scale = if abs_max > 0.0 {
            abs_max / (half_levels - 1.0)
        } else {
            1.0
        };

        // Quantize: q = round(x / scale), then dequantize back: x' = q * scale
        // This simulates quantization while staying in f32
        let scale_tensor = Tensor::new(&[scale], tensor.device())?;
        let quantized = tensor_f32.broadcast_div(&scale_tensor)?;
        let quantized = quantized.round()?;
        let clamped = quantized.clamp(-(half_levels - 1.0) as f64, (half_levels - 1.0) as f64)?;
        let data = clamped.broadcast_mul(&scale_tensor)?;

        Ok(Self {
            data,
            scale,
            zero_point: 0.0, // Symmetric quantization
            num_levels,
        })
    }

    /// Get the quantized tensor data
    pub fn data(&self) -> &Tensor {
        &self.data
    }

    /// Get the scale value
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get theoretical memory savings ratio compared to f32
    /// (In practice, data is still stored as f32, but this shows potential savings)
    pub fn theoretical_memory_savings(&self) -> f32 {
        match self.num_levels {
            256 => 4.0,   // int8 would be 4x smaller than f32
            65536 => 2.0, // int16 would be 2x smaller
            _ => 1.0,
        }
    }
}

/// Check if a layer name should skip quantization
fn should_skip_layer(name: &str, config: &QuantizeConfig) -> bool {
    config.skip_layers.iter().any(|skip| name.contains(skip))
}

/// Quantize a collection of weights according to config
///
/// Returns quantized weights. Layers in skip_layers or smaller than min_size
/// are returned unchanged.
pub fn quantize_weights(
    weights: &HashMap<String, Tensor>,
    config: &QuantizeConfig,
) -> Result<HashMap<String, QuantizedTensor>> {
    let mut quantized = HashMap::new();

    for (name, tensor) in weights {
        // Skip small tensors and excluded layers
        if tensor.elem_count() < config.min_size || should_skip_layer(name, config) {
            // Keep unquantized (scale=1, no discretization)
            quantized.insert(
                name.clone(),
                QuantizedTensor {
                    data: tensor.clone(),
                    scale: 1.0,
                    zero_point: 0.0,
                    num_levels: 0, // Indicates not actually quantized
                },
            );
        } else {
            quantized.insert(
                name.clone(),
                QuantizedTensor::quantize(tensor, config.num_levels)?,
            );
        }
    }

    Ok(quantized)
}

/// Calculate signal-to-noise ratio between original and quantized tensors
pub fn calculate_snr(original: &Tensor, quantized: &Tensor) -> Result<f32> {
    let original_f32 = original.to_dtype(DType::F32)?;
    let quantized_f32 = quantized.to_dtype(DType::F32)?;

    // SNR = 10 * log10(signal_power / noise_power)
    let signal_power = original_f32.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let noise = (&original_f32 - &quantized_f32)?;
    let noise_power = noise.sqr()?.mean_all()?.to_scalar::<f32>()?;

    if noise_power <= 0.0 {
        return Ok(f32::INFINITY); // Perfect reconstruction
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quantize_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, -3.0, 4.5, -2.1], &device).unwrap();

        let quantized = QuantizedTensor::quantize(&tensor, 256).unwrap();

        // Check SNR - should be good for int8-like quantization
        let snr = calculate_snr(&tensor, &quantized.data).unwrap();
        assert!(snr > 30.0, "SNR {} is too low", snr);
    }

    #[test]
    fn test_quantize_large_tensor() {
        let device = Device::Cpu;
        // Create a larger tensor with varying values
        let values: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.01).sin() * 10.0).collect();
        let tensor = Tensor::new(&values[..], &device).unwrap();

        let quantized = QuantizedTensor::quantize(&tensor, 256).unwrap();
        let snr = calculate_snr(&tensor, &quantized.data).unwrap();

        // For larger tensors with varied values, expect good SNR
        assert!(snr > 30.0, "SNR {} is too low", snr);
    }

    #[test]
    fn test_quantize_config_skip_layers() {
        let config = QuantizeConfig::default();
        assert!(should_skip_layer("model.embed_tokens", &config));
        assert!(should_skip_layer("decoder.out_proj", &config));
        assert!(!should_skip_layer("encoder.layers.0.linear", &config));
    }

    #[test]
    fn test_theoretical_savings() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let quantized = QuantizedTensor::quantize(&tensor, 256).unwrap();
        assert_eq!(quantized.theoretical_memory_savings(), 4.0);
    }
}
