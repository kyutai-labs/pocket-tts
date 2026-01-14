use anyhow::Result;
use candle_core::{Device, Tensor};
use hound::{WavReader, WavSpec, WavWriter};
use std::path::Path;

pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<(Tensor, u32)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };

    let tensor = if channels > 1 {
        // Interleaved to [channels, samples]
        let num_total_samples = samples.len();
        let num_samples = num_total_samples / channels;
        let mut reshaped = vec![0.0f32; num_total_samples];
        for c in 0..channels {
            for i in 0..num_samples {
                reshaped[c * num_samples + i] = samples[i * channels + c];
            }
        }
        Tensor::from_vec(reshaped, (channels, num_samples), &Device::Cpu)?
    } else {
        let n = samples.len();
        Tensor::from_vec(samples, (1, n), &Device::Cpu)?
    };

    Ok((tensor, sample_rate))
}

pub fn write_wav<P: AsRef<Path>>(path: P, audio: &Tensor, sample_rate: u32) -> Result<()> {
    let shape = audio.dims();
    if shape.len() != 2 {
        anyhow::bail!(
            "Expected audio tensor with shape [channels, samples], got {:?}",
            shape
        );
    }
    let channels = shape[0] as u16;
    let _num_samples = shape[1];

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    let data = audio.to_vec2::<f32>()?;

    // Interleave channels if more than 1
    if !data.is_empty() {
        for (i, _) in data[0].iter().enumerate() {
            for channel_data in &data {
                let val = (channel_data[i].clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(val)?;
            }
        }
    }
    writer.finalize()?;
    Ok(())
}

pub fn normalize_peak(audio: &Tensor) -> Result<Tensor> {
    let max_abs = audio.abs()?.max_all()?.to_scalar::<f32>()?;
    if max_abs > 0.0 {
        Ok(audio.affine(1.0 / max_abs as f64, 0.0)?)
    } else {
        Ok(audio.clone())
    }
}

// Simple linear interpolation resampler for basic porting
pub fn resample_linear(audio: &Tensor, from_rate: u32, to_rate: u32) -> Result<Tensor> {
    if from_rate == to_rate {
        return Ok(audio.clone());
    }

    let shape = audio.dims();
    let channels = shape[0];
    let num_samples = shape[1];
    let ratio = to_rate as f32 / from_rate as f32;
    let new_num_samples = (num_samples as f32 * ratio) as usize;

    let mut new_audio = Vec::with_capacity(channels * new_num_samples);
    let data = audio.to_vec2::<f32>()?;

    for channel_vec in &data {
        for i in 0..new_num_samples {
            let pos = i as f32 / ratio;
            let idx = pos as usize;
            let frac = pos - idx as f32;

            if idx + 1 < num_samples {
                let val = (1.0 - frac) * channel_vec[idx] + frac * channel_vec[idx + 1];
                new_audio.push(val);
            } else {
                new_audio.push(channel_vec[idx]);
            }
        }
    }

    Ok(Tensor::from_vec(
        new_audio,
        (channels, new_num_samples),
        audio.device(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_normalize_peak() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![-0.5f32, 0.2, 0.5], (1, 3), &device)?;
        let normalized = normalize_peak(&t)?;
        let data = normalized.to_vec2::<f32>()?;
        assert_eq!(data[0], vec![-1.0, 0.4, 1.0]);
        Ok(())
    }

    #[test]
    fn test_resample_linear() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device)?;
        // Resample 1Hz to 2Hz -> [1.0, 1.5, 2.0, 2.5, 3.0, 3.0] (approx)
        let resampled = resample_linear(&t, 1, 2)?;
        assert_eq!(resampled.dims()[1], 6);
        Ok(())
    }

    #[test]
    fn test_wav_io() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![0.0f32, 0.5, -0.5, 1.0], (1, 4), &device)?;
        let path = "test_io.wav";
        write_wav(path, &t, 16000)?;

        let (read_t, sr) = read_wav(path)?;
        assert_eq!(sr, 16000);
        assert_eq!(read_t.dims(), t.dims());

        // Small tolerance for 16-bit quantization
        let diff = (read_t - t)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4);

        std::fs::remove_file(path)?;
        Ok(())
    }
}
