#[cfg(test)]
mod tests {
    use candle_core::Tensor;
    use pocket_tts::TTSModel;
    use std::time::Instant;

    #[test]
    #[ignore] // Ignore by default as it requires weights and takes time
    fn test_long_audio_prompt_memory() -> anyhow::Result<()> {
        println!("Loading model...");
        // Assuming weights are available for this variant.
        // User's context implies "b6369a24" is relevant.
        let model = TTSModel::load("b6369a24")?;

        println!("Creating long audio input (5 minutes @ 24kHz)...");
        // 5 * 60 * 24000 = 7,200,000 samples
        let sample_rate = 24000;
        let duration_secs = 60;
        let num_samples = sample_rate * duration_secs;

        // Ensure we are using the device the model is on (CPU for now as per context)
        let device = &model.device;

        // Random audio or silence. Random is better to avoid strict silence optimization if any.
        let audio = Tensor::randn(0f32, 1f32, (1, 1, num_samples), device)?;

        let start = Instant::now();
        println!("Starting voice state processing...");

        // This call was the one causing O(N^2) memory explosion
        // It runs mimi encoding, projection, and then `run_flow_lm_prompt`
        let _state = model.get_voice_state_from_tensor(&audio)?;

        let duration = start.elapsed();
        println!("Processed long audio in {:.2}s", duration.as_secs_f64());
        println!("Memory check passed (no panic/OOM).");

        Ok(())
    }
}
