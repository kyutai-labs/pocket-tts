#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_core::Tensor;
    use pocket_tts::TTSModel;

    // Helper to compare tensors
    fn assert_tensors_close(t1: &Tensor, t2: &Tensor, tolerance: f64) -> Result<()> {
        let diff = (t1 - t2)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()? as f64;
        assert!(
            max_diff < tolerance,
            "Tensors differ by max {}. Tolerance: {}",
            max_diff,
            tolerance
        );
        Ok(())
    }

    #[test]
    fn test_streaming_matches_batch() -> Result<()> {
        // Load model (assumes weights are present from previous phases)
        let mut model = TTSModel::load("b6369a24")?;

        // Use deterministic settings for comparison
        model.temp = 0.0;

        // Get voice state (using ref.wav as standard)
        // Adjust path to point to project root ref.wav if running from crate dir
        let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let ref_wav = root_dir.join("assets").join("ref.wav");

        if !ref_wav.exists() {
            println!("Skipping test: ref.wav not found at {:?}", ref_wav);
            return Ok(());
        }

        let state = model.get_voice_state(&ref_wav)?;

        let text = "Hello world, this is a test.";

        // 1. Batch generation
        let batch_audio = model.generate(text, &state)?;

        // 2. Streaming generation
        let stream_chunks: Vec<Tensor> = model
            .generate_stream(text, &state)
            .collect::<Result<Vec<_>>>()?;

        assert!(
            !stream_chunks.is_empty(),
            "Stream should yield at least one chunk"
        );

        // Concatenate chunks
        let stream_audio = Tensor::cat(&stream_chunks, 2)?.squeeze(0)?;

        // 3. Compare
        // Note: Floating point non-associativity might cause slight differences
        // between batch (one big matmul mostly) and stream (step-by-step).
        // Using a slightly loose tolerance.
        assert_tensors_close(&batch_audio, &stream_audio, 1e-4)?;

        Ok(())
    }

    #[test]
    fn test_streaming_yields_multiple_chunks() -> Result<()> {
        let mut model = TTSModel::load("b6369a24")?;
        model.temp = 0.0;

        let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let ref_wav = root_dir.join("assets").join("ref.wav");

        if !ref_wav.exists() {
            return Ok(());
        }

        let state = model.get_voice_state(&ref_wav)?;

        // Use a long enough text to ensure multiple chunks
        let text = "This is a longer sentence to ensure that we get multiple chunks from the streaming generator.";

        let chunks: Vec<_> = model
            .generate_stream(text, &state)
            .collect::<Result<Vec<_>>>()?;

        println!("Generated {} chunks", chunks.len());
        assert!(
            chunks.len() > 1,
            "Should generate multiple chunks for long text"
        );

        // Verify chunk shapes (should be [1, 1, 1024] or similar depending on Mimi config)
        // First chunk might be different or last chunk might be padding, but generally checking correctness.
        for chunk in chunks.iter().take(chunks.len() - 1) {
            let dims = chunk.dims();
            assert_eq!(dims.len(), 3, "Chunk should be [B, C, T]");
            assert_eq!(dims[0], 1);
            assert_eq!(dims[1], 1);
            // In Mimi, one frame is 1024 samples (approx 80ms at 12.5Hz frame rate)?
            // Actually check expected size from config if strict, but just checking properties here.
        }

        Ok(())
    }

    #[test]
    fn test_long_text_handling() -> Result<()> {
        let mut model = TTSModel::load("b6369a24")?;
        model.temp = 0.0;

        let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let ref_wav = root_dir.join("assets").join("ref.wav");

        if !ref_wav.exists() {
            return Ok(());
        }

        let state = model.get_voice_state(&ref_wav)?;

        // Create a long text input (approx 50 sentences)
        let sentence = "This is a sentence that is repeated to simulate a long text input for the TTS model testing purposes.";
        let long_text = (0..50).map(|_| sentence).collect::<Vec<_>>().join(" ");

        // This effectively also tests memory usage if we monitor it, but here just checking it completes
        // and yields chunks without error.
        let mut chunks_count = 0;
        for chunk in model.generate_stream_long(&long_text, &state) {
            let _ = chunk?;
            chunks_count += 1;
        }

        assert!(
            chunks_count > 10,
            "Should generate many chunks for long text"
        );
        Ok(())
    }
}
