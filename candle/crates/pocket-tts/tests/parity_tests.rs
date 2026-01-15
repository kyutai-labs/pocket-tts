use candle_core::Tensor;
use pocket_tts::TTSModel;
use pocket_tts::audio::read_wav;
use pocket_tts::voice_state::init_states;
use std::path::PathBuf;

fn get_project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn get_ref_wav_path() -> PathBuf {
    get_project_root().join("ref_24k.wav")
}

fn assert_tensors_approx_eq(t1: &Tensor, t2: &Tensor, tolerance: f32) {
    let diff = (t1 - t2).expect("sub failed").abs().expect("abs failed");
    let max_diff = diff
        .max_all()
        .expect("max_all failed")
        .to_scalar::<f32>()
        .expect("to_scalar failed");

    let t1_max = t1
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let t2_max = t2
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    println!(
        "Max Diff: {}, T1 Max: {}, T2 Max: {}",
        max_diff, t1_max, t2_max
    );

    assert!(
        max_diff < tolerance,
        "Tensors differ by max {}, which is > tolerance {}",
        max_diff,
        tolerance
    );
}

#[test]
fn test_voice_conditioning_parity() {
    let root = get_project_root();
    let ref_path = root.join("ref_voice_conditioning.safetensors");
    if !ref_path.exists() {
        eprintln!("Skipping parity test: {:?} not found", ref_path);
        return;
    }

    let model = TTSModel::load("b6369a24").expect("Failed to load model");
    let (audio, sr) = read_wav(&get_ref_wav_path()).expect("Failed to read ref.wav");

    let audio = if sr != model.sample_rate as u32 {
        pocket_tts::audio::resample_linear(&audio, sr, model.sample_rate as u32)
            .expect("Failed to resample")
    } else {
        audio
    };
    let audio = audio.unsqueeze(0).expect("unsqueeze failed");

    // Pad audio
    let frame_size = model.mimi.frame_size();
    let (b, c, t) = audio.dims3().expect("dims3");
    let pad_len = if t % frame_size != 0 {
        frame_size - (t % frame_size)
    } else {
        0
    };
    let audio = if pad_len > 0 {
        let pad =
            candle_core::Tensor::zeros((b, c, pad_len), audio.dtype(), audio.device()).unwrap();
        candle_core::Tensor::cat(&[&audio, &pad], 2).unwrap()
    } else {
        audio
    };

    let mut state = init_states(1, 1000);
    let latents = model
        .mimi
        .encode_to_latent(&audio, &mut state)
        .expect("encode failed");

    // Calculate conditioning: latents * speaker_proj.t()
    // Rust: latents is [B, C, T] ? Wait, Mimi output is [B, D, T] (32 dim) -> No, now 512 dim!

    let latents_t = latents.permute((0, 2, 1)).expect("permute failed");
    // latents_t: [B, T, 512]

    let (b, t, d) = latents_t.dims3().expect("dims3");
    let latents_flat = latents_t.reshape((b * t, d)).expect("reshape failed");
    let speaker_proj_t = model.speaker_proj_weight.t().expect("transpose failed");

    let conditioning_flat = latents_flat.matmul(&speaker_proj_t).expect("matmul failed");
    let conditioning = conditioning_flat
        .reshape((b, t, 1024))
        .expect("reshape back failed");

    // Load ref
    let tensors = candle_core::safetensors::load(ref_path, &candle_core::Device::Cpu)
        .expect("failed to load ref tensors");
    let ref_cond = tensors
        .get("voice_conditioning")
        .expect("voice_conditioning not found");

    // Python output might have different length due to padding?
    // ref_cond same length as latents?
    assert_eq!(
        conditioning.dims(),
        ref_cond.dims(),
        "Conditioning shape mismatch"
    );

    assert_tensors_approx_eq(&conditioning, ref_cond, 2e-2);
}

#[test]
fn test_mimi_latents_parity() {
    let root = get_project_root();
    let ref_path = root.join("ref_mimi_latents.safetensors");
    if !ref_path.exists() {
        eprintln!("Skipping parity test: {:?} not found", ref_path);
        return;
    }

    let model = TTSModel::load("b6369a24").expect("Failed to load model");

    // Check initial layers weight stats
    {
        println!("Rust Mimi Encoder weight stats:");
        let vb = &model.vb;

        let layers = [
            ("mimi.encoder.model.0.conv", vec![64, 1, 7], true),
            ("mimi.encoder.model.1.block.1.conv", vec![32, 64, 3], true),
            ("mimi.encoder.model.1.block.3.conv", vec![64, 32, 1], true),
        ];

        for (name, shape, has_bias) in layers {
            let vb_l = vb.pp(name);
            let w = vb_l
                .get(shape.clone(), "weight")
                .expect(&format!("failed to get {}", name));
            println!(
                "{}.weight: shape={:?}, mean={:.6}, min={:.6}, max={:.6}, abs_max={:.6}",
                name,
                w.dims(),
                w.mean_all().unwrap().to_scalar::<f32>().unwrap(),
                w.min_all().unwrap().to_scalar::<f32>().unwrap(),
                w.max_all().unwrap().to_scalar::<f32>().unwrap(),
                w.abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            );

            if has_bias {
                let b_shape = shape[0];
                let b = vb_l
                    .get(b_shape, "bias")
                    .expect(&format!("failed to get {} bias", name));
                println!(
                    "{}.bias: shape={:?}, mean={:.6}, min={:.6}, max={:.6}, abs_max={:.6}",
                    name,
                    b.dims(),
                    b.mean_all().unwrap().to_scalar::<f32>().unwrap(),
                    b.min_all().unwrap().to_scalar::<f32>().unwrap(),
                    b.max_all().unwrap().to_scalar::<f32>().unwrap(),
                    b.abs()
                        .unwrap()
                        .max_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                );
            }
        }
    }

    let (audio, sr) = read_wav(&get_ref_wav_path()).expect("Failed to read ref.wav");
    // Preprocessing as in extract_refs.py (convert_audio logic)
    // The python script does: convert_audio(wav, sr, 24000, 1) which resamples and mixes to mono
    // Our read_wav returns (C, T)
    let audio = if sr != model.sample_rate as u32 {
        pocket_tts::audio::resample_linear(&audio, sr, model.sample_rate as u32)
            .expect("Failed to resample")
    } else {
        audio
    };
    // Ensure mono (if stereo, mean) - read_wav might return stereo
    // But let's assume ref.wav is mono or handled

    // Add batch dim: [1, C, T]
    let audio = audio.unsqueeze(0).expect("unsqueeze failed");

    // Encode with Mimi
    // Note: Rust definition of encode_to_latent takes (x, state)
    // But python script calls model.mimi.encode_to_latent(mimi_input)
    // We must ensure we use the same state initialization or stateless if applicable.
    // streaming convs need state.
    // Pad audio to a multiple of frame size
    let frame_size = model.mimi.frame_size();
    let (b, c, t) = audio.dims3().expect("dims3");
    let pad_len = if t % frame_size != 0 {
        frame_size - (t % frame_size)
    } else {
        0
    };
    let audio = if pad_len > 0 {
        let pad =
            candle_core::Tensor::zeros((b, c, pad_len), audio.dtype(), audio.device()).unwrap();
        candle_core::Tensor::cat(&[&audio, &pad], 2).unwrap()
    } else {
        audio
    };

    let mut state = init_states(1, 1000);
    let latents = model
        .mimi
        .encode_to_latent(&audio, &mut state)
        .expect("encode failed");

    // Load ref
    let tensors = candle_core::safetensors::load(ref_path, &candle_core::Device::Cpu)
        .expect("failed to load ref tensors");
    let ref_latents = tensors.get("mimi_latents").expect("mimi_latents not found");

    // Compare intermediates
    let mut state = init_states(1, 1000);

    // Layer 0: Conv
    let layer0_out = if let Some(layer0_ref) = tensors.get("layer0_out") {
        let out = model.mimi.encoder.layers[0]
            .forward(&audio, &mut state)
            .expect("layer0 failed");
        println!(
            "Layer0 Parity check... Rust: {:?}, Ref: {:?}",
            out.dims(),
            layer0_ref.dims()
        );
        assert_tensors_approx_eq(&out, layer0_ref, 2e-2);
        out
    } else {
        model.mimi.encoder.layers[0]
            .forward(&audio, &mut state)
            .expect("layer0 failed")
    };

    // Layer 1: Resnet
    let layer1_out = if let Some(layer1_ref) = tensors.get("layer1_out") {
        let out = model.mimi.encoder.layers[1]
            .forward(&layer0_out, &mut state)
            .expect("layer1 failed");
        println!(
            "Layer1 (Resnet) Parity check... Rust: {:?}, Ref: {:?}",
            out.dims(),
            layer1_ref.dims()
        );
        assert_tensors_approx_eq(&out, layer1_ref, 2e-2);
        out
    } else {
        model.mimi.encoder.layers[1]
            .forward(&layer0_out, &mut state)
            .expect("layer1 failed")
    };

    // Layer 2: ELU
    let layer2_out = if let Some(layer2_ref) = tensors.get("layer2_out") {
        let out = model.mimi.encoder.layers[2]
            .forward(&layer1_out, &mut state)
            .expect("layer2 failed");
        println!(
            "Layer2 (ELU) Parity check... Rust: {:?}, Ref: {:?}",
            out.dims(),
            layer2_ref.dims()
        );
        assert_tensors_approx_eq(&out, layer2_ref, 2e-2);
        out
    } else {
        model.mimi.encoder.layers[2]
            .forward(&layer1_out, &mut state)
            .expect("layer2 failed")
    };

    // Layer 3: Downsample (ratio 4)
    let layer3_out = if let Some(layer3_ref) = tensors.get("layer3_out") {
        let out = model.mimi.encoder.layers[3]
            .forward(&layer2_out, &mut state)
            .expect("layer3 failed");
        println!(
            "Layer3 (Downsample) Parity check... Rust: {:?}, Ref: {:?}",
            out.dims(),
            layer3_ref.dims()
        );
        assert_tensors_approx_eq(&out, layer3_ref, 2e-2);
        out
    } else {
        model.mimi.encoder.layers[3]
            .forward(&layer2_out, &mut state)
            .expect("layer3 failed")
    };

    if let Some(seanet_ref) = tensors.get("seanet_out") {
        // Run the full encoder to get seanet_out
        let mut state = init_states(1, 1000);
        let seanet_out = model
            .mimi
            .encoder
            .forward(&audio, &mut state)
            .expect("seanet failed");
        println!(
            "SEANet Parity check... Rust: {:?}, Ref: {:?}",
            seanet_out.dims(),
            seanet_ref.dims()
        );
        assert_tensors_approx_eq(&seanet_out, seanet_ref, 2e-2);
    }

    if let Some(tr_ref) = tensors.get("transformer_out") {
        let mut state = init_states(1, 1000);
        let seanet_out = model
            .mimi
            .encoder
            .forward(&audio, &mut state)
            .expect("seanet failed");
        let mut tr_state = init_states(1, 1000);
        let mut embs = model
            .mimi
            .encoder_transformer
            .forward(&seanet_out, &mut tr_state)
            .expect("tr failed");
        let tr_out = embs.remove(0);
        println!(
            "Transformer Parity check... Rust: {:?}, Ref: {:?}",
            tr_out.dims(),
            tr_ref.dims()
        );
        assert_tensors_approx_eq(&tr_out, tr_ref, 2e-2);
    }

    assert_tensors_approx_eq(&latents, ref_latents, 0.1);
}

#[test]
fn test_input_parity() {
    let root = get_project_root();
    let ref_path = root.join("ref_mimi_input.safetensors");
    if !ref_path.exists() {
        eprintln!("Skipping parity test: {:?} not found", ref_path);
        return;
    }

    let model = TTSModel::load("b6369a24").expect("Failed to load model");
    let (audio, sr) = read_wav(&get_ref_wav_path()).expect("Failed to read ref.wav");

    // Rust preprocessing
    let audio = if sr != model.sample_rate as u32 {
        pocket_tts::audio::resample_linear(&audio, sr, model.sample_rate as u32)
            .expect("Failed to resample")
    } else {
        audio
    };
    // Ensure mono/batch as in parity tests
    // Python mimi_input was: wav.unsqueeze(0) -> [1, C, T] ?
    // In extract_refs.py:
    // audio, sr = audio_read(...) -> [C, T]
    // wav = convert_audio(...) -> [C, T] (mono)
    // mimi_input = wav.unsqueeze(0) -> [1, C, T]

    // Rust audio is [C, T] after read (mono=1)
    let b = 1;
    let c = 1;
    let x = audio.unsqueeze(0).expect("unsqueeze failed");
    let t = x.dims()[2];
    let hop = model.mimi.frame_size();
    let audio = if t % hop != 0 {
        let padding = hop - (t % hop);
        let pad = Tensor::zeros((b, c, padding), x.dtype(), x.device()).expect("pad zeros failed");
        Tensor::cat(&[x, pad], 2).expect("pad cat failed")
    } else {
        x
    };

    // Load ref
    let tensors = candle_core::safetensors::load(ref_path, &candle_core::Device::Cpu)
        .expect("failed to load ref tensors");
    let ref_input = tensors.get("mimi_input").expect("mimi_input not found");

    // We expect ref_input to be [1, 1, T]
    // And audio to be [1, 1, T]
    println!(
        "Input parity check... Rust: {:?}, Ref: {:?}",
        audio.dims(),
        ref_input.dims()
    );
    assert_eq!(audio.dims(), ref_input.dims(), "Input shape mismatch");

    // Check parity
    assert_tensors_approx_eq(&audio, ref_input, 2e-2);
}

#[test]
fn test_audio_generation_parity() {
    let root = get_project_root();
    let ref_path = root.join("ref_output.wav");
    if !ref_path.exists() {
        eprintln!("Skipping parity test: {:?} not found", ref_path);
        return;
    }

    // Load Rust model with temperature = 0.0 for deterministic output
    let model = TTSModel::load_with_params(
        "b6369a24",
        0.0, // temp
        pocket_tts::config::defaults::LSD_DECODE_STEPS,
        pocket_tts::config::defaults::EOS_THRESHOLD,
    )
    .expect("Failed to load model");

    let voice_state = model
        .get_voice_state(&get_ref_wav_path())
        .expect("get_voice_state failed");

    // Rust generation
    let text = "Hello world.";
    let generated = model
        .generate(text, &voice_state)
        .expect("generation failed");

    // Load ref output for comparison
    let (ref_audio, _) = read_wav(&ref_path).expect("Failed to read ref_output.wav");

    println!(
        "Generated dims: {:?}, Ref dims: {:?}",
        generated.dims(),
        ref_audio.dims()
    );

    // Save Rust output for manual listening
    let output_path = root.join("test_output_parity.wav");
    pocket_tts::audio::write_wav(&output_path, &generated, model.sample_rate as u32)
        .expect("failed to write output");
    println!("Saved Rust output to {:?}", output_path);

    // Basic sanity checks
    let gen_samples = generated.dims()[1];
    let ref_samples = ref_audio.dims()[1];

    // Audio should be reasonably close in length (within 50%)
    let length_ratio = gen_samples as f64 / ref_samples as f64;
    assert!(
        length_ratio > 0.5 && length_ratio < 1.5,
        "Audio length significantly different: {} vs {} samples (ratio: {:.2})",
        gen_samples,
        ref_samples,
        length_ratio
    );

    // Audio should have reasonable amplitude (not silent, not clipping)
    let gen_max = generated
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        gen_max > 0.01,
        "Generated audio appears silent (max: {})",
        gen_max
    );
    assert!(
        gen_max < 2.0,
        "Generated audio appears clipped (max: {})",
        gen_max
    );

    println!(
        "Audio generation parity test passed (manual listening required for quality verification)"
    );
    println!(
        "Listen to: {:?} and compare with {:?}",
        output_path, ref_path
    );
}

#[test]
fn test_decoder_parity() {
    let root = get_project_root();
    let ref_path = root.join("ref_decoder_intermediates.safetensors");
    if !ref_path.exists() {
        eprintln!("Skipping decoder parity test: {:?} not found", ref_path);
        eprintln!("Run: uv run python extract_decoder_refs.py");
        return;
    }

    // Load reference intermediates
    let tensors = candle_core::safetensors::load(&ref_path, &candle_core::Device::Cpu)
        .expect("Failed to load ref_decoder_intermediates.safetensors");

    let ref_quantized = tensors.get("quantized").expect("missing quantized");
    let ref_after_upsample = tensors
        .get("after_upsample")
        .expect("missing after_upsample");
    let ref_after_decoder_tr = tensors
        .get("after_decoder_transformer")
        .expect("missing after_decoder_transformer");
    let ref_final_audio = tensors.get("final_audio").expect("missing final_audio");

    println!("Loaded reference decoder intermediates:");
    println!("  quantized: {:?}", ref_quantized.dims());
    println!("  after_upsample: {:?}", ref_after_upsample.dims());
    println!(
        "  after_decoder_transformer: {:?}",
        ref_after_decoder_tr.dims()
    );
    println!("  final_audio: {:?}", ref_final_audio.dims());

    // Load model
    let model = TTSModel::load_with_params(
        "b6369a24",
        0.0,
        pocket_tts::config::defaults::LSD_DECODE_STEPS,
        pocket_tts::config::defaults::EOS_THRESHOLD,
    )
    .expect("Failed to load model");

    // Use the same quantized input as Python (to isolate decoder issues)
    let mut mimi_state = init_states(1, 1000);

    // Test upsample
    let after_upsample = if let Some(ref up) = model.mimi.upsample {
        let out = up
            .forward(ref_quantized, &mut mimi_state)
            .expect("upsample failed");
        println!(
            "\nUpsample Rust: {:?}, Ref: {:?}",
            out.dims(),
            ref_after_upsample.dims()
        );
        assert_tensors_approx_eq(&out, ref_after_upsample, 0.05);
        println!("✓ Upsample PASSED");
        out
    } else {
        ref_quantized.clone()
    };

    // Test decoder transformer
    let mut after_decoder_tr_vec = model
        .mimi
        .decoder_transformer
        .forward(&after_upsample, &mut mimi_state)
        .expect("decoder_transformer failed");
    let after_decoder_tr = after_decoder_tr_vec.remove(0);

    println!(
        "\nDecoder Transformer Rust: {:?}, Ref: {:?}",
        after_decoder_tr.dims(),
        ref_after_decoder_tr.dims()
    );
    assert_tensors_approx_eq(&after_decoder_tr, ref_after_decoder_tr, 0.05);
    println!("✓ Decoder Transformer PASSED");

    // Test SEANet decoder
    let final_audio = model
        .mimi
        .decoder
        .forward(&after_decoder_tr, &mut mimi_state)
        .expect("decoder failed");

    println!(
        "\nFinal Audio Rust: {:?}, Ref: {:?}",
        final_audio.dims(),
        ref_final_audio.dims()
    );
    assert_tensors_approx_eq(&final_audio, ref_final_audio, 0.1);
    println!("✓ SEANet Decoder PASSED");

    println!("\n=== ALL DECODER STAGES PASS ===");
}
