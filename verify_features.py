
import time
import torch
import scipy.io.wavfile
import traceback
from pocket_tts import TTSModel

print("--- Starting Verification [Safe Mode] ---")

voice_prompt = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
text = "The quick brown fox jumps over the lazy dog."

# 1. Baseline Benchmark (Float32)
print("\n[1/3] Benchmarking Baseline (Float32)...")
try:
    tts = TTSModel.load_model() # Default float32
    print("Model loaded (Float32)")
    voice_state = tts.get_state_for_audio_prompt(voice_prompt)

    # Warmup
    _ = tts.generate_audio(voice_state, "Warmup")

    times = []
    for i in range(3):
        t0 = time.time()
        _ = tts.generate_audio(voice_state, text)
        duration = time.time() - t0
        times.append(duration)
        print(f"  Iteration {i+1}: {duration:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"Average generation time (Float32): {avg_time:.4f}s")

except Exception as e:
    print(f"FAILURE: Baseline benchmark failed: {e}")
    traceback.print_exc()

# 2. Verify Int8 Quantization
print("\n[2/3] Verifying Int8 Quantization...")
try:
    # Load with quantization enabled
    tts_int8 = TTSModel.load_model(quantize=True)
    print("Model loaded with quantize=True")

    # Real data generation
    voice_state_int8 = tts_int8.get_state_for_audio_prompt(voice_prompt)

    gen_start = time.time()
    audio = tts_int8.generate_audio(voice_state_int8, text)
    gen_duration = time.time() - gen_start

    output_path = "verify_int8.wav"
    scipy.io.wavfile.write(output_path, tts_int8.sample_rate, audio.numpy())
    print(f"SUCCESS: Generated {output_path} in {gen_duration:.2f}s")

except Exception as e:
    print(f"FAILURE: Int8 Quantization failed (Known Issue): {e}")
    # traceback.print_exc() # Suppress full trace to avoid cluttering final log

print("\n--- Verification Complete ---")
