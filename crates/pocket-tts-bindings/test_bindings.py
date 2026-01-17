import sys
import os
import time
import argparse
import wave
import struct

try:
    import pocket_tts_bindings
    print("✅ Successfully imported pocket_tts_bindings")
except ImportError:
    print("❌ Failed to import pocket_tts_bindings. Make sure you have built the bindings with 'maturin develop'.")
    sys.exit(1)

def save_wav(filename, samples, sample_rate=24000):
    """Save float samples to a 16-bit PCM WAV file."""
    # Scale to 16-bit integer range
    scaled = [max(-32768, min(32767, int(s * 32767))) for s in samples]
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('<' + 'h' * len(scaled), *scaled))
    print(f"💾 Saved audio to {filename}")

def resolve_voice(voice_spec):
    """Resolve voice specification to a local file path."""
    if os.path.exists(voice_spec):
        return os.path.abspath(voice_spec)
        
    PREDEFINED_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    
    if voice_spec.lower() in PREDEFINED_VOICES:
        print(f"Resolving predefined voice '{voice_spec}'...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="kyutai/pocket-tts-without-voice-cloning",
                filename=f"embeddings/{voice_spec.lower()}.safetensors"
            )
            print(f"   Downloaded to: {path}")
            return path
        except ImportError:
            print("❌ 'huggingface_hub' not installed. Cannot download predefined voices.")
            print("   Run: pip install huggingface-hub")
            return None
        except Exception as e:
            print(f"❌ Failed to download voice: {e}")
            return None
            
    # Try finding relative to script
    alt_path = os.path.join(os.path.dirname(__file__), voice_spec)
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)

    # Try finding in project root (common for ref.wav)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    root_path = os.path.join(project_root, voice_spec)
    if os.path.exists(root_path):
        return root_path

    print(f"❌ Voice file '{voice_spec}' not found.")
    return None

def test_generation(text, voice_spec, output_file, variant="b6369a24", model=None):
    voice_path = resolve_voice(voice_spec)
    if not voice_path:
        return

    print(f"Using voice reference: {voice_path}")
    
    if model is None:
        print(f"Loading model '{variant}'...")
        try:
            t0 = time.time()
            model = pocket_tts_bindings.PyTTSModel.load(variant)
            t1 = time.time()
            print(f"✅ Model loaded in {t1 - t0:.4f}s")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return model
    
    print(f"\nGenerating audio for: '{text}'")
    
    try:
        t0 = time.time()
        # Note: generate returns a list of floats
        audio = model.generate(text, voice_path)
        t1 = time.time()
        
        duration_sec = len(audio) / 24000.0 # Assuming 24khz
        print(f"✅ Generated {len(audio)} samples ({duration_sec:.2f}s audio) in {t1 - t0:.4f}s")
        print(f"⚡ Real-time factor: {(t1 - t0) / duration_sec:.4f}x (lower is better)")
        
        if output_file:
            save_wav(output_file, audio)
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pocket TTS Python Bindings")
    parser.add_argument("--text", type=str, default="This is a test of the Pocket TTS Rust bindings generated via PyO3.", help="Text to generate")
    parser.add_argument("--voice", type=str, default=None, help="Path to voice file OR predefined voice name. If not provided, runs a demo suite.")
    parser.add_argument("--output", type=str, default="test_output.wav", help="Output wav file")
    parser.add_argument("--variant", type=str, default="b6369a24", help="Model variant")
    
    args = parser.parse_args()
    
    if args.voice:
        test_generation(args.text, args.voice, args.output, args.variant)
    else:
        print("🔍 No voice specified. Running demo suite...")
        
        # Load model once to reuse
        print("\n=== Test 1: Predefined Voice (Alba) ===")
        model = test_generation(args.text, "alba", "output_alba.wav", args.variant)
        
        if model:
            print("\n=== Test 2: Reference WAV (ref.wav) ===")
            # Check for common ref.wav locations
            ref_candidates = ["ref.wav", "../../../ref.wav", "d:/pocket-tts-candle/ref.wav"]
            found_ref = False
            for cand in ref_candidates:
                if resolve_voice(cand):
                    test_generation(args.text, cand, "output_ref.wav", args.variant, model=model)
                    found_ref = True
                    break
            
            if not found_ref:
                print("⚠️  Could not find 'ref.wav' for second test. skipping.")
