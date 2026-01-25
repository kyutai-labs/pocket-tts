# Pocket TTS Troubleshooting Guide

This guide covers common issues, debugging techniques, and solutions for Pocket TTS.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Problems](#model-loading-problems)
- [Audio Generation Issues](#audio-generation-issues)
- [Performance Problems](#performance-problems)
- [Voice Cloning Issues](#voice-cloning-issues)
- [Audio Quality Problems](#audio-quality-problems)
- [Memory and Resource Issues](#memory-and-resource-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Debugging Techniques](#debugging-techniques)

## Installation Issues

### Problem: Package Installation Fails

**Symptoms:**
```bash
pip install pocket-tts
# ERROR: Could not install packages due to EnvironmentError
```

**Solutions:**

1. **Use uv for faster installation:**
```bash
uv add pocket-tts
# or
uvx pocket-tts generate
```

2. **Check Python version compatibility:**
```bash
python --version  # Should be 3.10+
```

3. **Install in clean environment:**
```bash
python -m venv pocket_tts_env
source pocket_tts_env/bin/activate  # On Windows: pocket_tts_env\Scripts\activate
pip install --upgrade pip
pip install pocket-tts
```

4. **Install PyTorch CPU version first:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pocket-tts
```

### Problem: Missing Dependencies

**Symptoms:**
```
ImportError: No module named 'torch'
ImportError: No module named 'safetensors'
```

**Solutions:**

1. **Install missing dependencies manually:**
```bash
pip install torch safetensors beartype
```

2. **Install with all extras:**
```bash
pip install "pocket-tts[all]"
```

3. **Check installed packages:**
```bash
pip list | grep -E "(torch|safetensors|beartype)"
```

## Model Loading Problems

### Problem: Model Download Fails

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'model.safetensors'
RuntimeError: Failed to download model weights
```

**Solutions:**

1. **Check internet connection and Hugging Face authentication:**
```bash
uvx hf auth login
# Follow the instructions to authenticate with Hugging Face
```

2. **Manually download model files:**
```bash
# Download from Hugging Face
wget https://huggingface.co/kyutai/pocket-tts/resolve/main/tts_b6369a24.safetensors
# Place in appropriate directory
mkdir -p ~/.cache/huggingface/hub/models--kyutai--pocket-tts
mv tts_b6369a24.safetensors ~/.cache/huggingface/hub/models--kyutai--pocket-tts/
```

3. **Use alternative model variant:**
```python
from pocket_tts import TTSModel

# Try different model variants
try:
    model = TTSModel.load_model("b6369a24")
except Exception:
    try:
        model = TTSModel.load_model("alternative_variant")
    except Exception:
        print("No model variants available")
```

### Problem: Voice Cloning Not Available

**Symptoms:**
```
ValueError: We could not download the weights for the model with voice cloning
```

**Solutions:**

1. **Accept Hugging Face terms:**
   - Visit https://huggingface.co/kyutai/pocket-tts
   - Accept the terms and conditions
   - Authenticate locally: `uvx hf auth login`

2. **Use predefined voices only:**
```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
# Use only predefined voices
voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
for voice in voices:
    try:
        voice_state = model.get_state_for_audio_prompt(voice)
        print(f"Successfully loaded voice: {voice}")
    except Exception as e:
        print(f"Failed to load voice {voice}: {e}")
```

## Audio Generation Issues

### Problem: Generation Takes Too Long

**Symptoms:**
- Generation takes minutes instead of seconds
- Real-time factor < 1.0

**Solutions:**

1. **Check CPU usage and optimize:**
```python
import torch
import psutil

# Check available CPU cores
print(f"Available CPU cores: {psutil.cpu_count()}")
print(f"PyTorch threads: {torch.get_num_threads()}")

# Optimize for CPU
torch.set_num_threads(min(4, psutil.cpu_count()))  # Use 4 threads max
```

2. **Use streaming for long texts:**
```python
# Instead of generate_audio(), use generate_audio_stream()
for chunk in model.generate_audio_stream(voice_state, long_text):
    # Process chunks immediately
    pass
```

3. **Reduce generation parameters:**
```python
# Use faster generation settings
model = TTSModel.load_model(
    temp=0.7,              # Lower temperature = faster
    lsd_decode_steps=4,     # Fewer steps = faster
    noise_clamp=1.0         # Enable clamping
)
```

### Problem: Audio Quality Issues

**Symptoms:**
- Audio sounds robotic or distorted
- Background noise or artifacts
- Inconsistent volume

**Solutions:**

1. **Adjust generation parameters:**
```python
# Higher quality settings
model = TTSModel.load_model(
    temp=0.8,              # Slightly higher for variety
    lsd_decode_steps=8,     # More steps for quality
    noise_clamp=None        # No clamping for natural sound
)
```

2. **Post-process audio:**
```python
from pocket_tts import normalize_audio, apply_fade
import numpy as np

# Generate audio
audio = model.generate_audio(voice_state, text)
audio_np = audio.numpy()

# Improve quality
audio_np = normalize_audio(audio_np, gain=1.1)  # Normalize and boost
audio_np = apply_fade(audio_np, fade_in_ms=10, fade_out_ms=10,
                      sample_rate=model.sample_rate)

# Convert back to tensor
audio_enhanced = torch.from_numpy(audio_np)
```

3. **Check input text formatting:**
```python
def clean_text(text):
    """Clean text for better generation."""
    text = text.strip()
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Ensure proper punctuation
    if not text.endswith(('.', '!', '?', ':', ';')):
        text += '.'
    return text

cleaned_text = clean_text(original_text)
audio = model.generate_audio(voice_state, cleaned_text)
```

## Performance Problems

### Problem: High Memory Usage

**Symptoms:**
- Out of memory errors
- System becomes slow
- Memory usage > 1GB

**Solutions:**

1. **Monitor and optimize memory:**
```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

# Monitor before and after generation
monitor_memory()
audio = model.generate_audio(voice_state, text)
monitor_memory()

# Clean up
del audio
gc.collect()
monitor_memory()
```

2. **Use streaming for long texts:**
```python
# Process in chunks to reduce memory
def generate_long_text_streaming(model, voice_state, text, chunk_size=500):
    """Generate long text in chunks to save memory."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk_text = ' '.join(words[i:i+chunk_size])
        for audio_chunk in model.generate_audio_stream(voice_state, chunk_text):
            yield audio_chunk
        # Clear intermediate results
        gc.collect()
```

3. **Optimize model state:**
```python
# Use voice state truncation for long audio prompts
voice_state = model.get_state_for_audio_prompt(
    audio_file,
    truncate=True  # Truncate to 30 seconds to save memory
)
```

### Problem: Slow First Generation

**Symptoms:**
- First generation takes much longer than subsequent ones
- Cold start latency > 10 seconds

**Solutions:**

1. **Warm up the model:**
```python
def warm_up_model(model, voice_state):
    """Warm up model with short generation."""
    try:
        # Generate very short text to initialize everything
        _ = model.generate_audio(voice_state, "Hi.", max_tokens=10)
        print("Model warmed up")
    except Exception as e:
        print(f"Warm up failed: {e}")

# Warm up before real usage
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")
warm_up_model(model, voice_state)
```

2. **Pre-load voice states:**
```python
# Cache commonly used voice states
voice_cache = {}
def get_voice_state(voice_name):
    if voice_name not in voice_cache:
        voice_cache[voice_name] = model.get_state_for_audio_prompt(voice_name)
    return voice_cache[voice_name]
```

## Voice Cloning Issues

### Problem: Poor Voice Cloning Quality

**Symptoms:**
- Cloned voice sounds different from original
- Audio artifacts or distortion
- Voice characteristics not preserved

**Solutions:**

1. **Check reference audio quality:**
```python
from pocket_tts import load_wav, compute_audio_metrics

def analyze_reference_audio(audio_path):
    """Analyze reference audio for quality."""
    audio, sr = load_wav(audio_path)
    metrics = compute_audio_metrics(audio.numpy())

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {audio.shape[1] / sr:.2f} seconds")
    print(f"RMS level: {metrics['rms']:.3f}")
    print(f"Peak level: {metrics['peak']:.3f}")
    print(f"Dynamic range: {metrics['dynamic_range_db']:.1f} dB")

    # Quality checks
    if sr != 24000:
        print("⚠️  Non-standard sample rate - will be resampled")
    if audio.shape[1] < 8000:  # Less than 0.33 seconds
        print("⚠️  Very short audio - may not capture voice characteristics")
    if metrics['rms'] < 0.1:
        print("⚠️  Very quiet audio - may affect cloning quality")
    if metrics['dynamic_range_db'] < 10:
        print("⚠️  Low dynamic range - may affect voice quality")

analyze_reference_audio("reference.wav")
```

2. **Use optimal reference audio:**
```python
# Guidelines for good reference audio:
def is_good_reference(audio_path):
    """Check if reference audio meets quality guidelines."""
    audio, sr = load_wav(audio_path)
    duration = audio.shape[1] / sr
    metrics = compute_audio_metrics(audio.numpy())

    checks = {
        "duration": 2.0 <= duration <= 30.0,  # 2-30 seconds ideal
        "sample_rate": sr >= 16000,  # At least 16kHz
        "rms_level": 0.1 <= metrics['rms'] <= 0.9,  # Good volume
        "dynamic_range": metrics['dynamic_range_db'] >= 15,  # Good dynamics
        "mono": audio.shape[0] == 1  # Mono audio
    }

    return all(checks.values()), checks

# Test reference audio
good, checks = is_good_reference("reference.wav")
if good:
    print("✅ Reference audio quality is good")
else:
    print("❌ Reference audio has issues:")
    for check, passed in checks.items():
        if not passed:
            print(f"  - {check}: Failed")
```

3. **Improve reference audio:**
```python
from pocket_tts import normalize_audio, apply_fade

def improve_reference_audio(audio_path, output_path):
    """Improve reference audio quality."""
    audio, sr = load_wav(audio_path)
    audio_np = audio.numpy()

    # Improve quality
    audio_np = normalize_audio(audio_np, gain=1.2)  # Normalize and boost
    audio_np = apply_fade(audio_np, fade_in_ms=50, fade_out_ms=50, sample_rate=sr)

    # Save improved version
    improved_audio = torch.from_numpy(audio_np)
    save_audio(output_path, improved_audio, sr)
    print(f"Improved reference audio saved to {output_path}")

improve_reference_audio("reference.wav", "reference_improved.wav")
```

## Audio Quality Problems

### Problem: Audio Contains Artifacts

**Symptoms:**
- Clicks, pops, or crackling sounds
- Digital distortion or clipping
- Background noise or hiss

**Solutions:**

1. **Check for clipping:**
```python
def check_clipping(audio):
    """Check if audio is clipped."""
    audio_np = audio.numpy()
    max_val = np.abs(audio_np).max()
    clipped_samples = np.sum(np.abs(audio_np) > 0.99)

    print(f"Peak level: {max_val:.3f}")
    print(f"Clipped samples: {clipped_samples}")

    if max_val >= 1.0:
        print("⚠️  Audio is clipped - reduce gain")
    if clipped_samples > 0:
        print("⚠️  Some samples are clipped")

    return max_val < 0.99 and clipped_samples == 0

# Check generated audio
audio = model.generate_audio(voice_state, text)
is_ok = check_clipping(audio)
```

2. **Apply audio processing:**
```python
def remove_artifacts(audio, sample_rate):
    """Remove common audio artifacts."""
    audio_np = audio.numpy()

    # Normalize to prevent clipping
    audio_np = normalize_audio(audio_np, gain=0.9)

    # Apply gentle fade to reduce clicks
    audio_np = apply_fade(audio_np, fade_in_ms=5, fade_out_ms=5,
                         sample_rate=sample_rate)

    return torch.from_numpy(audio_np)

# Clean up audio
clean_audio = remove_artifacts(audio, model.sample_rate)
```

3. **Adjust generation parameters:**
```python
# Reduce artifacts with better parameters
model = TTSModel.load_model(
    temp=0.7,              # Lower temperature reduces randomness
    lsd_decode_steps=6,    # Moderate steps for quality
    noise_clamp=1.0         # Clamp to prevent extreme values
)
```

## Memory and Resource Issues

### Problem: System Resources Exhausted

**Symptoms:**
- System becomes unresponsive
- Other applications crash
- CPU usage stays at 100%

**Solutions:**

1. **Limit concurrent generations:**
```python
import threading
import queue

class ThreadSafeTTS:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.generation_queue = queue.Queue(maxsize=3)  # Limit queue size

    def generate_audio(self, voice_state, text):
        with self.lock:
            return self.model.generate_audio(voice_state, text)

# Use thread-safe wrapper
safe_tts = ThreadSafeTTS(model)
```

2. **Monitor resource usage:**
```python
import psutil
import time

def monitor_resources(duration=60):
    """Monitor system resources during generation."""
    start_time = time.time()

    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        print(f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")

        if cpu_percent > 95:
            print("⚠️  High CPU usage detected")
        if memory.percent > 90:
            print("⚠️  High memory usage detected")

        time.sleep(5)

# Monitor in background
import threading
monitor_thread = threading.Thread(target=monitor_resources, args=(30,))
monitor_thread.daemon = True
monitor_thread.start()
```

3. **Optimize batch processing:**
```python
def batch_generate(texts, batch_size=5):
    """Generate multiple texts with resource management."""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Process batch
        batch_results = []
        for text in batch:
            try:
                audio = model.generate_audio(voice_state, text)
                batch_results.append(audio)
            except Exception as e:
                print(f"Failed to generate '{text[:30]}...': {e}")
                batch_results.append(None)

        results.extend(batch_results)

        # Clean up between batches
        gc.collect()
        time.sleep(1)  # Brief pause to let system recover

    return results
```

## Platform-Specific Issues

### Windows Issues

**Problem: DLL loading errors**

```bash
# Solution: Install Visual C++ Redistributable
# Download from Microsoft website or use conda
conda install vs2019_win-64

# Or use torch with CPU-only builds
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Problem: Path issues**

```python
# Use raw strings for Windows paths
audio_path = r"C:\Users\username\audio.wav"
# or use pathlib
from pathlib import Path
audio_path = Path("C:/Users/username/audio.wav")
```

### macOS Issues

**Problem: PyTorch installation**

```bash
# Use conda on macOS for better compatibility
conda install pytorch torchvision torchaudio -c pytorch
pip install pocket-tts
```

**Problem: Audio permissions**

```bash
# Grant microphone access if needed
# Check System Preferences > Security & Privacy > Privacy > Microphone
```

### Linux Issues

**Problem: Missing system libraries**

```bash
# Install audio libraries
sudo apt-get install libsndfile1 libsndfile1-dev  # Ubuntu/Debian
sudo yum install libsndfile-devel                  # CentOS/RHEL
```

**Problem: Threading issues**

```python
# Limit thread usage on Linux
import torch
import os

# Limit PyTorch threads
torch.set_num_threads(4)

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
```

## Debugging Techniques

### Enable Verbose Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Specific module logging
logging.getLogger('pocket_tts').setLevel(logging.DEBUG)
```

### Profile Generation Performance

```python
import cProfile
import pstats

def profile_generation(model, voice_state, text):
    """Profile TTS generation performance."""
    profiler = cProfile.Profile()
    profiler.enable()

    audio = model.generate_audio(voice_state, text)

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return audio

# Profile generation
audio = profile_generation(model, voice_state, "Hello world!")
```

### Debug Voice Cloning

```python
def debug_voice_cloning(audio_path):
    """Debug voice cloning process step by step."""
    print(f"Debugging voice cloning for: {audio_path}")

    try:
        # Step 1: Load audio
        print("1. Loading audio...")
        audio, sr = load_wav(audio_path)
        print(f"   Loaded: {audio.shape}, {sr} Hz")

        # Step 2: Check audio quality
        print("2. Analyzing audio quality...")
        metrics = compute_audio_metrics(audio.numpy())
        for key, value in metrics.items():
            print(f"   {key}: {value}")

        # Step 3: Create voice state
        print("3. Creating voice state...")
        voice_state = model.get_state_for_audio_prompt(audio_path)
        print("   Voice state created successfully")

        # Step 4: Test generation
        print("4. Testing generation...")
        test_audio = model.generate_audio(voice_state, "Test")
        print(f"   Generated: {test_audio.shape}")

        print("✅ Voice cloning debug completed successfully")

    except Exception as e:
        print(f"❌ Error in step: {e}")
        import traceback
        traceback.print_exc()

debug_voice_cloning("reference.wav")
```

### Generate Diagnostic Report

```python
def generate_diagnostic_report():
    """Generate comprehensive diagnostic report."""
    import platform
    import torch
    import psutil

    report = {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3)
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads()
        },
        "pocket_tts": {
            "import_success": True,
            "model_loading": None,
            "voice_cloning": None
        }
    }

    try:
        from pocket_tts import TTSModel
        report["pocket_tts"]["import_success"] = True

        # Test model loading
        try:
            model = TTSModel.load_model()
            report["pocket_tts"]["model_loading"] = "Success"
            report["pocket_tts"]["sample_rate"] = model.sample_rate
        except Exception as e:
            report["pocket_tts"]["model_loading"] = f"Failed: {e}"

        # Test voice cloning
        try:
            voice_state = model.get_state_for_audio_prompt("alba")
            report["pocket_tts"]["voice_cloning"] = "Success"
        except Exception as e:
            report["pocket_tts"]["voice_cloning"] = f"Failed: {e}"

    except ImportError as e:
        report["pocket_tts"]["import_success"] = f"Failed: {e}"

    return report

# Generate and print report
report = generate_diagnostic_report()
print("=== Pocket TTS Diagnostic Report ===")
for category, data in report.items():
    print(f"\n{category.upper()}:")
    for key, value in data.items():
        print(f"  {key}: {value}")
```

## Getting Help

If you're still experiencing issues:

1. **Check the GitHub Issues**: https://github.com/kyutai-labs/pocket-tts/issues
2. **Create a Minimal Reproducible Example**:
   ```python
   from pocket_tts import TTSModel

   # Minimal example that fails
   model = TTSModel.load_model()
   voice_state = model.get_state_for_audio_prompt("alba")
   audio = model.generate_audio(voice_state, "Hello")
   ```

3. **Include System Information**:
   - Operating system and version
   - Python version
   - Pocket TTS version
   - Error messages and tracebacks
   - Steps to reproduce

4. **Provide Diagnostic Report**:
   ```python
   # Run the diagnostic report function from above
   report = generate_diagnostic_report()
   # Include this in your issue report
   ```

This troubleshooting guide should help resolve most common issues with Pocket TTS. For more specific problems, don't hesitate to reach out to the community.
