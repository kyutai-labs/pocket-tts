Rust Audio Processing
====================

This module provides high-performance audio processing functions implemented in Rust with Python fallbacks.

.. automodule:: pocket_tts.rust_audio
   :members:
   :undoc-members:

Functions
---------

normalize_audio
~~~~~~~~~~~~~~

.. autofunction:: pocket_tts.rust_audio.normalize_audio

apply_gain
~~~~~~~~~~

.. autofunction:: pocket_tts.rust_audio.apply_gain

resample_audio
~~~~~~~~~~~~~~

.. autofunction:: pocket_tts.rust_audio.resample_audio

apply_fade
~~~~~~~~~~

.. autofunction:: pocket_tts.rust_audio.apply_fade

compute_audio_metrics
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pocket_tts.rust_audio.compute_audio_metrics

Usage Examples
--------------

Audio Normalization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pocket_tts import normalize_audio

   # Create quiet audio
   audio = np.random.randn(24000) * 0.1
   print(f"Original RMS: {np.sqrt(np.mean(audio**2)):.3f}")

   # Normalize with additional gain
   normalized = normalize_audio(audio, gain=2.0)
   print(f"Normalized RMS: {np.sqrt(np.mean(normalized**2)):.3f}")

Gain and Fade Effects
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pocket_tts import apply_gain, apply_fade

   # Generate test tone
   sample_rate = 24000
   t = np.linspace(0, 2, sample_rate * 2)
   audio = np.sin(2 * np.pi * 440 * t) * 0.5

   # Apply gain
   louder = apply_gain(audio, 2.0)

   # Apply fade in/out
   faded = apply_fade(audio, fade_in_ms=100, fade_out_ms=100, sample_rate=sample_rate)

Audio Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pocket_tts import compute_audio_metrics

   # Generate test audio
   audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)) * 0.8

   # Analyze
   metrics = compute_audio_metrics(audio)
   print(f"RMS level: {metrics['rms']:.3f}")
   print(f"Peak level: {metrics['peak']:.3f}")
   print(f"Dynamic range: {metrics['dynamic_range_db']:.1f} dB")

Performance
-----------

The Rust implementation provides significant performance benefits:

* **Memory Efficiency**: Zero-copy operations where possible
* **Speed**: Optimized SIMD instructions for audio processing
* **Fallback**: Automatic Python fallback when Rust library is not available

The functions automatically detect and use the Rust implementation when available, with seamless fallback to Python implementations.
