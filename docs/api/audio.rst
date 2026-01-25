Audio I/O Functions
===================

This module provides functions for loading and saving audio files with automatic format detection and conversion.

.. automodule:: pocket_tts.data.audio
   :members:
   :undoc-members:

Functions
---------

load_wav
~~~~~~~

.. autofunction:: pocket_tts.data.audio.load_wav

save_audio
~~~~~~~~~~

.. autofunction:: pocket_tts.data.audio_output.save_audio

Usage Examples
--------------

Loading Audio Files
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import load_wav

   # Load a WAV file
   audio, sample_rate = load_wav("speech.wav")
   print(f"Loaded audio with shape: {audio.shape}")
   print(f"Sample rate: {sample_rate} Hz")

   # Load other formats (requires soundfile)
   audio, sample_rate = load_wav("audio.mp3")

Saving Audio Files
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from pocket_tts import save_audio

   # Create some test audio
   sample_rate = 24000
   duration = 1.0  # 1 second
   t = torch.linspace(0, duration, int(sample_rate * duration))
   audio = torch.sin(2 * torch.pi * 440 * t) * 0.5  # 440 Hz sine wave

   # Save to WAV file
   save_audio("output.wav", audio, sample_rate)

Format Support
--------------

Supported formats:

* **WAV**: Native support using Python's wave module
* **MP3, FLAC, OGG**: Via soundfile (install with `pip install soundfile`)

All audio is automatically:

* Converted to mono (multi-channel mixed to single channel)
* Normalized to float32 range [-1, 1]
* Reshaped to [1, samples] format for consistency
