Examples
========

This section provides practical examples for common Pocket TTS use cases.

Basic Usage
-----------

Simple Text-to-Speech
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, save_audio

   # Load the model
   model = TTSModel.load_model()

   # Use a pre-defined voice
   voice_state = model.get_state_for_audio_prompt("alba")

   # Generate speech
   audio = model.generate_audio(voice_state, "Hello, this is Pocket TTS!")

   # Save to file
   save_audio("hello.wav", audio, model.sample_rate)
   print(f"Audio saved to hello.wav ({model.sample_rate} Hz)")

Voice Cloning
-------------

Clone Voice from Audio File
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, load_wav, save_audio

   model = TTSModel.load_model()

   # Load reference audio for voice cloning
   reference_audio, sr = load_wav("reference_voice.wav")

   # Create voice state from the reference
   voice_state = model.get_state_for_audio_prompt(reference_audio)

   # Generate speech with cloned voice
   text = "This sounds like the original speaker!"
   audio = model.generate_audio(voice_state, text)

   save_audio("cloned_voice.wav", audio, model.sample_rate)

Voice Cloning from URL
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, save_audio

   model = TTSModel.load_model()

   # Use voice from Hugging Face
   voice_url = "https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav"
   voice_state = model.get_state_for_audio_prompt(voice_url)

   audio = model.generate_audio(voice_state, "Voice cloning from URL works!")
   save_audio("url_cloned.wav", audio, model.sample_rate)

Streaming and Real-time
-----------------------

Real-time Audio Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel
   import torch
   import time

   model = TTSModel.load_model()
   voice_state = model.get_state_for_audio_prompt("alba")

   long_text = "This is a very long text that will be processed in chunks to demonstrate real-time streaming capabilities."

   print("Starting real-time generation...")
   start_time = time.time()

   # Generate audio in streaming chunks
   total_samples = 0
   for i, chunk in enumerate(model.generate_audio_stream(voice_state, long_text)):
       chunk_samples = chunk.shape[0]
       total_samples += chunk_samples

       # In a real application, you would play these chunks immediately
       print(f"Chunk {i+1}: {chunk_samples} samples")

       # Simulate real-time processing
       time.sleep(0.01)  # Small delay to simulate processing

   duration = total_samples / model.sample_rate
   generation_time = time.time() - start_time
   real_time_factor = duration / generation_time

   print(f"Generated {duration:.2f}s of audio in {generation_time:.2f}s")
   print(f"Real-time factor: {real_time_factor:.2f}x")

Audio Processing
----------------

Audio Enhancement
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pocket_tts import TTSModel, normalize_audio, apply_fade, compute_audio_metrics, save_audio

   model = TTSModel.load_model()
   voice_state = model.get_state_for_audio_prompt("alba")

   # Generate audio
   audio = model.generate_audio(voice_state, "This audio will be enhanced.")

   # Convert to numpy for processing
   audio_np = audio.numpy()

   # Analyze original audio
   original_metrics = compute_audio_metrics(audio_np)
   print(f"Original RMS: {original_metrics['rms']:.3f}")

   # Normalize and enhance
   enhanced = normalize_audio(audio_np, gain=1.2)

   # Apply fade effects
   enhanced = apply_fade(enhanced, fade_in_ms=50, fade_out_ms=50, sample_rate=model.sample_rate)

   # Analyze enhanced audio
   enhanced_metrics = compute_audio_metrics(enhanced)
   print(f"Enhanced RMS: {enhanced_metrics['rms']:.3f}")

   # Convert back to tensor and save
   enhanced_tensor = torch.from_numpy(enhanced)
   save_audio("enhanced.wav", enhanced_tensor, model.sample_rate)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, save_audio
   import concurrent.futures
   import time

   model = TTSModel.load_model()
   voice_state = model.get_state_for_audio_prompt("alba")

   texts = [
       "First sentence to generate.",
       "Second sentence to generate.",
       "Third sentence to generate.",
       "Fourth sentence to generate.",
       "Fifth sentence to generate."
   ]

   def generate_speech(text, index):
       """Generate speech for a single text."""
       audio = model.generate_audio(voice_state, text)
       filename = f"output_{index:02d}.wav"
       save_audio(filename, audio, model.sample_rate)
       return filename

   # Sequential processing
   start_time = time.time()
   for i, text in enumerate(texts):
       filename = generate_speech(text, i)
       print(f"Generated {filename}")
   sequential_time = time.time() - start_time

   print(f"Sequential processing: {sequential_time:.2f}s")

   # Parallel processing (note: model is not thread-safe, this is for illustration)
   print("For parallel processing, create separate model instances per thread.")

Error Handling
--------------

Robust Error Handling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, load_wav, save_audio
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def safe_tts_generation(model, voice_source, text, output_path):
       """Safely generate TTS with error handling."""
       try:
           # Create voice state
           if isinstance(voice_source, str) and voice_source.endswith('.wav'):
               voice_state = model.get_state_for_audio_prompt(load_wav(voice_source))
           else:
               voice_state = model.get_state_for_audio_prompt(voice_source)

           # Generate audio
           audio = model.generate_audio(voice_state, text)

           # Save audio
           save_audio(output_path, audio, model.sample_rate)
           logger.info(f"Successfully generated {output_path}")
           return True

       except FileNotFoundError as e:
           logger.error(f"File not found: {e}")
       except ValueError as e:
           logger.error(f"Invalid input: {e}")
       except RuntimeError as e:
           logger.error(f"Generation failed: {e}")
       except Exception as e:
           logger.error(f"Unexpected error: {e}")

       return False

   # Usage
   model = TTSModel.load_model()
   success = safe_tts_generation(model, "alba", "Hello world!", "output.wav")

   if success:
       print("Generation successful!")
   else:
       print("Generation failed, check logs.")

Performance Tips
----------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel
   import gc

   # Good: Keep model loaded for multiple generations
   model = TTSModel.load_model()
   voice_state = model.get_state_for_audio_prompt("alba")

   # Generate multiple utterances efficiently
   for i in range(10):
       audio = model.generate_audio(voice_state, f"Sentence number {i+1}.")
       # Process audio here...

   # Clean up if needed
   del model, voice_state
   gc.collect()

Voice State Caching
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel
   import functools

   model = TTSModel.load_model()

   # Cache voice states to avoid recomputation
   @functools.lru_cache(maxsize=5)
   def get_cached_voice_state(voice_name):
       """Cache voice states for frequently used voices."""
       return model.get_state_for_audio_prompt(voice_name)

   # Use cached voice states
   voices = ["alba", "marius", "javert", "jean", "fantine"]

   for voice in voices:
       voice_state = get_cached_voice_state(voice)
       audio = model.generate_audio(voice_state, f"Hello from {voice}!")
       # Process audio...

   print("Voice states cached for efficient reuse")
