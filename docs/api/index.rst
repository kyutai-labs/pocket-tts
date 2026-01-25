Pocket TTS API Documentation
=============================

Welcome to the Pocket TTS API documentation. This page provides comprehensive documentation for all public APIs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models
   audio
   rust_audio
   examples

Core Features
------------

* **Text-to-Speech Generation**: Convert text to natural-sounding speech
* **Voice Cloning**: Clone voices from audio samples
* **Audio Processing**: Comprehensive audio I/O and processing utilities
* **Streaming**: Real-time audio generation with low latency
* **CPU Optimized**: Efficient CPU-based processing, no GPU required

Getting Started
---------------

Basic usage example:

.. code-block:: python

   from pocket_tts import TTSModel
   from pocket_tts import save_audio

   # Load the model
   model = TTSModel.load_model()

   # Create voice state from a pre-defined voice or audio file
   voice_state = model.get_state_for_audio_prompt("alba")

   # Generate speech
   audio = model.generate_audio(voice_state, "Hello, this is Pocket TTS!")

   # Save to file
   save_audio("output.wav", audio, model.sample_rate)

API Reference
-------------

.. autoclass:: pocket_tts.TTSModel
   :members:

.. automodule:: pocket_tts.data.audio
   :members:

.. automodule:: pocket_tts.rust_audio
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
