TTSModel
=========

.. autoclass:: pocket_tts.models.tts_model.TTSModel
   :members:
   :undoc-members:
   :show-inheritance:

Core Methods
------------

The main methods you'll use for text-to-speech generation:

.. automethod:: pocket_tts.models.tts_model.TTSModel.load_model
.. automethod:: pocket_tts.models.tts_model.TTSModel.generate_audio
.. automethod:: pocket_tts.models.tts_model.TTSModel.generate_audio_stream
.. automethod:: pocket_tts.models.tts_model.TTSModel.get_state_for_audio_prompt

Properties
----------

.. autoattribute:: pocket_tts.models.tts_model.TTSModel.device
.. autoattribute:: pocket_tts.models.tts_model.TTSModel.sample_rate

Usage Examples
--------------

Basic Text-to-Speech
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, save_audio

   # Load model with default configuration
   model = TTSModel.load_model()

   # Use a pre-defined voice
   voice_state = model.get_state_for_audio_prompt("alba")

   # Generate speech
   audio = model.generate_audio(voice_state, "Hello world!")

   # Save the audio
   save_audio("output.wav", audio, model.sample_rate)

Voice Cloning
~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel, load_audio, save_audio

   model = TTSModel.load_model()

   # Clone voice from an audio file
   voice_state = model.get_state_for_audio_prompt("path/to/voice.wav")

   # Generate speech with cloned voice
   audio = model.generate_audio(voice_state, "This sounds like the original speaker!")

   save_audio("cloned_output.wav", audio, model.sample_rate)

Streaming Generation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pocket_tts import TTSModel
   import torch

   model = TTSModel.load_model()
   voice_state = model.get_state_for_audio_prompt("alba")

   # Generate audio in real-time chunks
   for chunk in model.generate_audio_stream(voice_state, "This is a long text..."):
       # Process each chunk as it becomes available
       print(f"Got chunk with {chunk.shape[0]} samples")
       # In a real application, you would play these chunks immediately

Performance Tips
----------------

* Keep the model loaded in memory for repeated use
* Reuse voice states when generating multiple utterances with the same voice
* Use streaming for long texts to reduce latency
* The model is optimized for CPU usage; GPU provides no significant speedup
