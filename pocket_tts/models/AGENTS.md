# Models (pocket_tts/models)

Core TTS models orchestrating flow-based generation with Mimi codec for neural audio compression/decompression.

## WHERE TO LOOK

- `tts_model.py`: TTSModel orchestrates entire pipeline - voice state creation, streaming generation, EOS detection (929 lines, core orchestration)
- `__init__.py`: Package initialization; exports only `TTSModel` to keep internals private
- `flow_lm.py`: FlowLMModel implements transformer with Lagrangian Self Distillation for latent sampling
- `mimi.py`: MimiModel provides neural audio codec (SEANet encoder/decoder with ProjectedTransformer)
- Key entry: TTSModel.load_model() → get_state_for_audio_prompt() → generate_audio_stream()

## ALGORITHMIC HOTSPOTS

- **FlowLM Generation**: Uses `lsd_decode()` with configurable steps (default 1) - more steps = better quality but slower. Takes noise samples and iteratively refines via flow network.
- **EOS Detection**: FlowLM's `out_eos` head predicts sequence end; generation continues for `frames_after_eos` frames after detection to smooth audio tail.
- **Voice Cloning**: Audio encoded through Mimi → projected to flow space via `speaker_proj_weight` → conditions generation. LRU cache on `_cached_get_state_for_audio_prompt()` avoids reprocessing; cache size configurable via `POCKET_TTS_PROMPT_CACHE_SIZE`.
- **Streaming**: Multithreaded pipeline separates latent generation (main thread) and Mimi decoding (decoder thread) for real-time audio output. Queues pass latents for immediate decoding. Decoder thread runs in background as daemon.
- **Frame Rate**: Generates at 12.5 Hz (80ms/frame). Mimi operates at 12.5 Hz but encoder/decoder at 75 Hz, with ConvUpsample1d/Downsample1d bridging frame rates.
- **LSD Algorithm**: Lagrangian Self Distillation (flow matching) progressively denoises from known distribution to target latent. Flow direction v_t(s, t, x_t) guides reconstruction.

## STATE MANAGEMENT

- StatefulModule base provides KV cache and step tracking for streaming
- model_state dict holds hidden states, positional encodings across generation
- `trim_model_state()` reduces memory for cached voice prompts by truncating unused KV cache
- `increment_steps()` advances positional tracking after each forward pass; critical for streaming to maintain sequence continuity
- Non-thread-safe - single generation per model instance; concurrent requests require separate TTSModel instances
- `copy_state=True` in generation methods preserves original state for reuse; `False` modifies in-place for memory efficiency
- `torch.set_num_threads(1)` in tts_model.py for optimal CPU performance (single-threaded PyTorch operations)
