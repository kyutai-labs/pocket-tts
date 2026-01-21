import copy
import logging
import os
import queue
import statistics
import threading
import time
from functools import lru_cache
from pathlib import Path

import safetensors
import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.utils.config import Config, load_config
from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    display_execution_time,
    download_if_necessary,
    load_predefined_voice,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import get_flow_lm_state_dict, get_mimi_state_dict

torch.set_num_threads(1)
logger = logging.getLogger(__name__)


class TTSModel(nn.Module):
    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold,
        config: Config,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True

    @property
    def device(self) -> str:
        return next(self.parameters()).device.type

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls, config: Config, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm, latent_dim=config.mimi.quantizer.dimension
        )
        tts_model = cls(flow_lm, temp, lsd_decode_steps, noise_clamp, eos_threshold, config)
        return tts_model

    @classmethod
    def _from_pydantic_config_with_weights(
        cls, config: Config, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        tts_model = cls._from_pydantic_config(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros((1024, 512), dtype=torch.float32)
        )
        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
            state_dict_flowlm = get_flow_lm_state_dict(
                download_if_necessary(config.flow_lm.weights_path)
            )
            tts_model.flow_lm.load_state_dict(state_dict_flowlm, strict=True)

        # safetensors.torch.save_file(tts_model.state_dict(), "7442637a.safetensors")
        # Create mimi config directly from the provided config using model_dump
        mimi_config = config.mimi.model_dump()

        # Build mimi model from config
        encoder = SEANetEncoder(**mimi_config["seanet"])
        decoder = SEANetDecoder(**mimi_config["seanet"])

        encoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        decoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        quantizer = DummyQuantizer(**mimi_config["quantizer"])

        tts_model.mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        ).to(device="cpu")

        # Load mimi weights from the config safetensors file with complete mapping for strict loading

        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
            mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
            tts_model.mimi.load_state_dict(mimi_state, strict=True)

        tts_model.mimi.eval()
        # tts_model.to(dtype=torch.float32)

        # uncomment to save the weights
        # tts_model = tts_model.to(dtype=torch.bfloat16)
        # safetensors.torch.save_file(tts_model.state_dict(), "tts_b6369a24.safetensors")
        if config.weights_path is not None:
            logger.info(f"Loading TTSModel weights from {config.weights_path}")
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            state_dict = safetensors.torch.load_file(weights_file)
            tts_model.load_state_dict(state_dict, strict=True)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )
        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
        logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")

        return tts_model

    def load_model(
        variant: str = DEFAULT_VARIANT,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    ) -> Self:
        """Load a pre-trained TTS model with specified configuration.

        This class method loads a complete TTS model including the flow language model
        and Mimi compression model from pre-trained weights. The model is initialized
        with the specified generation parameters and ready for inference.

        Args:
            variant: Model variant identifier corresponding to a config file name
                (e.g., '610b0b2c'). Must match a YAML file in the config directory.
            temp: Sampling temperature for generation. Higher values produce more
                diverse but potentially lower quality output.
            lsd_decode_steps: Number of steps for Lagrangian Self Distillation
                decoding. More steps can improve quality but increase computation.
            noise_clamp: Maximum value for noise sampling. If None, no clamping
                is applied. Helps prevent extreme values in generation.
            eos_threshold: Threshold for end-of-sequence detection. Higher values
                make the model more likely to continue generating.

        Returns:
            TTSModel: Fully initialized model with loaded weights on cpu, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.
        """
        config = load_config(Path(__file__).parents[1] / f"config/{variant}.yaml")
        tts_model = TTSModel._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: dict,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First one is the backbone output, second one is the audio decoding output."""
        if text_tokens is None:
            text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=self.flow_lm.device)
        if backbone_input_latents is None:
            backbone_input_latents = torch.empty(
                (1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )
        if audio_conditioning is None:
            audio_conditioning = torch.empty(
                (1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm(
        self,
        model_state: dict,
        text_tokens: torch.Tensor,
        backbone_input_latents: torch.Tensor,
        audio_conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        encoded = self.mimi.encode_to_latent(audio)
        latents = encoded.transpose(-1, -2).to(torch.float32)
        conditioning = F.linear(latents, self.flow_lm.speaker_proj_weight)
        return conditioning

    def _slice_kv_cache(self, model_state: dict, num_frames: int) -> None:
        """Slice KV cache to only keep the first num_frames elements.

        This optimizes memory usage when caching voice states by discarding
        unused cache capacity beyond the actual audio prompt length.

        Args:
            model_state: The model state dict containing KV caches for all modules
            num_frames: Number of frames to keep in the KV cache
        """
        original_size = 0
        sliced_size = 0
        for module_name, module_state in model_state.items():
            if "cache" in module_state:
                # KV cache has shape [2, batch_size, sequence_length, num_heads, dim_per_head]
                cache = module_state["cache"]
                original_size += cache.numel() * cache.element_size()
                # Slice to keep only the first num_frames positions
                module_state["cache"] = cache[:, :, :num_frames, :, :].clone()
                sliced_size += module_state["cache"].numel() * module_state["cache"].element_size()

        memory_saved_mb = (original_size - sliced_size) / (1024 * 1024)
        logger.info(
            f"Sliced KV cache from {original_size / (1024 * 1024):.1f} MB to {sliced_size / (1024 * 1024):.1f} MB "
            f"(saved {memory_saved_mb:.1f} MB)"
        )

    def _expand_kv_cache(self, model_state: dict, sequence_length: int) -> None:
        """Expand KV cache back to full sequence_length for generation.

        When a model state is retrieved from cache with sliced KV caches,
        this method expands them back to the full size needed for generation.

        Args:
            model_state: The model state dict containing potentially sliced KV caches
            sequence_length: Target sequence length to expand caches to
        """
        for module_name, module_state in model_state.items():
            if "cache" in module_state:
                cache = module_state["cache"]
                # KV cache has shape [2, batch_size, current_length, num_heads, dim_per_head]
                current_length = cache.shape[2]
                if current_length < sequence_length:
                    # Create expanded cache filled with NaN for unused positions
                    expanded_cache = torch.full(
                        (
                            cache.shape[0],
                            cache.shape[1],
                            sequence_length,
                            cache.shape[3],
                            cache.shape[4],
                        ),
                        float("NaN"),
                        device=cache.device,
                        dtype=cache.dtype,
                    )
                    # Copy existing data to the beginning
                    expanded_cache[:, :, :current_length, :, :] = cache
                    module_state["cache"] = expanded_cache

    @torch.no_grad
    def _decode_audio_worker(self, latents_queue: queue.Queue, result_queue: queue.Queue):
        """Worker thread function for decoding audio latents from queue with immediate streaming."""
        try:
            audio_chunks = []
            mimi_state = init_states(self.mimi, batch_size=1, sequence_length=1000)
            while True:
                latent = latents_queue.get()
                if latent is None:
                    break
                mimi_decoding_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                transposed = mimi_decoding_input.transpose(-1, -2)
                quantized = self.mimi.quantizer(transposed)

                t = time.monotonic()
                audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
                increment_steps(self.mimi, mimi_state, increment=16)
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate
                # We could log the timings here.
                logger.debug(
                    " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    int((time.monotonic() - t) * 1000),
                )
                audio_chunks.append(audio_frame)

                result_queue.put(("chunk", audio_frame))

                latents_queue.task_done()

            # Signal completion
            result_queue.put(("done", None))

        except Exception as e:
            # Put error in result queue
            result_queue.put(("error", e))

    @torch.no_grad
    def generate_audio(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> torch.Tensor:
        """Generate complete audio tensor from text input.

        This method generates the full audio output for the given text prompt
        and returns it as a single tensor. It internally uses the streaming
        generation method but collects all chunks before returning.

        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames).
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.

        Returns:
            torch.Tensor: Generated audio tensor with shape [channels, samples]
                at the model's sample rate (typically 24kHz). The audio is
                normalized and ready for playback or saving.
                You can get the sample rate from the `sample_rate` attribute.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors.
        """
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
        ):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad
    def generate_audio_stream(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ):
        """Generate audio streaming chunks from text input.

        This method generates audio from text and yields chunks as they become
        available, enabling real-time playback or processing. It uses multithreading
        to parallelize generation and decoding for optimal performance.
        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames). Defaults to None.
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.

        Yields:
            torch.Tensor: Audio chunks with shape [samples] at the model's
                sample rate (typically 24kHz). Chunks are yielded as soon as
                they are decoded, enabling real-time streaming.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors or threading issues.

        Note:
            This method uses multithreading to parallelize latent generation
            and audio decoding. Generation performance is logged including
            real-time factor (RTF) metrics.
        """

        # This is a very simplistic way of handling long texts. We could do much better
        # by using teacher forcing, but it would be a bit slower.
        # TODO: add the teacher forcing method for long texts where we use the audio of one chunk
        # as conditioning for the next chunk.
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer, text_to_generate, max_tokens
        )

        for chunk in chunks:
            text_to_generate, frames_after_eos_guess = prepare_text_prompt(chunk)
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=chunk,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
            )

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self, model_state: dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
    ):
        if copy_state:
            model_state = copy.deepcopy(model_state)

        # Expand sliced KV caches back to full size for generation
        self._expand_kv_cache(model_state, sequence_length=1000)

        # Set up multithreaded generation and decoding
        latents_queue = queue.Queue()
        result_queue = queue.Queue()

        # Start decoder worker thread
        decoder_thread = threading.Thread(
            target=self._decode_audio_worker, args=(latents_queue, result_queue), daemon=True
        )
        logger.info("starting timer now!")
        t_generating = time.monotonic()
        decoder_thread.start()

        # Generate latents and add them to queue (decoder processes them in parallel)
        self._generate(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
        )

        # Stream audio chunks as they become available
        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                # Audio chunk available immediately for streaming/playback
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]  # Remove batch, channel
            elif result[0] == "done":
                # Generation complete
                break
            elif result[0] == "error":
                # Wait for decoder thread to finish cleanly before propagating error
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                # Propagate error
                raise result[1]

        # Wait for decoder thread to finish cleanly
        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

        # Print timing information
        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / generation_time

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    @torch.no_grad
    def _generate(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
    ):
        gen_len_sec = len(text_to_generate.split()) * 1 + 2.0
        max_gen_len = int(gen_len_sec * 12.5)
        prepared = self.flow_lm.conditioner.prepare(text_to_generate)

        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state, text_tokens=prepared.tokens
            )

        def run_generation():
            try:
                self._autoregressive_generation(
                    model_state, max_gen_len, frames_after_eos, latents_queue
                )
            except Exception as e:
                logger.error(f"Error in autoregressive generation: {e}")
                # Signal decoder to stop by putting None (completion sentinel)
                if latents_queue is not None:
                    latents_queue.put(None)
                # Report error to main thread
                if result_queue is not None:
                    result_queue.put(("error", e))

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    @torch.no_grad
    def _autoregressive_generation(
        self, model_state: dict, max_gen_len: int, frames_after_eos: int, latents_queue: queue.Queue
    ):
        backbone_input = torch.full(
            (1, 1, self.flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(self.flow_lm.parameters())).device,
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        for generation_step in range(max_gen_len):
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )
                if is_eos.item() and eos_step is None:
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                # Add generated latent to queue for immediate decoding
                latents_queue.put(next_latent)
                backbone_input = next_latent
            steps_times.append(timer.elapsed_time_ms)
        else:
            if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, this very often indicates an error."
            )

        # Add sentinel value to signal end of generation
        latents_queue.put(None)
        logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        """Create model state conditioned on audio prompt for continuation.

        This method processes an audio prompt and creates a model state that
        captures the acoustic characteristics (speaker voice, style, prosody)
        for use in subsequent text-to-speech generation. The resulting state
        enables voice cloning and audio continuation with speaker consistency.

        Args:
            audio_conditioning: Audio prompt to condition on. Can be:
                - Path: Local file path to audio file
                - str: URL to download audio file from
                - torch.Tensor: Pre-loaded audio tensor with shape [channels, samples]
            truncate: Whether to truncate long audio prompts to 30 seconds.
                Helps prevent memory issues with very long inputs. Defaults to False.

        Returns:
            dict: Model state dictionary containing hidden states and positional
                information conditioned on the audio prompt. This state can be
                passed to `generate_audio()` or `generate_audio_stream()` for
                voice-consistent generation.

        Raises:
            FileNotFoundError: If audio file path doesn't exist.
            ValueError: If audio tensor is invalid or empty.
            RuntimeError: If audio processing or encoding fails.

        Note:
            - Audio is automatically resampled to the model's sample rate (24kHz)
            - The audio is encoded using the Mimi compression model and projected
              to the flow model's latent space
            - Processing time is logged for performance monitoring
            - The state preserves speaker characteristics for voice cloning
        """
        if isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICES:
            # We get the audio conditioning directly from the safetensors file.
            prompt = load_predefined_voice(audio_conditioning)
        else:
            if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
                raise ValueError(
                    f"We could not download the weights for the model with voice cloning, "
                    f"but you're trying to use voice cloning. "
                    f"Without voice cloning, you can use our catalog of voices {list(PREDEFINED_VOICES)}. "
                    f"If you want access to the model with voice cloning, go to "
                    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
                    f"then make sure you're logged in locally with `uvx hf auth login`."
                )
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            if isinstance(audio_conditioning, Path):
                audio, conditioning_sample_rate = audio_read(audio_conditioning)

                if truncate:
                    max_samples = int(30 * conditioning_sample_rate)  # 30 seconds of audio
                    if audio.shape[-1] > max_samples:
                        audio = audio[..., :max_samples]
                        logger.info(f"Audio truncated to first 30 seconds ({max_samples} samples)")

                audio_conditioning = convert_audio(
                    audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
                )

            with display_execution_time("Encoding audio prompt"):
                prompt = self._encode_audio(audio_conditioning.unsqueeze(0).to(self.device))
                # import safetensors.torch
                # safetensors.torch.save_file(
                #     {"audio_prompt": prompt},
                #     "/projects/huggingface/pocket-tts/embeddings/cosette.safetensors"
                # )

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=1000)

        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        # Optimize memory by slicing KV cache to only keep frames from the audio prompt
        num_audio_frames = prompt.shape[1]
        self._slice_kv_cache(model_state, num_audio_frames)

        return model_state


def prepare_text_prompt(text: str) -> tuple[str, int]:
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    # Make sure it starts with an uppercase letter
    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    # Let's make sure it ends with some kind of punctuation
    # If it ends with a letter or digit, we add a period.
    if text[-1].isalnum():
        text = text + "."

    # The model does not perform well when there are very few tokens, so
    # we can add empty spaces at the beginning to increase the token count.
    if len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


def split_into_best_sentences(tokenizer, text_to_generate: str, max_tokens: int) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(text_to_generate)
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()

    end_of_sentences_indices = [0]
    previous_was_end_of_sentence_token = False

    for token_idx, token in enumerate(list_of_tokens):
        if token in end_of_sentence_tokens:
            previous_was_end_of_sentence_token = True
        else:
            if previous_was_end_of_sentence_token:
                end_of_sentences_indices.append(token_idx)
            previous_was_end_of_sentence_token = False
    end_of_sentences_indices.append(len(list_of_tokens))

    nb_tokens_and_sentences = []
    for i in range(len(end_of_sentences_indices) - 1):
        # let's print
        start = end_of_sentences_indices[i]
        end = end_of_sentences_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        nb_tokens_and_sentences.append((end - start, text))

    max_nb_tokens_in_a_chunk = max_tokens
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in nb_tokens_and_sentences:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    return chunks
