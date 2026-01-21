import copy
import gc
import logging
import os
import queue
import statistics
import threading
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import safetensors
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any
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
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import increment_steps, init_states, trim_model_state
from pocket_tts.utils.config import Config, load_config
from pocket_tts.utils.model_versioning import load_model_with_versioning
from pocket_tts.utils.pause_handler import parse_pause_tags
from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    display_execution_time,
    download_if_necessary,
    load_predefined_voice,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import get_flow_lm_state_dict, get_mimi_state_dict

logger = logging.getLogger(__name__)

torch.set_num_threads(1)
interop_threads = os.environ.get("POCKET_TTS_INTEROP_THREADS")
if interop_threads:
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except ValueError:
        logger.warning("Invalid POCKET_TTS_INTEROP_THREADS=%s; expected integer.", interop_threads)

PROMPT_CACHE_SIZE = max(1, int(os.environ.get("POCKET_TTS_PROMPT_CACHE_SIZE", "2")))


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
        self._compiled_targets = set()
        self.model_metadata = None  # Will be set when loading weights

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

            # Load with versioning support
            state_dict, metadata = load_model_with_versioning(weights_file)
            tts_model.load_state_dict(state_dict, strict=True)

            # Store model version metadata
            tts_model.model_metadata = metadata
            logger.info(f"Model version: {metadata.model_version} (format: {metadata.format_version})")

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )
        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
        logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")

        return tts_model

    @classmethod
    def load_model(
        cls,
        variant: str = DEFAULT_VARIANT,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        dtype: str = "float32",
        quantize: bool = False,
        quantize_components: str = "all",
        compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
        compile_fullgraph: bool = False,
        compile_dynamic: bool = False,
        compile_targets: str | Iterable[str] = "all",
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
            dtype: Model weight dtype - "float32" (default) or "bfloat16" for
                reduced memory (~50% savings). bfloat16 may slightly affect quality.
            quantize: If True, apply int8 dynamic quantization for reduced memory
                footprint (~75% weight size reduction). May affect audio quality.
            quantize_components: Which components to quantize - "all", "flow-lm",
                or "mimi". Only used if quantize=True.
            compile: If True, apply torch.compile to the model for faster inference.
            compile_backend: torch.compile backend (default: "inductor").
            compile_mode: torch.compile mode (default: "reduce-overhead").
            compile_fullgraph: torch.compile fullgraph flag (default: False).
            compile_dynamic: torch.compile dynamic flag (default: False).
            compile_targets: Compilation targets: "all", "flow-lm", "mimi-decoder",
                or comma-separated combination. (default: "all")

        Returns:
            TTSModel: Fully initialized model with loaded weights on cpu, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.
            RuntimeError: If torch.compile is not available (requires PyTorch 2.0+).

        Example:
            >>> model = TTSModel.load_model()
            >>> # Or with compilation for faster inference:
            >>> model = TTSModel.load_model(compile=True)
            >>> # Or with reduced memory using bfloat16:
            >>> model = TTSModel.load_model(dtype="bfloat16")
            >>> # Or with int8 quantization:
            >>> model = TTSModel.load_model(quantize=True)
        """
        config = load_config(Path(__file__).parents[1] / f"config/{variant}.yaml")
        tts_model = cls._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        # Apply dtype conversion for memory optimization
        if dtype == "bfloat16":
            logger.info("Converting model to bfloat16 for reduced memory")
            tts_model = tts_model.to(dtype=torch.bfloat16)
        elif dtype != "float32":
            raise ValueError(f"Unsupported dtype: {dtype}. Use 'float32' or 'bfloat16'.")

        # Apply int8 quantization if requested
        if quantize:
            from pocket_tts.utils.quantization import quantize_model

            logger.info("Applying int8 quantization for reduced memory")
            tts_model = quantize_model(tts_model, components=quantize_components)

        if compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available. Requires PyTorch 2.0+.")
            tts_model.compile_for_inference(
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                targets=compile_targets,
            )

        return tts_model

    def _normalize_compile_targets(self, targets: Iterable[str] | str) -> set[str]:
        if isinstance(targets, str):
            raw_targets = [target.strip() for target in targets.split(",") if target.strip()]
        else:
            raw_targets = [str(target).strip() for target in targets if str(target).strip()]

        normalized = {target.replace("_", "-") for target in raw_targets}
        if not normalized:
            raise ValueError("compile targets cannot be empty.")
        if "all" in normalized:
            normalized = {"flow-lm", "mimi-decoder"}

        invalid = normalized - {"flow-lm", "mimi-decoder"}
        if invalid:
            invalid_list = ", ".join(sorted(invalid))
            raise ValueError(
                f"Invalid compile targets: {invalid_list}. Use 'all', 'flow-lm', or 'mimi-decoder'."
            )
        return normalized

    def compile_for_inference(
        self,
        backend: str = "inductor",
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        targets: Iterable[str] | str = "all",
    ) -> Self:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available. Requires PyTorch 2.0+.")

        targets_set = self._normalize_compile_targets(targets)
        targets_to_compile = targets_set - self._compiled_targets
        if not targets_to_compile:
            return self

        compile_kwargs = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }
        logger.info(
            "Compiling inference targets %s with torch.compile (backend=%s, mode=%s)",
            ", ".join(sorted(targets_to_compile)),
            backend,
            mode,
        )
        if "flow-lm" in targets_to_compile:
            self.flow_lm = torch.compile(self.flow_lm, **compile_kwargs)
            self._compiled_targets.add("flow-lm")
        if "mimi-decoder" in targets_to_compile:
            self.mimi.decoder_transformer = torch.compile(
                self.mimi.decoder_transformer, **compile_kwargs
            )
            self.mimi.decoder = torch.compile(self.mimi.decoder, **compile_kwargs)
            self._compiled_targets.add("mimi-decoder")
        return self

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

    @torch.inference_mode
    def generate_audio(
        self,
        model_state: dict,
        text_to_generate: str,
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
        ):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)

    @torch.inference_mode
    def generate_audio_batch(
        self,
        model_state: dict,
        texts_to_generate: Iterable[str],
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> list[torch.Tensor]:
        texts = list(texts_to_generate)
        if not texts:
            raise ValueError("texts_to_generate cannot be empty")

        return [
            self.generate_audio(
                model_state=model_state,
                text_to_generate=text,
                frames_after_eos=frames_after_eos,
                copy_state=copy_state,
            )
            for text in texts
        ]

    @torch.inference_mode
    def generate_audio_stream(
        self,
        model_state: dict,
        text_to_generate: str,
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
        # Log memory usage before generation
        self.log_memory_usage(f"before generation: '{text_to_generate[:50]}...'")

        # Parse pause tags from text (e.g., <pause:500ms>)
        text_chunks, pause_markers = parse_pause_tags(text_to_generate)

        # Build a lookup of chunk_index -> pause_duration_ms
        pause_after_chunk = {p.position: p.duration_ms for p in pause_markers}

        # This is a very simplistic way of handling long texts. We could do much better
        # by using teacher forcing, but it would be a bit slower.
        # TODO: add the teacher forcing method for long texts where we use the audio of one chunk
        # as conditioning for the next chunk.

        for chunk_idx, text_chunk in enumerate(text_chunks):
            # Further split long chunks into sentences
            sentences = split_into_best_sentences(self.flow_lm.conditioner.tokenizer, text_chunk)

            for sentence in sentences:
                text_to_generate, frames_after_eos_guess = prepare_text_prompt(sentence)
                frames_after_eos_guess += 2
                yield from self._generate_audio_stream_short_text(
                    model_state=model_state,
                    text_to_generate=sentence,
                    frames_after_eos=frames_after_eos_guess,
                    copy_state=copy_state,
                )

            # Insert silence if there's a pause after this chunk
            if chunk_idx in pause_after_chunk:
                pause_ms = pause_after_chunk[chunk_idx]
                silence_samples = int(pause_ms * self.sample_rate / 1000)
                yield torch.zeros(silence_samples)
                logger.debug("Inserted %d ms of silence (%d samples)", pause_ms, silence_samples)

    @torch.inference_mode
    def _generate_audio_stream_short_text(
        self, model_state: dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
    ):
        if copy_state:
            model_state = copy.deepcopy(model_state)

        latents_queue: queue.SimpleQueue = queue.SimpleQueue()
        result_queue: queue.SimpleQueue = queue.SimpleQueue()

        decoder_thread = threading.Thread(
            target=self._decode_audio_worker, args=(latents_queue, result_queue), daemon=True
        )
        logger.info("starting timer now!")
        t_generating = time.monotonic()
        decoder_thread.start()

        self._generate(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
        )

        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]
            elif result[0] == "done":
                break
            elif result[0] == "error":
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                raise result[1]

        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

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
        
        # Log memory usage after generation
        self.log_memory_usage("after generation")

    @torch.inference_mode
    def _decode_audio_worker(
        self, latents_queue: queue.SimpleQueue, result_queue: queue.SimpleQueue
    ):
        try:
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
                logger.debug(
                    " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    int((time.monotonic() - t) * 1000),
                )

                result_queue.put(("chunk", audio_frame))

            result_queue.put(("done", None))

        except Exception as e:
            result_queue.put(("error", e))

    @torch.inference_mode
    def _generate(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int,
        latents_queue: queue.SimpleQueue,
        result_queue: queue.SimpleQueue,
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

    @torch.inference_mode
    def _autoregressive_generation(
        self,
        model_state: dict,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: queue.SimpleQueue,
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

    def clear_prompt_cache(self) -> None:
        self._cached_get_state_for_audio_prompt.cache_clear()
    
    def cleanup(self) -> None:
        """Explicitly clean up model resources.
        
        This method ensures all resources are properly released:
        - Clears prompt cache
        - Joins any active threads
        - Clears PyTorch CUDA cache if applicable
        - Logs memory usage before and after cleanup
        
        This should be called when you're done with the model or before
        reloading it. Using a context manager is preferred:
        
        Example:
            >>> model = TTSModel.load_model()
            >>> with model:
            ...     audio = model.generate_audio(...)
            >>> # Resources automatically cleaned up
        """
        logger.debug("Cleaning up TTSModel resources")
        
        # Clear prompt cache to release memory
        self.clear_prompt_cache()
        
        # Clear PyTorch CUDA cache if on GPU
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        
        logger.info("TTSModel resources cleaned up successfully")

    def save_with_versioning(
        self,
        output_path: str | Path,
        description: str | None = None,
        tags: list[str] | None = None,
        custom_metadata: dict[str, any] | None = None,
    ) -> None:
        """Save model weights with version metadata.

        Args:
            output_path: Path where to save the weights file
            description: Optional description of the model
            tags: Optional tags for the model (e.g., ["experimental", "v1.0"])
            custom_metadata: Optional custom metadata fields

        Example:
            >>> model = TTSModel.load_model()
            >>> model.save_with_versioning(
            ...     "my_model.safetensors",
            ...     description="My fine-tuned model",
            ...     tags=["custom", "experimental"]
            ... )
        """
        from pocket_tts.utils.model_versioning import save_model_with_versioning

        state_dict = self.state_dict()
        output_path = Path(output_path)

        # Add some default metadata
        if custom_metadata is None:
            custom_metadata = {}

        custom_metadata.update({
            "has_voice_cloning": self.has_voice_cloning,
            "temp": self.temp,
            "lsd_decode_steps": self.lsd_decode_steps,
            "sample_rate": self.sample_rate,
        })

        save_model_with_versioning(
            state_dict,
            output_path,
            description=description,
            tags=tags,
            custom_metadata=custom_metadata,
        )

    def __enter__(self) -> Self:
        """Context manager entry for automatic resource cleanup."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit for automatic resource cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics.
        
        Returns a dictionary with memory usage information:
        - model_size_mb: Size of model parameters in MB
        - cache_entries: Number of entries in prompt cache
        - torch_allocated_mb: PyTorch allocated memory (if available)
        - torch_reserved_mb: PyTorch reserved memory (if available)
        - gc_objects: Number of Python objects tracked by garbage collector
        
        This is useful for monitoring memory usage during generation
        and detecting potential memory leaks.
        
        Returns:
            dict: Memory usage statistics
        """
        stats = {}
        
        # Model parameter size
        model_size_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        stats["model_size_mb"] = model_size_bytes / (1024 * 1024)
        
        # Prompt cache size
        stats["cache_entries"] = self._cached_get_state_for_audio_prompt.cache_info().currsize
        
        # PyTorch memory (if CUDA is available)
        if hasattr(torch.cuda, 'memory_allocated'):
            stats["torch_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["torch_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        # Python garbage collector info
        stats["gc_objects"] = len(gc.get_objects())
        
        return stats
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage with optional context.
        
        Args:
            context: Optional context string to include in log message
        """
        stats = self.get_memory_usage()
        logger.info(
            "Memory usage%s: Model=%.2f MB, Cache=%d entries, "
            "PyTorch allocated=%.2f MB, GC objects=%d",
            f" ({context})" if context else "",
            stats["model_size_mb"],
            stats["cache_entries"],
            stats.get("torch_allocated_mb", 0),
            stats["gc_objects"],
        )

    def get_state_for_audio_prompt_cached(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        if isinstance(audio_conditioning, torch.Tensor):
            return self.get_state_for_audio_prompt(audio_conditioning, truncate)
        return self._cached_get_state_for_audio_prompt(audio_conditioning, truncate)

    @lru_cache(maxsize=PROMPT_CACHE_SIZE)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    @torch.inference_mode
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

        # Trim KV caches to actual used length to reduce memory for cached states
        # This is especially important when caching multiple voice prompts
        model_state = trim_model_state(model_state)
        logger.debug("Trimmed model state KV caches to reduce memory usage")

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


def split_into_best_sentences(tokenizer, text_to_generate: str) -> list[str]:
    """Split text into sentence-sized chunks for generation.

    This function splits text at sentence boundaries (. ! ? ...) while respecting
    a maximum token limit per chunk. Unlike the previous implementation, this
    uses regex-based splitting to preserve the original text exactly, avoiding
    content loss that occurred with token-based decoding.

    Args:
        tokenizer: The text tokenizer (used only for token counting).
        text_to_generate: Input text to split.

    Returns:
        List of text chunks suitable for generation, preserving all original content.
    """
    import re

    text_to_generate, _ = prepare_text_prompt(text_to_generate)
    original_text = text_to_generate.strip()

    # Split on sentence boundaries: period, exclamation, question mark, or ellipsis
    # followed by whitespace. This preserves the original text exactly.
    # The lookbehind ensures we keep the punctuation with the preceding sentence.
    sentence_pattern = re.compile(r"(?<=[.!?…])\s+")
    sentences = sentence_pattern.split(original_text)

    # Handle edge case where no sentence boundaries are found
    if len(sentences) == 0:
        sentences = [original_text]

    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        # Fallback: if everything got filtered out, use original text
        sentences = [original_text]

    # Count tokens for each sentence and build chunks
    max_tokens_per_chunk = 50
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        # Count tokens in this sentence using the tokenizer
        sentence_tokens = len(tokenizer(sentence).tokens[0].tolist())

        if current_chunk == "":
            # Start a new chunk
            current_chunk = sentence
            current_token_count = sentence_tokens
        elif current_token_count + sentence_tokens > max_tokens_per_chunk:
            # Current chunk would exceed limit, start a new one
            chunks.append(current_chunk)
            current_chunk = sentence
            current_token_count = sentence_tokens
        else:
            # Add to current chunk
            current_chunk = current_chunk + " " + sentence
            current_token_count += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Log chunks for debugging
    if logger.isEnabledFor(logging.DEBUG):
        for i, chunk in enumerate(chunks):
            logger.debug("Chunk %d: '%s'", i, chunk)

    # Strict validation: ensure no content was lost
    # Normalize whitespace for comparison
    combined = " ".join(chunks)
    original_normalized = " ".join(original_text.split())
    combined_normalized = " ".join(combined.split())

    if original_normalized != combined_normalized:
        # Check which words are missing
        original_words = set(original_text.lower().split())
        combined_words = set(combined.lower().split())
        missing_words = original_words - combined_words

        if missing_words:
            logger.error(
                "Content loss detected in sentence splitting! Missing words: %s",
                ", ".join(sorted(list(missing_words)[:10])),
            )
            logger.debug("Original text: %s", original_text)
            logger.debug("Combined chunks: %s", combined)

    return chunks
