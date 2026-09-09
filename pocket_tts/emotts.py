import json
import logging
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn

from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.utils import download_if_necessary

logger = logging.getLogger(__name__)


class EmoProsodyParams(TypedDict):
    temp: float
    noise_clamp: float | None
    lsd_decode_steps: int


EMOTION_PROSODY_DEFAULTS: dict[str, EmoProsodyParams] = {
    "angry": {"temp": 1.3, "noise_clamp": 3.0, "lsd_decode_steps": 1},
    "disgust": {"temp": 0.5, "noise_clamp": 1.5, "lsd_decode_steps": 2},
    "fear": {"temp": 1.6, "noise_clamp": None, "lsd_decode_steps": 1},
    "happy": {"temp": 0.9, "noise_clamp": 2.5, "lsd_decode_steps": 2},
    "neutral": {"temp": 0.7, "noise_clamp": None, "lsd_decode_steps": 1},
    "sad": {"temp": 0.4, "noise_clamp": 0.8, "lsd_decode_steps": 3},
}


class EmoShiftLayer(nn.Module):
    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

    def __init__(self, hidden_size: int = 1024, injection_layer: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.injection_layer = injection_layer
        self.steering_vectors = nn.ParameterDict(
            {emotion: nn.Parameter(torch.zeros(hidden_size)) for emotion in self.EMOTIONS}
        )
        self.active_emotion = "neutral"
        self.intensity = 0.0

    def set_emotion(self, emotion: str, intensity: float):
        assert emotion in self.EMOTIONS, f"Choose from: {self.EMOTIONS}"
        assert intensity >= 0.0, "Intensity must be non-negative"
        self.active_emotion = emotion
        self.intensity = intensity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_emotion == "neutral" or self.intensity == 0.0:
            return x
        vec = self.steering_vectors[self.active_emotion]
        device = x.device
        dtype = x.dtype
        vec_shifted = vec.to(device=device, dtype=dtype)
        return x + self.intensity * vec_shifted.unsqueeze(0).unsqueeze(0)


class EmoTTS(TTSModel):
    """
    EmoTTS class extending TTSModel to support EmoShift activation steering.
    Automatically downloads and registers EmoShift weights on top of pocket-tts.
    """

    def __init__(
        self,
        tts_model: TTSModel,
        weights_path: Path,
        meta: dict | None = None,
        prosody_overrides: dict[str, dict] | None = None,
    ):
        # Initialize nn.Module first
        nn.Module.__init__(self)

        # Copy attributes from base model
        self.__dict__.update(tts_model.__dict__)

        # Instantiate and load EmoShiftLayer
        device = next(tts_model.parameters()).device
        self.emo_layer = EmoShiftLayer(hidden_size=1024, injection_layer=5)

        state = torch.load(weights_path, map_location="cpu")
        self.emo_layer.load_state_dict(state)
        self.emo_layer.to(device)
        self.emo_layer.eval()

        self.meta = meta or {}
        self._hook_handle = None
        self._register_hook()

        # Capture base generation parameters
        self._base_params = {
            "temp": float(self.temp),
            "noise_clamp": float(self.noise_clamp) if self.noise_clamp is not None else None,
            "lsd_decode_steps": int(self.lsd_decode_steps),
        }

        # Initialize prosody profile
        self._prosody_profile = {
            emo: params.copy() for emo, params in EMOTION_PROSODY_DEFAULTS.items()
        }

        # Validate and apply overrides
        if prosody_overrides is not None:
            if not isinstance(prosody_overrides, dict):
                raise TypeError("prosody_overrides must be a dictionary")
            for emo, params in prosody_overrides.items():
                if emo not in self.available_emotions():
                    raise ValueError(
                        f"Unknown emotion in overrides: '{emo}'. "
                        f"Choose from {self.available_emotions()}"
                    )
                if not isinstance(params, dict):
                    raise TypeError(f"Overrides for emotion '{emo}' must be a dictionary")

                # Check for required keys
                for key in ["temp", "noise_clamp", "lsd_decode_steps"]:
                    if key not in params:
                        raise ValueError(f"Missing required key '{key}' in overrides for '{emo}'")

                temp = params["temp"]
                nc = params["noise_clamp"]
                steps = params["lsd_decode_steps"]

                if not isinstance(temp, (int, float)):
                    raise TypeError(f"'temp' for '{emo}' must be a float or int, got {type(temp)}")
                if nc is not None and not isinstance(nc, (int, float)):
                    raise TypeError(
                        f"'noise_clamp' for '{emo}' must be float, int, or None, got {type(nc)}"
                    )
                if not isinstance(steps, int) or isinstance(steps, bool):
                    raise TypeError(
                        f"'lsd_decode_steps' for '{emo}' must be an int, got {type(steps)}"
                    )

                self._prosody_profile[emo] = {
                    "temp": float(temp),
                    "noise_clamp": float(nc) if nc is not None else None,
                    "lsd_decode_steps": int(steps),
                }

    def _register_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

        def emo_hook(module, input, output):
            if isinstance(output, tuple):
                activations = output[0]
                steered = self.emo_layer(activations)
                return (steered,) + output[1:]
            return self.emo_layer(output)

        self._hook_handle = self.flow_lm.transformer.layers[5].register_forward_hook(emo_hook)

    def set_emotion(self, emotion: str | None, intensity: float = 1.0):
        """Set the active emotion and scaling intensity, overriding acoustic prosody parameters."""
        if emotion is None:
            emotion = "neutral"

        if emotion not in self.available_emotions():
            raise ValueError(
                f"Unknown emotion: '{emotion}'. Supported emotions are: {self.available_emotions()}"
            )

        # Update steering vector layer
        self.emo_layer.set_emotion(emotion, intensity)

        # Override or restore prosody parameters
        if emotion == "neutral" or intensity == 0.0:
            # Restore base parameters
            self.temp = self._base_params["temp"]
            self.noise_clamp = self._base_params["noise_clamp"]
            self.lsd_decode_steps = self._base_params["lsd_decode_steps"]
        else:
            # Apply emotion-specific overrides
            params = self._prosody_profile[emotion]
            self.temp = params["temp"]
            self.noise_clamp = params["noise_clamp"]
            self.lsd_decode_steps = params["lsd_decode_steps"]

    def available_emotions(self) -> list[str]:
        """Return the list of supported emotions."""
        return EmoShiftLayer.EMOTIONS

    def cleanup(self):
        """Remove the PyTorch hook from the model."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        self.cleanup()

    @classmethod
    def load_model(
        cls,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        quantize: bool = False,
        weights_url: str = "hf://Sourajit123/SouraTTS/SouraTTS.pt",
        meta_url: str = "hf://Sourajit123/SouraTTS/SouraTTS.json",
        prosody_overrides: dict[str, dict] | None = None,
    ) -> "EmoTTS":
        """Load the base model and wrap it with EmoShift steering."""
        logger.info("Loading base TTSModel...")
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )

        logger.info(f"Downloading EmoShift weights from {weights_url}...")
        weights_path = download_if_necessary(weights_url)
        meta_path = download_if_necessary(meta_url)

        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata json: {e}")

        return cls(tts_model, weights_path, meta, prosody_overrides)
