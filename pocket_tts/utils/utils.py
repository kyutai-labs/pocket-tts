import hashlib
import logging
import os
import time
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from torch import nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEBUG_MIMI = os.environ.get("DEBUG_MIMI", "0") == "1"

_ORIGINS_OF_PREDEFINED_VOICES = {
    "cosette": "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "marius": "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    "javert": "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "jean": "hf://kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav",
    "anna": "hf://kyutai/tts-voices/vctk/p228_023_enhanced.wav",
    "vera": "hf://kyutai/tts-voices/vctk/p229_023_enhanced.wav",
    "fantine": "hf://kyutai/tts-voices/vctk/p244_023_enhanced.wav",
    "charles": "hf://kyutai/tts-voices/vctk/p254_023_enhanced.wav",
    "paul": "hf://kyutai/tts-voices/vctk/p259_023_enhanced.wav",
    "eponine": "hf://kyutai/tts-voices/vctk/p262_023_enhanced.wav",
    "azelma": "hf://kyutai/tts-voices/vctk/p303_023_enhanced.wav",
    "george": "hf://kyutai/tts-voices/vctk/p315_023_enhanced.wav",
    "mary": "hf://kyutai/tts-voices/vctk/p333_023_enhanced.wav",
    "jane": "hf://kyutai/tts-voices/vctk/p339_023_enhanced.wav",
    "michael": "hf://kyutai/tts-voices/vctk/p360_023_enhanced.wav",
    "eve": "hf://kyutai/tts-voices/vctk/p361_023_enhanced.wav",
    "bill_boerst": "hf://kyutai/tts-voices/voice-zero/bill_boerst.wav",
    "peter_yearsley": "hf://kyutai/tts-voices/voice-zero/peter_yearsley.wav",
    "stuart_bell": "hf://kyutai/tts-voices/voice-zero/stuart_bell.wav",
    "caro_davy": "hf://kyutai/tts-voices/voice-zero/caro_davy.wav",
}


def get_predefined_voice(language: str, name: str) -> str:
    return f"hf://kyutai/pocket-tts/languages/{language}/embeddings/{name}.safetensors@25443bdc51dd9d08cca86e0f080d25e8a05c39b3"


def make_cache_directory() -> Path:
    cache_dir = Path.home() / ".cache" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def print_nb_parameters(model: nn.Module, model_name: str):
    logger = logging.getLogger(__name__)
    state_dict = model.state_dict()
    total = 0
    for key, value in state_dict.items():
        logger.info("%s: %,d", key, value.numel())
        total += value.numel()
    logger.info("Total number of parameters in %s: %,d", model_name, total)


def size_of_dict(state_dict: dict) -> int:
    total_size = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()
        elif isinstance(value, dict):
            total_size += size_of_dict(value)
    return total_size


class display_execution_time:
    def __init__(self, task_name: str, print_output: bool = True):
        self.task_name = task_name
        self.print_output = print_output
        self.start_time = None
        self.elapsed_time_ms = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.monotonic()
        self.elapsed_time_ms = int((end_time - self.start_time) * 1000)
        if self.print_output:
            self.logger.info("%s took %d ms", self.task_name, self.elapsed_time_ms)
        return False  # Don't suppress exceptions


def download_if_necessary(file_path: str) -> Path:
    if file_path.startswith("http://") or file_path.startswith("https://"):
        cache_dir = make_cache_directory()
        cached_file = cache_dir / (
            hashlib.sha256(file_path.encode()).hexdigest() + "." + file_path.split(".")[-1]
        )
        if not cached_file.exists():
            response = requests.get(file_path)
            response.raise_for_status()
            with open(cached_file, "wb") as f:
                f.write(response.content)
        return cached_file
    elif file_path.startswith("hf://"):
        file_path = file_path.removeprefix("hf://")
        splitted = file_path.split("/")
        repo_id = "/".join(splitted[:2])
        filename = "/".join(splitted[2:])
        if "@" in filename:
            filename, revision = filename.split("@")
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    else:
        return Path(file_path)
