import os

# POCKET_TTS_NO_BEARTYPE=1 disables runtime type-checking: the beartype claw
# wraps every function in the package, which both adds per-call overhead in
# hot training loops and blocks torch.compile (dynamo cannot trace the
# generated wrappers). Inference default is unchanged.
if os.environ.get("POCKET_TTS_NO_BEARTYPE") != "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # is_pep484_tower: accept int where float is annotated, as PEP 484 says.
    # Without it an `int | float` hint is not redundant and ruff's PYI041
    # advice to collapse it would break callers passing ints.
    beartype_this_package(conf=BeartypeConf(is_color=False, is_pep484_tower=True))

from pocket_tts.models.model_state import export_model_state
from pocket_tts.models.tts_model import (  # noqa: E402
    TTSModel,
)

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.get_state_for_audio_prompt

__all__ = ["TTSModel", "export_model_state"]
