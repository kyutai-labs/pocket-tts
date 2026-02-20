from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_color=False))

from pocket_tts.models.tts_model import (  # noqa: E402
    TTSModel,
    export_model_state,
)

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.get_state_for_audio_prompt

__all__ = ["TTSModel", "export_model_state"]
