import sys

import scipy.io.wavfile

from pocket_tts import TTSModel, export_model_state
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

model = TTSModel.load_model(language=sys.argv[1])

for voice_name, voice_origin in _ORIGINS_OF_PREDEFINED_VOICES.items():
    print(f"Processing voice: {voice_name} from origin: {voice_origin}")
    # Export a voice state for fast loading later
    model_state = model.get_state_for_audio_prompt(voice_origin)
    export_model_state(model_state, f"./built-in-voices/{voice_name}.safetensors")

    model_state_copy = model.get_state_for_audio_prompt(
        f"./built-in-voices/{voice_name}.safetensors"
    )

    audio = model.generate_audio(
        model_state_copy,
        "An Sommernachmittagen spielten die Kinder fröhlich auf dem Platz der kleinen Stadt.",
    )
    scipy.io.wavfile.write(
        f"./built-in-voices-generated/{voice_name}.wav", model.sample_rate, audio.numpy()
    )
