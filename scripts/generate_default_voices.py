from pathlib import Path

from pocket_tts import TTSModel, export_model_state
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

languages = [x.stem for x in Path("./pocket_tts/config").glob("*.yaml")]


for language in languages:
    model = TTSModel.load_model(language=language)

    for voice_name, voice_origin in _ORIGINS_OF_PREDEFINED_VOICES.items():
        print(
            f"Processing voice: {voice_name} from origin: {voice_origin} for language: {language}"
        )
        # Export a voice state for fast loading later
        model_state = model.get_state_for_audio_prompt(voice_origin)
        dest = f"/projects/huggingface/pocket-tts/languages/{language}/embeddings/{voice_name}.safetensors"
        export_model_state(model_state, dest)

        model_state_copy = model.get_state_for_audio_prompt(dest)

        # audio = model.generate_audio(
        #    model_state_copy,
        #    "An Sommernachmittagen spielten die Kinder fröhlich auf dem Platz der kleinen Stadt.",
        # )
        # scipy.io.wavfile.write(
        #    f"./built-in-voices-generated/{voice_name}.wav", model.sample_rate, audio.numpy()
        # )
