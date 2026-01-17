"""We also add the imports in the functions to make sure we didn't forget them."""
# ruff: noqa: F841

import pytest


def test_readme_example():
    import scipy.io.wavfile

    from pocket_tts import TTSModel

    tts_model = TTSModel.load_model()
    voice_state = tts_model.get_state_for_audio_prompt("cosette")
    audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
    # Audio is a torch tensor containing PCM data.
    scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())


def test_quick_start():
    import scipy.io.wavfile

    from pocket_tts import TTSModel

    # Load the model
    tts_model = TTSModel.load_model()

    # Get voice state from an audio file
    voice_state = tts_model.get_state_for_audio_prompt("marius")

    # Generate audio
    audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")

    # Save to file
    scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())


def test_load_model():
    from pocket_tts import TTSModel

    # Load with default settings
    model = TTSModel.load_model()

    # Load with custom parameters
    model = TTSModel.load_model(
        variant="b6369a24", temp=0.5, lsd_decode_steps=5, eos_threshold=-3.0
    )


def test_device():
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    print(f"Model running on: {model.device}")


def test_sample_rate():
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    print(f"Sample rate: {model.sample_rate} Hz")


@pytest.fixture
def make_my_voice_file():
    import requests

    url = "https://huggingface.co/kyutai/tts-voices/resolve/main/expresso/ex01-ex02_default_001_channel1_168s.wav"
    response = requests.get(url)
    with open("my_voice.wav", "wb") as f:
        f.write(response.content)


@pytest.mark.usefixtures("make_my_voice_file")
def test_get_state_for_audio_prompt():
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    # hack to make it work without auth
    model.has_voice_cloning = True

    # From HuggingFace URL
    voice_state = model.get_state_for_audio_prompt(
        "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    )

    # From local file
    voice_state = model.get_state_for_audio_prompt("./my_voice.wav")

    # From HTTP URL
    voice_state = model.get_state_for_audio_prompt(
        "https://huggingface.co/kyutai/tts-voices/resolve"
        "/main/expresso/ex01-ex02_default_001_channel1_168s.wav"
    )


def test_generate_audio():
    from pocket_tts import TTSModel

    model = TTSModel.load_model()

    voice_state = model.get_state_for_audio_prompt("marius")

    # Generate audio
    audio = model.generate_audio(voice_state, "Hello world!", frames_after_eos=2, copy_state=True)

    print(f"Generated audio shape: {audio.shape}")
    print(f"Audio duration: {audio.shape[-1] / model.sample_rate:.2f} seconds")


def test_generate_audio_stream():
    from pocket_tts import TTSModel

    model = TTSModel.load_model()

    voice_state = model.get_state_for_audio_prompt("fantine")
    # Stream generation
    for chunk in model.generate_audio_stream(voice_state, "Long text content..."):
        # Process each chunk as it's generated
        print(f"Generated chunk: {chunk.shape[0]} samples")
        # Could save chunks to file or play in real-time


def test_voice_management():
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    # Preload multiple voices
    voices = {
        "casual": model.get_state_for_audio_prompt("alba"),
        "serious": model.get_state_for_audio_prompt("marius"),
    }

    # Generate with different voices
    casual_audio = model.generate_audio(voices["casual"], "Hey there!")
    funny_audio = model.generate_audio(voices["serious"], "Good morning.")


def test_batch_processing():
    import scipy.io.wavfile
    import torch

    from pocket_tts import TTSModel

    model = TTSModel.load_model()

    voice_state = model.get_state_for_audio_prompt("azelma")
    # Process multiple texts efficiently by re-using the same voice state
    texts = [
        "First sentence to generate.",
        "Second sentence to generate.",
        "Third sentence to generate.",
    ]

    audios = []
    for text in texts:
        audio = model.generate_audio(voice_state, text)
        audios.append(audio)

    # Concatenate all audio
    full_audio = torch.cat(audios, dim=0)
    scipy.io.wavfile.write("batch_output.wav", model.sample_rate, full_audio.numpy())
