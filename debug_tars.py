from pathlib import Path

import scipy.io.wavfile
import torch

from pocket_tts import TTSModel

VOICE_WAV = Path("assets/voice-references/tars/tars-voice-sample-01.wav")
TRANSCRIPT = Path("assets/voice-references/tars/tars-voice-sample-01.txt")
OUTPUT_WAV = Path("output/debug-tars.wav")


def main() -> None:
    print("Reading transcript...")
    text = TRANSCRIPT.read_text(encoding="utf-8").strip()

    print("Loading model...")
    model = TTSModel.load_model()

    print("Getting voice state...")
    voice_state = model.get_state_for_audio_prompt(VOICE_WAV)

    print("Generating audio...")
    audio = model.generate_audio(voice_state, text)

    print(f"Audio Shape: {audio.shape}")
    print(f"Audio Min: {audio.min().item()}")
    print(f"Audio Max: {audio.max().item()}")
    print(f"Audio Mean: {audio.mean().item()}")
    print(f"Audio Std: {audio.std().item()}")
    print(f"Has NaN: {torch.isnan(audio).any().item()}")
    print(f"Has Inf: {torch.isinf(audio).any().item()}")

    if torch.isnan(audio).any():
        print("CRITICAL: Output contains NaNs!")
        return

    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(OUTPUT_WAV, model.sample_rate, audio.numpy())
    print(f"Wrote {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
