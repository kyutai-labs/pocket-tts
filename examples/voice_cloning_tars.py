from pathlib import Path

import scipy.io.wavfile

from pocket_tts import TTSModel

VOICE_WAV = Path("assets/voice-references/tars/tars-voice-sample-01.wav")
TRANSCRIPT = Path("assets/voice-references/tars/tars-voice-sample-01.txt")
OUTPUT_WAV = Path("output/tars-voice-clone.wav")


def main() -> None:
    text = TRANSCRIPT.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Transcript is empty.")

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(VOICE_WAV)
    audio = model.generate_audio(voice_state, text)

    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(OUTPUT_WAV, model.sample_rate, audio.numpy())
    print(f"Wrote {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
