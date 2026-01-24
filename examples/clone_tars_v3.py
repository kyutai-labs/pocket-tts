"""
Robust voice cloning script for TARS (V3 - High Quality).
Optimized parameters for better quality.
"""

import logging
from pathlib import Path

import scipy.io.wavfile
import torch

from pocket_tts import TTSModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
VOICE_WAV = Path("assets/voice-references/tars/tars-voice-sample-01.wav")
TRANSCRIPT = Path("assets/voice-references/tars/tars-voice-sample-01.txt")
OUTPUT_WAV = Path("output/tars-voice-clone-v3.wav")

# High Quality Parameters
PARAMS = {
    "variant": "b6369a24",  # Default variant
    "temp": 0.4,  # Lower temp for more stable/focused generation (reduced from 0.7)
    "lsd_decode_steps": 10,  # More decoding steps for higher quality (increased from 1)
    "eos_threshold": -4.0,  # Default
    "noise_clamp": None,
}


def main() -> None:
    logger.info("Starting TARS Voice Clone V3 (High Quality)")

    # 1. Read Transcript
    if not TRANSCRIPT.exists():
        raise FileNotFoundError(f"Transcript not found at {TRANSCRIPT}")

    text = TRANSCRIPT.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Transcript is empty.")

    logger.info(f"Transcript: {text}")

    # 2. Load Model with Custom Params
    logger.info(f"Loading TTS Model with params: {PARAMS}")
    try:
        model = TTSModel.load_model(**PARAMS)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(
        f"Model loaded. Device: {model.device}, Sample Rate: {model.sample_rate}"
    )

    # 3. Process Voice Prompt
    if not VOICE_WAV.exists():
        raise FileNotFoundError(f"Voice sample not found at {VOICE_WAV}")

    logger.info(f"Processing voice sample: {VOICE_WAV}")
    try:
        voice_state = model.get_state_for_audio_prompt(VOICE_WAV, truncate=True)
    except Exception as e:
        logger.error(f"Failed to process voice prompt: {e}")
        return

    # 4. Generate Audio
    logger.info("Generating audio...")
    try:
        # Generate full audio tensor
        audio = model.generate_audio(
            voice_state,
            text,
            frames_after_eos=None,  # Let model decide
            copy_state=True,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return

    logger.info(
        f"Generated Audio Info: Shape={audio.shape}, Min={audio.min():.4f}, Max={audio.max():.4f}"
    )

    # 5. Save as 16-bit PCM for compatibility
    logger.info(f"Saving to {OUTPUT_WAV}...")
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)

    # Clamp and convert to int16
    audio_clamped = audio.clamp(-1, 1)
    audio_int16 = (audio_clamped * 32767).to(torch.int16).cpu().numpy()

    scipy.io.wavfile.write(OUTPUT_WAV, model.sample_rate, audio_int16)
    logger.info("Success! Audio saved.")


if __name__ == "__main__":
    main()
