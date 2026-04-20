DEFAULT_AUDIO_PROMPT = "alba"
DEFAULT_LANGUAGE = "english"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_LSD_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = None
DEFAULT_EOS_THRESHOLD = -4.0
DEFAULT_FRAMES_AFTER_EOS = None
# TODO: make this dynamic since english_2026-04 supports bigger chunks
MAX_TOKEN_PER_CHUNK = 50

DEFAULT_TEXT_FOR_LANGUAGE = {
    "english": (
        "Hello world. I am Kyutai's Pocket TTS. "
        "I'm fast enough to run on small CPUs. "
        "I hope you'll like me."
    ),
    "french": (
        "Bonjour le monde. Je suis le TTS de poche de Kyutai. "
        "Je suis assez rapide pour fonctionner sur de petits CPU. "
        "J'espère que vous m'aimerez."
    ),
    "german": (
        "Hallo Welt. Ich bin Kyutais Pocket TTS. "
        "Ich bin schnell genug, um auf kleinen CPUs zu laufen. "
        "Ich hoffe, du wirst mich mögen."
    ),
    "portuguese": (
        "Olá mundo. Eu sou o Pocket TTS da Kyutai. "
        "Sou rápido o suficiente para rodar em CPUs pequenas. "
        "Espero que você goste de mim."
    ),
    "italian": (
        "Ciao mondo. Sono il Pocket TTS di Kyutai. "
        "Sono abbastanza veloce da funzionare su piccole CPU. "
        "Spero che ti piacerò."
    ),
    "spanish": (
        "Hola mundo. Soy el Pocket TTS de Kyutai. "
        "Soy lo suficientemente rápido para funcionar en pequeñas CPU. "
        "Espero que te guste."
    ),
}


def get_default_text_for_language(language: str | None) -> str:
    for key, text in DEFAULT_TEXT_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return text
    return DEFAULT_TEXT_FOR_LANGUAGE[DEFAULT_LANGUAGE]
