//! Port of `pocket_tts/default_parameters.py` and of the voice catalog in `pocket_tts/utils/utils.py`.

pub const DEFAULT_LANGUAGE: &str = "english";
pub const DEFAULT_SAMPLER_DECODE_STEPS: usize = 1;
pub const DEFAULT_EOS_THRESHOLD: f32 = -4.0;
pub const MAX_TOKEN_PER_CHUNK: usize = 50;

const DEFAULT_TEXT_FOR_LANGUAGE: &[(&str, &str)] = &[
    (
        "english",
        "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    ),
    (
        "french",
        "Bonjour le monde. Je suis le TTS de poche de Kyutai. Je suis assez rapide pour fonctionner sur de petits CPU. J'espère que vous m'aimerez.",
    ),
    (
        "german",
        "Hallo Welt. Ich bin Pocket TTS von Kyutai. Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. Ich hoffe, ich gefalle dir.",
    ),
    (
        "portuguese",
        "Olá mundo. Eu sou o Pocket TTS da Kyutai. Sou rápido o suficiente para rodar em CPUs pequenas. Espero que você goste de mim.",
    ),
    (
        "italian",
        "Ciao mondo. Sono il Pocket TTS di Kyutai. Sono abbastanza veloce da funzionare su piccole CPU. Spero che ti piacerò.",
    ),
    (
        "dutch",
        "Hallo wereld. Ik ben Pocket TTS van Kyutai. Ik ben snel genoeg om op kleine CPU's te draaien. Ik hoop dat je me leuk vindt.",
    ),
    (
        "spanish",
        "Hola mundo. Soy el Pocket TTS de Kyutai. Soy lo suficientemente rápido para funcionar en pequeñas CPU. Espero que te guste.",
    ),
];

const DEFAULT_VOICE_FOR_LANGUAGE: &[(&str, &str)] = &[
    ("italian", "giovanni"),
    ("spanish", "lola"),
    ("german", "juergen"),
    ("portuguese", "rafael"),
    ("french", "estelle"),
    ("dutch", "daan"),
];
const DEFAULT_VOICE_FALLBACK: &str = "alba";

/// The audio each predefined voice was computed from.
pub const ORIGINS_OF_PREDEFINED_VOICES: &[(&str, &str)] = &[
    (
        "cosette",
        "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    ),
    (
        "marius",
        "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    ),
    (
        "javert",
        "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    ),
    ("alba", "hf://kyutai/tts-voices/alba-mackenna/casual.wav"),
    (
        "jean",
        "hf://kyutai/tts-voices/ears/p010/freeform_speech_01_enhanced.wav",
    ),
    ("anna", "hf://kyutai/tts-voices/vctk/p228_023_enhanced.wav"),
    ("vera", "hf://kyutai/tts-voices/vctk/p229_023_enhanced.wav"),
    (
        "fantine",
        "hf://kyutai/tts-voices/vctk/p244_023_enhanced.wav",
    ),
    (
        "charles",
        "hf://kyutai/tts-voices/vctk/p254_023_enhanced.wav",
    ),
    ("paul", "hf://kyutai/tts-voices/vctk/p259_023_enhanced.wav"),
    (
        "eponine",
        "hf://kyutai/tts-voices/vctk/p262_023_enhanced.wav",
    ),
    (
        "azelma",
        "hf://kyutai/tts-voices/vctk/p303_023_enhanced.wav",
    ),
    (
        "george",
        "hf://kyutai/tts-voices/vctk/p315_023_enhanced.wav",
    ),
    ("mary", "hf://kyutai/tts-voices/vctk/p333_023_enhanced.wav"),
    ("jane", "hf://kyutai/tts-voices/vctk/p339_023_enhanced.wav"),
    (
        "michael",
        "hf://kyutai/tts-voices/vctk/p360_023_enhanced.wav",
    ),
    ("eve", "hf://kyutai/tts-voices/vctk/p361_023_enhanced.wav"),
    (
        "bill_boerst",
        "hf://kyutai/tts-voices/voice-zero/bill_boerst.wav",
    ),
    (
        "peter_yearsley",
        "hf://kyutai/tts-voices/voice-zero/peter_yearsley.wav",
    ),
    (
        "stuart_bell",
        "hf://kyutai/tts-voices/voice-zero/stuart_bell.wav",
    ),
    (
        "caro_davy",
        "hf://kyutai/tts-voices/voice-zero/caro_davy.wav",
    ),
    (
        "giovanni",
        "hf://kyutai/pocket-tts/common_voice_it_36520747-enhanced-v2.mp3@64ab7d24c479d736a83b8cc666c4a776fca30fda",
    ),
    (
        "lola",
        "hf://kyutai/pocket-tts/common_voice_es_19762977-enhanced-v2.mp3@64ab7d24c479d736a83b8cc666c4a776fca30fda",
    ),
    (
        "juergen",
        "hf://kyutai/pocket-tts/de-DE-juergen.mp3@64ab7d24c479d736a83b8cc666c4a776fca30fda",
    ),
    (
        "rafael",
        "hf://kyutai/pocket-tts/g-Vi8PgmSY0-enhanced-v2.wav@64ab7d24c479d736a83b8cc666c4a776fca30fda",
    ),
    (
        "daan",
        "hf://kyutai/pocket-tts/cml_tts_nl_11290_9412_000030_0001.flac@2dd944b099d06bb9edbd221fef138685711d9bf0",
    ),
    (
        "estelle",
        "hf://kyutai/tts-voices/unmute-prod-website/developpeuse-3.wav@1fc7395b7e012e2bbebfca14b942a4ef62ccc899",
    ),
];

pub fn is_predefined_voice(name: &str) -> bool {
    ORIGINS_OF_PREDEFINED_VOICES
        .iter()
        .any(|(voice, _)| *voice == name)
}

pub fn predefined_voice_names() -> Vec<&'static str> {
    ORIGINS_OF_PREDEFINED_VOICES
        .iter()
        .map(|(name, _)| *name)
        .collect()
}

/// The state of a predefined voice, precomputed with the released weights of `language`.
pub fn get_predefined_voice(language: &str, name: &str) -> String {
    format!(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/{language}/embeddings/{name}.safetensors@a3055dcb533cf2ec90458008a01e7cef550bae2a"
    )
}

pub fn get_default_text_for_language(language: Option<&str>) -> &'static str {
    for (key, text) in DEFAULT_TEXT_FOR_LANGUAGE {
        if language.is_some_and(|language| language.contains(key)) {
            return text;
        }
    }
    DEFAULT_TEXT_FOR_LANGUAGE[0].1
}

/// The voice to use when the user didn't pick one.
///
/// `config` and `checkpoint` both mean custom weights, which cannot use the predefined voices,
/// hence the audio file instead of the voice name.
pub fn get_default_voice_for_language(language: Option<&str>, custom_weights: bool) -> String {
    if custom_weights {
        return ORIGINS_OF_PREDEFINED_VOICES
            .iter()
            .find(|(name, _)| *name == DEFAULT_VOICE_FALLBACK)
            .map(|(_, origin)| origin.to_string())
            .unwrap();
    }
    for (key, voice) in DEFAULT_VOICE_FOR_LANGUAGE {
        if language.is_some_and(|language| language.contains(key)) {
            return voice.to_string();
        }
    }
    DEFAULT_VOICE_FALLBACK.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_language_substring() {
        assert!(get_default_text_for_language(Some("french_24l")).starts_with("Bonjour"));
        assert!(get_default_text_for_language(None).starts_with("Hello"));
        assert_eq!(
            get_default_voice_for_language(Some("italian_24l"), false),
            "giovanni"
        );
        assert_eq!(
            get_default_voice_for_language(Some("english_2026-04"), false),
            "alba"
        );
        assert_eq!(
            get_default_voice_for_language(Some("italian"), true),
            "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
        );
    }
}
