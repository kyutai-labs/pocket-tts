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

/// The languages a model is published for.
///
/// Derived from the table above rather than written out again, so there is one list to maintain
/// and not two. English is appended because it is the fallback of
/// [`get_default_voice_for_language`] and so never needed a row of its own.
pub fn language_names() -> Vec<&'static str> {
    let mut names = vec![DEFAULT_LANGUAGE];
    names.extend(
        DEFAULT_VOICE_FOR_LANGUAGE
            .iter()
            .map(|(language, _)| *language),
    );
    names
}

pub fn is_predefined_voice(name: &str) -> bool {
    crate::voices::is_predefined(name)
}

pub fn predefined_voice_names() -> Vec<&'static str> {
    crate::voices::names()
}

/// The state of a predefined voice, precomputed with the released weights of `language`.
///
/// `None` when the catalogue does not have `name`; callers reach this through
/// [`is_predefined_voice`], so that is a programming error rather than a user one.
pub fn get_predefined_voice(language: &str, name: &str) -> Option<String> {
    crate::voices::find(name).map(|voice| voice.embedding_for(language))
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
        // The precomputed embeddings are per language, so custom weights get the original
        // recording instead and recompute from it.
        return crate::voices::find(DEFAULT_VOICE_FALLBACK)
            .and_then(|voice| voice.source.clone())
            .expect("the fallback voice must be in the catalogue, with its source recording");
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
