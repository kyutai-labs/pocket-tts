//! Port of `pocket_tts/models/text_chunking.py`: turning a text prompt into the chunks the model
//! generates one at a time.
//!
//! The model is trained on single sentences, so long inputs are split on sentence boundaries and
//! regrouped into chunks that fit `max_tokens`.

use anyhow::{Result, bail};
use std::collections::{BTreeMap, HashSet};

pub trait TextTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
}

impl TextTokenizer for crate::tokenizer::Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        crate::tokenizer::Tokenizer::encode(self, text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        crate::tokenizer::Tokenizer::decode(self, tokens)
    }
}

/// The text options of a model config.
#[derive(Debug, Clone)]
pub struct TextOptions {
    pub pad_with_spaces_for_short_inputs: bool,
    pub remove_semicolons: bool,
    pub append_terminal_punctuation: bool,
    pub capitalize_first_letter: bool,
    pub replace_characters: BTreeMap<char, String>,
}

impl TextOptions {
    pub fn from_config(config: &crate::config::Config) -> Self {
        let replace_characters = config
            .replace_characters
            .iter()
            .map(|(key, value)| {
                (
                    key.chars().next().unwrap(),
                    value.clone().unwrap_or_default(),
                )
            })
            .collect();
        Self {
            pad_with_spaces_for_short_inputs: config.pad_with_spaces_for_short_inputs,
            remove_semicolons: config.remove_semicolons,
            append_terminal_punctuation: config.append_terminal_punctuation,
            capitalize_first_letter: config.capitalize_first_letter,
            replace_characters,
        }
    }
}

/// Python's `" ".join(text.split())`.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Python's `re.sub(r"([.!?…])\s*[,;:]", r"\1", text)`: deleted quotes leave
/// '"Hi?", she said' as 'Hi?, she said', which reads as a sentence end followed by a stray comma;
/// keep the sentence mark only.
fn drop_punctuation_after_sentence_end(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        out.push(c);
        i += 1;
        if matches!(c, '.' | '!' | '?' | '\u{2026}') {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && matches!(chars[j], ',' | ';' | ':') {
                i = j + 1;
            }
        }
    }
    out
}

/// Returns the prepared text and a guess of the frames to generate after the end of speech.
pub fn prepare_text_prompt(text: &str, options: &TextOptions) -> Result<(String, usize)> {
    let mut text = text.trim().to_string();
    if !options.replace_characters.is_empty() {
        let translated: String = text
            .chars()
            .map(|c| match options.replace_characters.get(&c) {
                Some(replacement) => replacement.clone(),
                None => c.to_string(),
            })
            .collect();
        text = drop_punctuation_after_sentence_end(&collapse_whitespace(&translated));
    }
    if text.is_empty() {
        bail!("Text prompt cannot be empty")
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");
    if options.remove_semicolons {
        text = text.replace(';', ",");
    }
    let number_of_words = text.split_whitespace().count();
    let frames_after_eos_guess = if number_of_words <= 4 { 3 } else { 1 };

    // Make sure it starts with an uppercase letter. Only meaningful for orthographies that have
    // case; see the config's capitalize_first_letter.
    if options.capitalize_first_letter {
        let mut chars = text.chars();
        let first = chars.next().unwrap();
        if !first.is_uppercase() {
            text = first.to_uppercase().chain(chars).collect();
        }
    }

    if options.append_terminal_punctuation {
        text = ensure_terminal_punctuation(&text);
    }

    // The model does not perform well when there are very few tokens, so we can add empty spaces
    // at the beginning to increase the token count.
    if options.pad_with_spaces_for_short_inputs && text.split_whitespace().count() < 5 {
        text = format!("{}{text}", " ".repeat(8));
    }

    Ok((text, frames_after_eos_guess))
}

const TERMINAL_PUNCTUATION: &str = ".!?\u{2026}";
const WEAK_PUNCTUATION: &str = ",;:-\u{2013}\u{2014}";
const CLOSERS: &str = "\"'\u{201d}\u{2019})]\u{bb}";

/// Make sure the prompt ends with sentence-final punctuation.
///
/// The model is trained on sentences that end with a period, a question mark or an exclamation
/// mark. Without one, the last word is often mispronounced or repeated. Text that already ends
/// with one, possibly followed by closing quotes or brackets, is left alone. A trailing comma,
/// colon or dash is replaced by a period. Anything else gets a period appended after the closing
/// quote or bracket.
fn ensure_terminal_punctuation(text: &str) -> String {
    let core = text.trim_end_matches(|c| CLOSERS.contains(c) || c == ' ');
    let closers = text[core.len()..].trim();
    match core.chars().last() {
        None => text.to_string(),
        Some(last) if TERMINAL_PUNCTUATION.contains(last) => text.to_string(),
        Some(last) if WEAK_PUNCTUATION.contains(last) => {
            let core = core.trim_end_matches(|c| WEAK_PUNCTUATION.contains(c) || c == ' ');
            format!("{core}.{closers}")
        }
        Some(_) => format!("{text}."),
    }
}

/// Whether `segment_start_idx` begins right after a decimal period.
fn is_decimal_period_boundary(
    tokens: &[u32],
    segment_start_idx: usize,
    tokenizer: &dyn TextTokenizer,
) -> Result<bool> {
    let prefix: Vec<char> = tokenizer
        .decode(&tokens[..segment_start_idx])?
        .chars()
        .collect();
    let suffix = tokenizer.decode(&tokens[segment_start_idx..])?;
    Ok(prefix.len() >= 2
        && prefix[prefix.len() - 1] == '.'
        && prefix[prefix.len() - 2].is_numeric()
        && suffix.chars().next().is_some_and(char::is_numeric))
}

/// Token indices where the text should be split. Each consecutive pair of the result delimits a
/// segment, the first is always 0 and the last `tokens.len()`.
fn find_boundary_indices(
    tokens: &[u32],
    boundary_tokens: &[u32],
    decimal_tokenizer: Option<&dyn TextTokenizer>,
) -> Result<Vec<usize>> {
    let boundary_set: HashSet<u32> = boundary_tokens.iter().copied().collect();
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (idx, token) in tokens.iter().enumerate() {
        if boundary_set.contains(token) {
            previous_was_boundary = true;
            continue;
        }
        if previous_was_boundary {
            let is_decimal = match decimal_tokenizer {
                Some(tokenizer) => is_decimal_period_boundary(tokens, idx, tokenizer)?,
                None => false,
            };
            if !is_decimal {
                indices.push(idx);
            }
        }
        previous_was_boundary = false;
    }
    indices.push(tokens.len());
    Ok(indices)
}

fn segments_from_boundaries(
    tokens: &[u32],
    boundaries: &[usize],
    tokenizer: &dyn TextTokenizer,
) -> Result<Vec<(usize, String)>> {
    boundaries
        .windows(2)
        .map(|w| Ok((w[1] - w[0], tokenizer.decode(&tokens[w[0]..w[1]])?)))
        .collect()
}

/// The tokens of `text` without the first one, which carries the leading space marker.
fn marker_tokens(tokenizer: &dyn TextTokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer.encode(text)?.into_iter().skip(1).collect())
}

pub fn split_into_best_sentences(
    tokenizer: &dyn TextTokenizer,
    text_to_generate: &str,
    max_tokens: usize,
    options: &TextOptions,
) -> Result<Vec<String>> {
    let (text, _) = prepare_text_prompt(text_to_generate, options)?;
    let text = text.trim();
    let tokens = tokenizer.encode(text)?;

    let end_of_sentence_tokens = marker_tokens(tokenizer, ".!...?")?;
    let sentence_boundaries =
        find_boundary_indices(&tokens, &end_of_sentence_tokens, Some(tokenizer))?;
    let sentences = segments_from_boundaries(&tokens, &sentence_boundaries, tokenizer)?;

    // Sub-split oversized sentences on commas, semicolons, and colons to prevent skipped words.
    let fallback_tokens = marker_tokens(tokenizer, ",;:")?;
    let mut refined_segments = Vec::new();
    for (nb_tokens, sentence) in sentences {
        if nb_tokens <= max_tokens {
            refined_segments.push((nb_tokens, sentence));
            continue;
        }
        let sub_tokens = tokenizer.encode(sentence.trim())?;
        let sub_boundaries = find_boundary_indices(&sub_tokens, &fallback_tokens, None)?;
        let sub_segments = segments_from_boundaries(&sub_tokens, &sub_boundaries, tokenizer)?;
        if sub_segments.len() > 1 {
            refined_segments.extend(sub_segments);
        } else {
            refined_segments.push((nb_tokens, sentence));
        }
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_nb_of_tokens = 0;
    for (nb_tokens, sentence) in refined_segments {
        if current_chunk.is_empty() {
            current_chunk = sentence;
            current_nb_of_tokens = nb_tokens;
            continue;
        }
        if current_nb_of_tokens + nb_tokens > max_tokens {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = sentence;
            current_nb_of_tokens = nb_tokens;
        } else {
            current_chunk.push(' ');
            current_chunk.push_str(&sentence);
            current_nb_of_tokens += nb_tokens;
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    for chunk in chunks.iter() {
        let nb_tokens = tokenizer.encode(chunk.trim())?.len();
        if nb_tokens > max_tokens {
            let preview: String = chunk.chars().take(50).collect();
            log::warn!(
                "Chunk has {nb_tokens} tokens (max {max_tokens}), generation may skip words: '{preview}...'"
            );
        }
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character, with a leading space marker like SentencePiece.
    struct CharTokenizer;

    impl TextTokenizer for CharTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(std::iter::once(0)
                .chain(text.chars().map(|c| c as u32))
                .collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens
                .iter()
                .filter(|&&t| t != 0)
                .map(|&t| char::from_u32(t).unwrap())
                .collect())
        }
    }

    fn options() -> TextOptions {
        TextOptions {
            pad_with_spaces_for_short_inputs: false,
            remove_semicolons: false,
            append_terminal_punctuation: true,
            capitalize_first_letter: true,
            replace_characters: BTreeMap::new(),
        }
    }

    #[test]
    fn prompts_are_prepared_like_python() {
        let o = options();
        assert_eq!(
            prepare_text_prompt("hello world", &o).unwrap(),
            ("Hello world.".into(), 3)
        );
        assert_eq!(
            prepare_text_prompt("  one two three four five  ", &o).unwrap(),
            ("One two three four five.".into(), 1)
        );
        assert_eq!(prepare_text_prompt("Wait,", &o).unwrap().0, "Wait.");
        assert_eq!(
            prepare_text_prompt("He said \"hi\"", &o).unwrap().0,
            "He said \"hi\"."
        );
        assert_eq!(
            prepare_text_prompt("Is it \"so?\"", &o).unwrap().0,
            "Is it \"so?\""
        );
        assert_eq!(prepare_text_prompt("über", &o).unwrap().0, "Über.");
        assert_eq!(prepare_text_prompt("a\nb   c", &o).unwrap().0, "A b  c.");
        assert!(prepare_text_prompt("   ", &o).is_err());

        let mut o = options();
        o.pad_with_spaces_for_short_inputs = true;
        o.remove_semicolons = true;
        assert_eq!(prepare_text_prompt("a; b", &o).unwrap().0, "        A, b.");

        let mut o = options();
        o.replace_characters = [('"', String::new()), ('«', String::new())].into();
        assert_eq!(
            prepare_text_prompt("\"Hi?\", she said", &o).unwrap().0,
            "Hi? she said."
        );
    }

    // The expected values are what the Python implementation returns with the same tokenizer,
    // whose decoded segments keep their leading space (hence the double spaces).
    #[test]
    fn sentences_are_grouped_up_to_max_tokens() {
        let o = options();
        let text = "First one. Second one! Third one?";
        let chunks = split_into_best_sentences(&CharTokenizer, text, 25, &o).unwrap();
        assert_eq!(chunks, ["First one.  Second one!", "Third one?"]);
        let chunks = split_into_best_sentences(&CharTokenizer, text, 1000, &o).unwrap();
        assert_eq!(chunks, ["First one.  Second one!  Third one?"]);
    }

    #[test]
    fn decimal_periods_do_not_split() {
        let chunks =
            split_into_best_sentences(&CharTokenizer, "It costs 3.5 euros. Yes.", 5, &options())
                .unwrap();
        assert_eq!(chunks, ["It costs 3.5 euros.", "Yes."]);
    }

    #[test]
    fn long_sentences_split_on_commas() {
        let chunks =
            split_into_best_sentences(&CharTokenizer, "aaaa, bbbb, cccc.", 8, &options()).unwrap();
        assert_eq!(chunks, ["Aaaa,", "bbbb,", "cccc."]);
    }
}
