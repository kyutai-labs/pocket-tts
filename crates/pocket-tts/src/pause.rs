//! Pause/silence handling for text-to-speech
//!
//! Supports:
//! - Explicit pause markers: `[pause:Xms]` or `[pause:Xs]`
//! - Natural pauses from punctuation: `...`, `,`

use regex::Regex;
use std::sync::LazyLock;

/// Pause marker found in text
#[derive(Debug, Clone, PartialEq)]
pub struct PauseMarker {
    /// Original text that was matched
    pub original: String,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Position in the original text (byte offset)
    pub position: usize,
}

/// Default pause durations (in milliseconds) for punctuation
pub mod defaults {
    /// Ellipsis "..." pause duration
    pub const ELLIPSIS_MS: u32 = 500;
    /// Comma pause duration
    pub const COMMA_MS: u32 = 200;
    /// Period/sentence end pause duration
    pub const PERIOD_MS: u32 = 400;
    /// Semicolon pause duration
    pub const SEMICOLON_MS: u32 = 300;
}

// Regex patterns for pause parsing
static EXPLICIT_PAUSE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // Matches [pause:500ms] or [pause:1s] or [pause:1.5s]
    Regex::new(r"\[pause:(\d+(?:\.\d+)?)(ms|s)\]").unwrap()
});

static ELLIPSIS_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\.{3,}").unwrap());

/// Parse explicit pause markers from text
///
/// # Example
/// ```
/// use pocket_tts::pause::parse_explicit_pauses;
///
/// let pauses = parse_explicit_pauses("Hello [pause:500ms] world [pause:1s] done");
/// assert_eq!(pauses.len(), 2);
/// assert_eq!(pauses[0].duration_ms, 500);
/// assert_eq!(pauses[1].duration_ms, 1000);
/// ```
pub fn parse_explicit_pauses(text: &str) -> Vec<PauseMarker> {
    EXPLICIT_PAUSE_REGEX
        .captures_iter(text)
        .filter_map(|cap| {
            let full_match = cap.get(0)?;
            let value: f64 = cap.get(1)?.as_str().parse().ok()?;
            let unit = cap.get(2)?.as_str();

            let duration_ms = match unit {
                "ms" => value as u32,
                "s" => (value * 1000.0) as u32,
                _ => return None,
            };

            Some(PauseMarker {
                original: full_match.as_str().to_string(),
                duration_ms,
                position: full_match.start(),
            })
        })
        .collect()
}

/// Parse natural pauses from punctuation
pub fn parse_natural_pauses(text: &str) -> Vec<PauseMarker> {
    let mut pauses = Vec::new();

    // Find ellipses
    for cap in ELLIPSIS_REGEX.find_iter(text) {
        pauses.push(PauseMarker {
            original: cap.as_str().to_string(),
            duration_ms: defaults::ELLIPSIS_MS,
            position: cap.start(),
        });
    }

    // Find commas (but not inside numbers like "1,000")
    for (i, c) in text.char_indices() {
        if c == ',' {
            // Check if it's not surrounded by digits
            let prev_is_digit =
                i > 0 && text[..i].chars().last().is_some_and(|c| c.is_ascii_digit());
            let next_is_digit = text[(i + 1)..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit());

            if !prev_is_digit || !next_is_digit {
                pauses.push(PauseMarker {
                    original: ",".to_string(),
                    duration_ms: defaults::COMMA_MS,
                    position: i,
                });
            }
        }
    }

    // Sort by position
    pauses.sort_by_key(|p| p.position);
    pauses
}

/// Remove pause markers from text, returning clean text for TTS
pub fn strip_pause_markers(text: &str) -> String {
    EXPLICIT_PAUSE_REGEX.replace_all(text, " ").to_string()
}

/// Parsed text with pause information
#[derive(Debug, Clone)]
pub struct ParsedText {
    /// Text with pause markers removed
    pub clean_text: String,
    /// All pause markers (explicit + natural) with adjusted positions
    pub pauses: Vec<PauseMarker>,
}

/// Parse text for all pause markers (explicit and natural)
pub fn parse_text_with_pauses(text: &str) -> ParsedText {
    // First, find explicit pauses in original text
    let mut all_pauses = parse_explicit_pauses(text);

    // Strip explicit markers to get clean text
    let clean_text = strip_pause_markers(text);

    // Find natural pauses in clean text
    let natural_pauses = parse_natural_pauses(&clean_text);

    // Note: positions in all_pauses are relative to original text
    // We need to adjust them to the clean text
    // For simplicity, we'll recalculate based on clean text positions

    // Clear and rebuild with correct positions
    all_pauses.clear();
    all_pauses.extend(natural_pauses);

    // Re-parse explicit pauses and calculate where they would be in clean text
    let mut offset = 0;
    for cap in EXPLICIT_PAUSE_REGEX.captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let original_pos = full_match.start();
        let adjusted_pos = original_pos.saturating_sub(offset);
        let value: f64 = cap.get(1).unwrap().as_str().parse().unwrap_or(0.0);
        let unit = cap.get(2).unwrap().as_str();

        let duration_ms = match unit {
            "ms" => value as u32,
            "s" => (value * 1000.0) as u32,
            _ => 0,
        };

        if duration_ms > 0 {
            all_pauses.push(PauseMarker {
                original: full_match.as_str().to_string(),
                duration_ms,
                position: adjusted_pos,
            });
        }

        offset += full_match.len() - 1; // -1 for the space we replace with
    }

    // Sort by position
    all_pauses.sort_by_key(|p| p.position);

    ParsedText {
        clean_text,
        pauses: all_pauses,
    }
}

/// Calculate the number of silence samples for a given duration
pub fn silence_samples(duration_ms: u32, sample_rate: u32) -> usize {
    ((duration_ms as u64 * sample_rate as u64) / 1000) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_explicit_pause_ms() {
        let pauses = parse_explicit_pauses("Hello [pause:500ms] world");
        assert_eq!(pauses.len(), 1);
        assert_eq!(pauses[0].duration_ms, 500);
        assert_eq!(pauses[0].original, "[pause:500ms]");
    }

    #[test]
    fn test_parse_explicit_pause_seconds() {
        let pauses = parse_explicit_pauses("Test [pause:1s] and [pause:1.5s]");
        assert_eq!(pauses.len(), 2);
        assert_eq!(pauses[0].duration_ms, 1000);
        assert_eq!(pauses[1].duration_ms, 1500);
    }

    #[test]
    fn test_parse_ellipsis() {
        let pauses = parse_natural_pauses("Hello... world");
        assert_eq!(pauses.len(), 1);
        assert_eq!(pauses[0].duration_ms, defaults::ELLIPSIS_MS);
    }

    #[test]
    fn test_parse_comma() {
        let pauses = parse_natural_pauses("Hello, world");
        assert_eq!(pauses.len(), 1);
        assert_eq!(pauses[0].duration_ms, defaults::COMMA_MS);
    }

    #[test]
    fn test_comma_in_number_ignored() {
        let pauses = parse_natural_pauses("That costs 1,000 dollars");
        // The comma in 1,000 should be ignored
        assert_eq!(pauses.len(), 0);
    }

    #[test]
    fn test_strip_pause_markers() {
        let clean = strip_pause_markers("Hello [pause:500ms] world [pause:1s] done");
        assert_eq!(clean, "Hello   world   done");
    }

    #[test]
    fn test_parse_text_with_pauses() {
        let parsed = parse_text_with_pauses("Hello... [pause:500ms] world, done");
        assert_eq!(parsed.clean_text, "Hello...   world, done");
        // Should have: ellipsis, explicit pause, comma
        assert_eq!(parsed.pauses.len(), 3);
    }

    #[test]
    fn test_silence_samples() {
        // 500ms at 24kHz = 12000 samples
        assert_eq!(silence_samples(500, 24000), 12000);
        // 1s at 24kHz = 24000 samples
        assert_eq!(silence_samples(1000, 24000), 24000);
    }
}
