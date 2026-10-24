//! From what an agent wrote to the chunks the engine is given.
//!
//! Two decisions live here and both change how the result sounds:
//!
//! * [`speakable`] removes what should not be read aloud: fenced code, HTML,
//!   and Markdown syntax. It keeps link text and the words inside inline code,
//!   because an agent's prose leans on them. Tables are announced, then read
//!   row by row under their column names, or only announced past four columns.
//! * [`chunk`] decides where one generation ends and the next begins. The model
//!   decides for itself when it has finished speaking and gets it wrong at the
//!   end of a generation that closes on very short sentences (measured in
//!   `android/docs/eleven-utterances.md`), so where a chunk ends is where the
//!   words are at risk. [`Chunking`] offers the two strategies in use: one
//!   sentence per generation, as speech-kit does, and ~200-character chunks
//!   with a trailing run of short sentences joined, as the Android app does.

/// A piece of text to synthesise, and the silence to leave after it.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    /// What the engine is given.
    pub speech: String,
    pub pause_after_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chunking {
    /// Every sentence is its own generation.
    Sentence,
    /// Sentences packed toward `target` characters, never past `max`. With
    /// `join_tail`, a trailing run of short sentences is joined by commas;
    /// without it the chunk is given as written, which exists to measure
    /// whether the join is still needed.
    Packed {
        target: usize,
        max: usize,
        join_tail: bool,
    },
}

impl Chunking {
    pub const PACKED_DEFAULT: Self = Self::Packed {
        target: 200,
        max: 400,
        join_tail: true,
    };
}

const PARAGRAPH_PAUSE_SECONDS: f32 = 0.45;

/// A sentence of this many words or fewer is "short" for [`join_short_tail`].
/// Two, as in `TextChunker.kt`: "It compiles. Yes." lost "Yes." in one run of
/// three, and nothing with a three-word sentence before the end lost anything.
const SHORT_SENTENCE_WORDS: usize = 2;

/// Markdown in, paragraphs of plain prose out, separated by blank lines, with
/// anything this adds of its own (a table's size) in English.
pub fn speakable(markdown: &str) -> String {
    speakable_in(markdown, "en")
}

/// [`speakable`], with what it adds in `language`, a BCP-47 tag of one of the
/// models' languages. Any other reads as English.
pub fn speakable_in(markdown: &str, language: &str) -> String {
    let phrases = Phrases::for_language(language);
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut fence: Option<String> = None;
    let flush = |current: &mut Vec<String>, paragraphs: &mut Vec<String>| {
        let joined = current.join(" ");
        let joined = joined.split_whitespace().collect::<Vec<_>>().join(" ");
        if joined.chars().any(char::is_alphanumeric) {
            paragraphs.push(joined);
        }
        current.clear();
    };
    let lines: Vec<&str> = markdown.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim();
        index += 1;
        if let Some(marker) = &fence {
            if line.starts_with(marker.as_str()) {
                fence = None;
            }
            continue;
        }
        if line.starts_with("```") || line.starts_with("~~~") {
            flush(&mut current, &mut paragraphs);
            fence = Some(line[..3].to_string());
            continue;
        }
        if line.is_empty() {
            flush(&mut current, &mut paragraphs);
            continue;
        }
        // A table: a header row, a delimiter row, then rows until one has no
        // pipe. Read as the app reads it (`MarkdownSpeech.kt`, issue #10).
        if line.contains('|')
            && lines
                .get(index)
                .is_some_and(|next| is_table_delimiter(next))
        {
            flush(&mut current, &mut paragraphs);
            let header = cells(line);
            let mut rows = Vec::new();
            index += 1;
            while let Some(row) = lines.get(index).map(|row| row.trim()) {
                if row.is_empty() || !row.contains('|') {
                    break;
                }
                rows.push(cells(row));
                index += 1;
            }
            paragraphs.extend(speak_table(&header, &rows, phrases));
            continue;
        }
        if line.starts_with('|') || is_rule(line) {
            flush(&mut current, &mut paragraphs);
            continue;
        }
        let heading = line.starts_with('#');
        let item = list_item_body(line);
        let body = item
            .unwrap_or(line)
            .trim_start_matches('#')
            .trim_start_matches('>')
            .trim();
        if heading || item.is_some() {
            // Headings and list items are read as their own sentences.
            flush(&mut current, &mut paragraphs);
            current.push(end_sentence(&inline(body)));
            flush(&mut current, &mut paragraphs);
        } else {
            current.push(inline(body));
        }
    }
    flush(&mut current, &mut paragraphs);
    paragraphs.join("\n\n")
}

/// The words a reading adds of its own, in one language. The same table is in
/// `MarkdownSpeech.kt`.
struct Phrases {
    table: &'static str,
    not_read: &'static str,
    row: &'static str,
    rows: &'static str,
    columns: &'static str,
    headed: &'static str,
    and: &'static str,
}

impl Phrases {
    const ENGLISH: Phrases = Phrases {
        table: "Table",
        not_read: "Table not read",
        row: "row",
        rows: "rows",
        columns: "columns",
        headed: "headed",
        and: "and",
    };

    fn for_language(language: &str) -> &'static Phrases {
        const OTHERS: &[(&str, Phrases)] = &[
            (
                "fr",
                Phrases {
                    table: "Tableau",
                    not_read: "Tableau non lu",
                    row: "ligne",
                    rows: "lignes",
                    columns: "colonnes",
                    headed: "avec les en-têtes",
                    and: "et",
                },
            ),
            (
                "de",
                Phrases {
                    table: "Tabelle",
                    not_read: "Tabelle nicht vorgelesen",
                    row: "Zeile",
                    rows: "Zeilen",
                    columns: "Spalten",
                    headed: "mit den Überschriften",
                    and: "und",
                },
            ),
            (
                "es",
                Phrases {
                    table: "Tabla",
                    not_read: "Tabla no leída",
                    row: "fila",
                    rows: "filas",
                    columns: "columnas",
                    headed: "con los encabezados",
                    and: "y",
                },
            ),
            (
                "it",
                Phrases {
                    table: "Tabella",
                    not_read: "Tabella non letta",
                    row: "riga",
                    rows: "righe",
                    columns: "colonne",
                    headed: "con le intestazioni",
                    and: "e",
                },
            ),
            (
                "pt",
                Phrases {
                    table: "Tabela",
                    not_read: "Tabela não lida",
                    row: "linha",
                    rows: "linhas",
                    columns: "colunas",
                    headed: "com os cabeçalhos",
                    and: "e",
                },
            ),
            (
                "nl",
                Phrases {
                    table: "Tabel",
                    not_read: "Tabel niet voorgelezen",
                    row: "rij",
                    rows: "rijen",
                    columns: "kolommen",
                    headed: "met de kopjes",
                    and: "en",
                },
            ),
        ];
        let primary = language.split(['-', '_']).next().unwrap_or("");
        OTHERS
            .iter()
            .find(|(tag, _)| tag.eq_ignore_ascii_case(primary))
            .map(|(_, phrases)| phrases)
            .unwrap_or(&Self::ENGLISH)
    }
}

/// The widest table read row by row, as in the app: each row is read with its
/// column names, so four columns is already a long sentence.
const MAX_READ_COLUMNS: usize = 4;

/// `|---|:--:|`: every cell two or more dashes, optionally colon-aligned.
fn is_table_delimiter(line: &str) -> bool {
    let line = line.trim();
    line.contains('|')
        && line.contains("--")
        && cells_raw(line).iter().all(|cell| {
            let dashes = cell.trim_start_matches(':').trim_end_matches(':');
            dashes.len() >= 2 && dashes.chars().all(|c| c == '-')
        })
}

fn cells_raw(line: &str) -> Vec<&str> {
    let row = line.trim();
    let row = row.strip_prefix('|').unwrap_or(row);
    let row = row.strip_suffix('|').unwrap_or(row);
    row.split('|').map(str::trim).collect()
}

fn cells(line: &str) -> Vec<String> {
    cells_raw(line)
        .into_iter()
        .map(|cell| inline(cell).trim().to_string())
        .collect()
}

/// What a table says, as a screen reader would put it: how big it is, then
/// each row as its first cell followed by each other cell under its column's
/// name, "alpha: before, one; after, two." Wider than [`MAX_READ_COLUMNS`], it
/// is announced with its column names and not read.
fn speak_table(header: &[String], rows: &[Vec<String>], phrases: &Phrases) -> Vec<String> {
    let columns = rows
        .iter()
        .map(Vec::len)
        .chain([header.len()])
        .max()
        .unwrap_or(0);
    let count = if rows.len() == 1 {
        format!("1 {}", phrases.row)
    } else {
        format!("{} {}", rows.len(), phrases.rows)
    };
    let names: Vec<&str> = header
        .iter()
        .map(String::as_str)
        .filter(|name| !name.is_empty())
        .collect();
    if columns > MAX_READ_COLUMNS {
        let headed = if names.is_empty() {
            String::new()
        } else {
            format!(", {} {}", phrases.headed, spoken_list(&names, phrases.and))
        };
        return vec![format!(
            "{}: {count}, {columns} {}{headed}.",
            phrases.not_read, phrases.columns
        )];
    }
    let mut spoken = vec![format!("{}, {count}.", phrases.table)];
    for row in rows {
        let label = row.first().map(String::as_str).unwrap_or("");
        let pairs: Vec<String> = (1..columns)
            .filter_map(|j| {
                let value = row.get(j).map(String::as_str).unwrap_or("");
                let name = header.get(j).map(String::as_str).unwrap_or("");
                match (value.is_empty(), name.is_empty()) {
                    (true, _) => None,
                    (false, true) => Some(value.to_string()),
                    (false, false) => Some(format!("{name}, {value}")),
                }
            })
            .collect();
        let text = match (label.is_empty(), pairs.is_empty()) {
            (true, true) => continue,
            (true, false) => pairs.join("; "),
            (false, true) => label.to_string(),
            (false, false) => format!("{label}: {}", pairs.join("; ")),
        };
        spoken.push(end_sentence(&text));
    }
    spoken
}

/// "a", "a and b", "a, b and c", with `and` in the reading's language.
fn spoken_list(items: &[&str], and: &str) -> String {
    match items.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} {and} {last}", rest.join(", ")),
        _ => items.join(""),
    }
}

fn is_rule(line: &str) -> bool {
    let compact: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    compact.len() >= 3
        && (compact.chars().all(|c| c == '-')
            || compact.chars().all(|c| c == '*')
            || compact.chars().all(|c| c == '_'))
}

fn list_item_body(line: &str) -> Option<&str> {
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(marker) {
            return Some(rest.trim_start_matches("[ ] ").trim_start_matches("[x] "));
        }
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 {
        let rest = &line[digits..];
        if let Some(rest) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some(rest);
        }
    }
    None
}

fn end_sentence(text: &str) -> String {
    let trimmed = text.trim_end();
    match trimmed.chars().last() {
        Some('.' | '!' | '?' | ':' | '…') => trimmed.to_string(),
        Some(_) => format!("{trimmed}."),
        None => String::new(),
    }
}

/// Inline Markdown: images dropped, links reduced to their text, code spans to
/// their contents, emphasis markers and HTML tags removed, bare URLs to their
/// host.
fn inline(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '!'
            && chars.get(i + 1) == Some(&'[')
            && let Some((_, end)) = link_at(&chars, i + 1)
        {
            i = end;
            continue;
        }
        if c == '['
            && let Some((label, end)) = link_at(&chars, i)
        {
            out.push_str(&inline(&label));
            i = end;
            continue;
        }
        if c == '`' {
            let ticks = chars[i..].iter().take_while(|&&t| t == '`').count();
            let open_end = i + ticks;
            let close = (open_end..chars.len())
                .find(|&j| chars[j..].iter().take_while(|&&t| t == '`').count() == ticks);
            if let Some(close) = close {
                out.extend(&chars[open_end..close]);
                i = close + ticks;
                continue;
            }
        }
        if c == '<'
            && let Some(close) = chars[i..].iter().position(|&t| t == '>')
        {
            let tag: String = chars[i + 1..i + close].iter().collect();
            if tag.starts_with("http") {
                out.push_str(&url_host(&tag));
                i += close + 1;
                continue;
            }
            if tag
                .chars()
                .next()
                .is_some_and(|t| t.is_ascii_alphabetic() || t == '/')
            {
                i += close + 1;
                continue;
            }
        }
        if starts_with(&chars, i, "http://") || starts_with(&chars, i, "https://") {
            let end = (i..chars.len())
                .find(|&j| chars[j].is_whitespace() || matches!(chars[j], ')' | ']' | '>' | '"'))
                .unwrap_or(chars.len());
            let url: String = chars[i..end].iter().collect();
            let url = url.trim_end_matches(['.', ',', ';', ':']);
            out.push_str(&url_host(url));
            i += url.chars().count();
            continue;
        }
        if c == '*' || (c == '_' && is_emphasis_underscore(&chars, i)) || c == '~' {
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn starts_with(chars: &[char], at: usize, prefix: &str) -> bool {
    prefix
        .chars()
        .enumerate()
        .all(|(k, p)| chars.get(at + k) == Some(&p))
}

/// `_` marks emphasis only at a word edge; `snake_case` keeps its underscores
/// out of the way by being read as written.
fn is_emphasis_underscore(chars: &[char], i: usize) -> bool {
    let before = i.checked_sub(1).map(|j| chars[j]);
    let after = chars.get(i + 1).copied();
    let word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    !(word(before) && word(after))
}

/// `[label](target)` starting at `open`, returning the label and the index
/// after the closing parenthesis.
fn link_at(chars: &[char], open: usize) -> Option<(String, usize)> {
    let close = (open + 1..chars.len()).find(|&j| chars[j] == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = (close + 2..chars.len()).find(|&j| chars[j] == ')')?;
    Some((chars[open + 1..close].iter().collect(), end + 1))
}

fn url_host(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let host = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    host.trim_start_matches("www.").to_string()
}

/// Cuts speakable text into chunks.
pub fn chunk(text: &str, chunking: Chunking) -> Vec<Chunk> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    let mut chunks = Vec::new();
    for (index, paragraph) in paragraphs.iter().enumerate() {
        let sentences = sentences(paragraph);
        let pieces: Vec<String> = match chunking {
            Chunking::Sentence => sentences.into_iter().map(str::to_string).collect(),
            Chunking::Packed {
                target,
                max,
                join_tail,
            } => pack(&sentences, target, max)
                .into_iter()
                .map(|piece| {
                    if join_tail {
                        join_short_tail(&piece)
                    } else {
                        piece
                    }
                })
                .collect(),
        };
        let last_paragraph = index + 1 == paragraphs.len();
        let count = pieces.len();
        for (i, speech) in pieces.into_iter().enumerate() {
            let pause = if i + 1 == count && !last_paragraph {
                PARAGRAPH_PAUSE_SECONDS
            } else {
                0.0
            };
            chunks.push(Chunk {
                speech,
                pause_after_seconds: pause,
            });
        }
    }
    chunks
}

/// Sentences of a paragraph, each keeping its terminal punctuation.
pub fn sentences(paragraph: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let bytes: Vec<(usize, char)> = paragraph.char_indices().collect();
    let mut i = 0;
    while i < bytes.len() {
        let (_, c) = bytes[i];
        if matches!(c, '.' | '!' | '?' | '…') {
            let mut j = i + 1;
            while j < bytes.len()
                && matches!(
                    bytes[j].1,
                    '.' | '!' | '?' | '…' | '"' | '\'' | ')' | ']' | '”' | '’'
                )
            {
                j += 1;
            }
            if j == bytes.len() || bytes[j].1.is_whitespace() {
                let end = bytes.get(j).map_or(paragraph.len(), |(offset, _)| *offset);
                let sentence = paragraph[start..end].trim();
                if !sentence.is_empty() {
                    result.push(sentence);
                }
                start = end;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    let rest = paragraph[start..].trim();
    if !rest.is_empty() {
        result.push(rest);
    }
    result
}

/// Packs sentences toward `target` characters. A sentence longer than `max` is
/// cut at the last clause mark, else the last space, before `max`.
fn pack(sentences: &[&str], target: usize, max: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for sentence in sentences {
        for part in split_long(sentence, max) {
            if !current.is_empty() && current.len() + 1 + part.len() > max {
                pieces.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&part);
            if current.len() >= target {
                pieces.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

fn split_long(sentence: &str, max: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = sentence.trim();
    while rest.len() > max {
        let mut window_end = max;
        while !rest.is_char_boundary(window_end) {
            window_end -= 1;
        }
        let window = &rest[..window_end];
        let cut = window
            .rfind([',', ';', ':'])
            .map(|at| at + 1)
            .filter(|&at| at > max / 3)
            .or_else(|| window.rfind(' '))
            .unwrap_or(window_end);
        parts.push(rest[..cut].trim().to_string());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        parts.push(rest.to_string());
    }
    parts
}

/// Joins a trailing run of two or more very short sentences with commas:
/// `One. Two. Three.` becomes `One, Two, Three.` Ported from
/// `TextChunker.joinShortTail`, whose doc carries the measurements.
pub fn join_short_tail(text: &str) -> String {
    let parts = sentences(text);
    let mut head = parts.len();
    while head > 0 && word_count(parts[head - 1]) <= SHORT_SENTENCE_WORDS {
        head -= 1;
    }
    if parts.len() - head < 2 {
        return text.to_string();
    }
    let mut out: Vec<String> = parts[..head].iter().map(|s| s.to_string()).collect();
    let tail = &parts[head..];
    for (i, sentence) in tail.iter().enumerate() {
        if i + 1 == tail.len() {
            out.push(sentence.to_string());
        } else {
            let body = sentence.trim_end_matches(['.', '!', '?', '…']);
            out.push(format!("{body},"));
        }
    }
    out.join(" ")
}

fn word_count(sentence: &str) -> usize {
    sentence
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trailing_run_of_short_sentences_is_joined() {
        assert_eq!(
            join_short_tail("Reading from Termux. One. Two. Three."),
            "Reading from Termux. One, Two, Three."
        );
        assert_eq!(
            join_short_tail("One. Two. Three. Four. Five."),
            "One, Two, Three, Four, Five."
        );
    }

    #[test]
    fn a_single_short_sentence_at_the_end_is_left_alone() {
        let text = "There is plenty of audio to chew through. Done.";
        assert_eq!(join_short_tail(text), text);
    }

    #[test]
    fn short_sentences_before_prose_are_left_alone() {
        let text = "One. Two. Three. And this final ordinary sentence closes it.";
        assert_eq!(join_short_tail(text), text);
    }

    #[test]
    fn fenced_code_is_not_read() {
        let markdown = "Here is the fix.\n\n```rust\nfn main() {}\n```\n\nIt builds.";
        assert_eq!(speakable(markdown), "Here is the fix.\n\nIt builds.");
    }

    /// The same outputs as the app's `MarkdownSpeechTest`, issue #10's tables
    /// among them.
    #[test]
    fn tables_are_announced_then_read_under_their_column_names() {
        let named = "| stage | before | after |\n|-------|--------|-------|\n| alpha | one | two |\n| bravo | three | four |";
        let blank =
            "| | before | after |\n|---|---|---|\n| alpha | one | two |\n| bravo | three | four |";
        let expected = "Table, 2 rows.\n\nalpha: before, one; after, two.\n\nbravo: before, three; after, four.";
        assert_eq!(speakable(named), expected);
        assert_eq!(speakable(blank), expected);
        assert_eq!(
            speakable("key | value\n--- | ---\nmodel | 7m\n\nAfter."),
            "Table, 1 row.\n\nmodel: value, 7m.\n\nAfter."
        );
        assert_eq!(
            speakable(
                "| name | role | note |\n|---|---|---|\n| Ada | | first |\n| | lead | |\n| | | |"
            ),
            "Table, 3 rows.\n\nAda: note, first.\n\nrole, lead."
        );
    }

    #[test]
    fn a_table_wider_than_four_columns_is_only_announced() {
        let markdown =
            "| a | b | c | d | e |\n|---|---|---|---|---|\n| 1 | 2 | 3 | 4 | 5 |\n\nAfter.";
        assert_eq!(
            speakable(markdown),
            "Table not read: 1 row, 5 columns, headed a, b, c, d and e.\n\nAfter."
        );
    }

    /// A French model reads "Table" as a French word it has never seen next
    /// to "rows"; the announcement is in the language being read.
    #[test]
    fn a_table_is_announced_in_the_reading_language() {
        let two = "| stage | before |\n|---|---|\n| alpha | one |\n| bravo | two |";
        assert_eq!(
            speakable_in(two, "fr"),
            "Tableau, 2 lignes.\n\nalpha: before, one.\n\nbravo: before, two."
        );
        assert!(speakable_in(two, "de-DE").starts_with("Tabelle, 2 Zeilen."));
        let wide = "| a | b | c | d | e |\n|---|---|---|---|---|\n| 1 | 2 | 3 | 4 | 5 |";
        assert_eq!(
            speakable_in(wide, "es"),
            "Tabla no leída: 1 fila, 5 columnas, con los encabezados a, b, c, d y e."
        );
        // Anything without its own words reads as English.
        assert_eq!(speakable_in(wide, "ja"), speakable(wide));
    }

    #[test]
    fn a_heading_underline_does_not_make_a_table() {
        assert!(!is_table_delimiter("---"));
        assert!(is_table_delimiter("|---|:--:|"));
        assert!(is_table_delimiter("--- | ---"));
        assert!(!is_table_delimiter("| a | b |"));
    }

    #[test]
    fn links_code_spans_and_emphasis_are_reduced_to_words() {
        let markdown = "See **[the PR](https://github.com/x/y/pull/8)** and run `cargo test` on snake_case_name.";
        assert_eq!(
            speakable(markdown),
            "See the PR and run cargo test on snake_case_name."
        );
    }

    #[test]
    fn bare_urls_are_read_as_their_host() {
        assert_eq!(
            speakable("Docs at https://www.example.com/a/b?c=d, then continue."),
            "Docs at example.com, then continue."
        );
    }

    #[test]
    fn headings_and_list_items_become_their_own_sentences() {
        assert_eq!(
            speakable("## Result\n- first item\n- second item\nAfter."),
            "Result.\n\nfirst item.\n\nsecond item.\n\nAfter."
        );
    }

    #[test]
    fn sentence_chunking_gives_one_generation_per_sentence() {
        let chunks = chunk("One. Two.\n\nThree!", Chunking::Sentence);
        let speech: Vec<_> = chunks.iter().map(|c| c.speech.as_str()).collect();
        assert_eq!(speech, ["One.", "Two.", "Three!"]);
        assert_eq!(chunks[1].pause_after_seconds, PARAGRAPH_PAUSE_SECONDS);
        assert_eq!(chunks[2].pause_after_seconds, 0.0);
    }

    #[test]
    fn packed_chunking_respects_max_and_joins_the_tail() {
        let long = "word ".repeat(120);
        let text = format!("{}. Short one. Done. Yes.", long.trim());
        let chunks = chunk(&text, Chunking::PACKED_DEFAULT);
        assert!(chunks.iter().all(|c| c.speech.len() <= 400), "{chunks:?}");
        assert!(
            chunks.last().unwrap().speech.ends_with("Done, Yes."),
            "{chunks:?}"
        );
    }

    #[test]
    fn sentences_keep_decimals_and_paths_together() {
        assert_eq!(
            sentences("It took 1.4x on v0.2.2 at ~/a.b/c.md. Next."),
            ["It took 1.4x on v0.2.2 at ~/a.b/c.md.", "Next."]
        );
    }
}
