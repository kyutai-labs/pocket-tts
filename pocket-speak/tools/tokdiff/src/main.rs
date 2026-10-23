// Reads one text per line on stdin and prints the sentencepiece-rs ids for each,
// space-separated, one line per input. Lines are JSON strings so embedded
// newlines and leading spaces survive the round trip unambiguously.
use std::io::{BufRead, Write};

use sentencepiece_rs::SentencePieceProcessor;

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: tokdiff <tokenizer.model>");
    let processor = SentencePieceProcessor::open(&model).expect("open tokenizer.model");
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("read stdin");
        let text = decode_json_string(&line);
        let ids = processor.encode_to_ids(&text).expect("encode");
        let joined = ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(out, "{joined}").expect("write");
    }
}

/// Minimal decoder for a JSON string literal as written by Python's json.dumps
/// with ensure_ascii=True, which is all the corpus writer emits.
fn decode_json_string(line: &str) -> String {
    let inner = line
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .expect("each line must be a JSON string");
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            result.push(c);
            continue;
        }
        match chars.next().expect("dangling escape") {
            '"' => result.push('"'),
            '\\' => result.push('\\'),
            '/' => result.push('/'),
            'b' => result.push('\u{8}'),
            'f' => result.push('\u{c}'),
            'n' => result.push('\n'),
            'r' => result.push('\r'),
            't' => result.push('\t'),
            'u' => {
                let high = read_hex4(&mut chars);
                if (0xD800..0xDC00).contains(&high) {
                    assert_eq!(chars.next(), Some('\\'));
                    assert_eq!(chars.next(), Some('u'));
                    let low = read_hex4(&mut chars);
                    let code = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                    result.push(char::from_u32(code).expect("valid surrogate pair"));
                } else {
                    result.push(char::from_u32(high).expect("valid code point"));
                }
            }
            other => panic!("unknown escape \\{other}"),
        }
    }
    result
}

fn read_hex4(chars: &mut std::str::Chars<'_>) -> u32 {
    let hex: String = chars.take(4).collect();
    u32::from_str_radix(&hex, 16).expect("four hex digits")
}
