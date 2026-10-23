"""Build a tokenizer-diff corpus from Markdown/text trees.

Usage: build_corpus.py BUNDLE_JSON OUT_DIR SOURCE [SOURCE ...]

Each sentence is emitted twice: raw, and as the Rust engine's prepare_text
hands it to the tokenizer (whitespace collapsed, first letter capitalised, a
final period added, and the bundle's semicolon/padding flags applied). Lines
of corpus.jsonl are JSON strings; corpus.kind says which form each line is.
"""

import json
import pathlib
import re
import sys

bundle_path, out_dir, *sources = sys.argv[1:]
bundle = json.load(open(bundle_path))
pad = bundle["pad_with_spaces_for_short_inputs"]
semicolons = bundle["remove_semicolons"]
out = pathlib.Path(out_dir)

files = []
for source in map(pathlib.Path, sources):
    if source.is_file():
        files.append(source)
    else:
        files += [p for p in source.rglob("*") if p.suffix in {".md", ".txt"} and p.is_file()]

sentences = set()
splitter = re.compile(r"(?<=[.!?])\s+")
for path in files:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in splitter.split(" ".join(paragraph.split())):
            if 1 <= len(sentence) <= 400:
                sentences.add(sentence)

sentences.update(
    [
        "Visit https://github.com/mycosavant/pocket-tts/pull/8 for details.",
        "Run `cargo check --workspace --all-targets` first.",
        "The file is at C:\\Users\\me\\AppData\\Local\\Programs\\App\\App.exe.",
        "It took 1.4-4x real time, 151069 bytes, and 0.83x on 2026-09-12.",
        "Tests 1, 2 and 3 are green.",
        "One. Two. Three. Four. Five.",
        "fn generate_chunk(&mut self, text: &str) -> Result<Vec<f32>, SynthesisError>",
        "café naïve résumé Straße Ελληνικά 日本語 😀 👍🏽",
        "Tabs\tand   multiple   spaces.",
        "   leading and trailing spaces   ",
        "ALL CAPS SHOUTING AND camelCaseIdentifiers and snake_case_names.",
        "Emoji only: 🎉",
        "Quotes: \"double\", 'single', “curly”, ‘curly’ — em dash – en dash …",
        "a",
        ".",
        "x = 3; y = 4; z = x + y;",
        "~/git/project/docs/voice.md:233",
        "$XDG_RUNTIME_DIR/app/local-control",
        "0x9e37_79b9_7f4a_7c15",
        "The sha256 is d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6.",
        "Ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ and ① ② ③ and ﬁ ligature.",
        "Zero\u200bwidth\u200bspace and non\u00a0breaking space.",
    ]
)


def prepare(text):
    prepared = " ".join(text.split())
    if not prepared:
        return None
    if semicolons:
        prepared = prepared.replace(";", ",")
    words = len(prepared.split())
    if prepared[:1].isascii():
        prepared = prepared[:1].upper() + prepared[1:]
    if prepared[-1].isalnum():
        prepared += "."
    if pad and words < 5:
        prepared = "        " + prepared
    return prepared


rows = []
for sentence in sorted(sentences):
    rows.append(("raw", sentence))
    prepared = prepare(sentence)
    if prepared is not None and prepared != sentence:
        rows.append(("prepared", prepared))

with open(out / "corpus.jsonl", "w") as corpus, open(out / "corpus.kind", "w") as kinds:
    for kind, text in rows:
        corpus.write(json.dumps(text, ensure_ascii=True) + "\n")
        kinds.write(kind + "\n")
print(f"files={len(files)} sentences={len(sentences)} rows={len(rows)}", file=sys.stderr)
