"""Compare two id files line by line and show the first divergences.

Usage: compare.py TOKENIZER_MODEL IN_JSONL GOOGLE_IDS RUST_IDS
Exits non-zero on any mismatch.
"""

import json
import sys

import sentencepiece

model, source, google_path, rust_path = sys.argv[1:]
processor = sentencepiece.SentencePieceProcessor(model_file=model)
texts = [json.loads(line) for line in open(source)]
google = open(google_path).read().split("\n")[: len(texts)]
rust = open(rust_path).read().split("\n")[: len(texts)]
mismatches = [i for i in range(len(texts)) if google[i] != rust[i]]
print(f"{len(mismatches)} mismatches of {len(texts)}")
for i in mismatches[:10]:
    print(repr(texts[i])[:120])
    print("  google:", [processor.id_to_piece(int(t)) for t in google[i].split()][:16])
    print("  rust:  ", [processor.id_to_piece(int(t)) for t in rust[i].split()][:16])
sys.exit(1 if mismatches else 0)
