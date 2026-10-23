"""Encode JSON-line texts with Google's sentencepiece, the way conditioners/text.py does.

Usage: google_ids.py TOKENIZER_MODEL IN_JSONL OUT_IDS
"""

import json
import sys

import sentencepiece

model, source, target = sys.argv[1:]
processor = sentencepiece.SentencePieceProcessor(model_file=model)
with open(source) as lines, open(target, "w") as out:
    for line in lines:
        ids = processor.encode(json.loads(line), out_type=int)
        out.write(" ".join(map(str, ids)) + "\n")
