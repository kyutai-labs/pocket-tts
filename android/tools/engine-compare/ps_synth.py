"""Same chunks through the desk's pocket-speak (Rust, ONNX Runtime), given raw as the phone's service gets them.
usage: ps_synth.py OUTDIR SEED"""

import json
import pathlib
import subprocess
import sys

import soundfile as sf
from chunker import chunk
from corpus import READALOUD, TALKBACK

PS = str(pathlib.Path(__file__).resolve().parents[3] / "pocket-speak/target/release/pocket-speak")
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
seed = int(sys.argv[2])
meta = []
jobs = [(n, t, "text") for n, t in TALKBACK] + [(n, t, "speech") for n, t in READALOUD]
for name, text, field in jobs:
    for ci, (ctext, cspeech, pause) in enumerate(chunk(text)):
        src = ctext if field == "text" else cspeech
        wav = out / f"{name}.c{ci:02d}.wav"
        subprocess.run(
            [PS, "--plain", "--seed", str(seed), "--threads", "2", "--out", str(wav)],
            input=src.encode(),
            check=True,
            stderr=subprocess.DEVNULL,
        )
        a, sr = sf.read(wav)
        meta.append(
            {"item": name, "chunk": ci, "text": src, "pause": pause, "dur": round(len(a) / sr, 3)}
        )
(out / "meta.json").write_text(
    json.dumps({"mode": "pocket-speak", "seed": seed, "steps": 1, "chunks": meta}, indent=1)
)
