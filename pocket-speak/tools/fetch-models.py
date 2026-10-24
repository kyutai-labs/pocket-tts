"""Fetches models from pocket-speak/models.json into DIR/<model>, verified.

The same layout `pocket-speak install` and the app use: a model's files in its
directory, voices in its embeddings/. Standard library only, for CI and for
machines without the CLI. Files already present and matching are kept.

    python3 pocket-speak/tools/fetch-models.py DIR [--model NAME ...] [--voice NAME ...]

Every model unless --model is given. Voices: alba and the model's default voice,
unless --voice is given.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import urllib.request

MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "models.json"


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: pathlib.Path, sha256: str) -> None:
    if target.is_file() and sha256_of(target) == sha256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    actual = sha256_of(part)
    if actual != sha256:
        part.unlink()
        sys.exit(f"{url}: sha256 {actual}, expected {sha256}")
    part.replace(target)
    print(f"fetched {target}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=pathlib.Path)
    parser.add_argument("--model", action="append")
    parser.add_argument("--voice", action="append")
    args = parser.parse_args()

    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    by_name = {model["name"]: model for model in models}
    for name in args.model or list(by_name):
        model = by_name.get(name) or sys.exit(f"no model {name}; offered: {', '.join(by_name)}")
        directory = args.dir / name
        for file in model["files"]:
            fetch(file["url"], directory / file["name"], file["sha256"])
        for voice in args.voice or sorted({"alba", model["default_voice"]}):
            pinned = model["voices"][voice]
            fetch(
                pinned["url"], directory / "embeddings" / f"{voice}.safetensors", pinned["sha256"]
            )


if __name__ == "__main__":
    main()
