# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml==6.0.2"]
# ///
"""Writes pocket-speak/models.json: every file the offered models need, pinned.

The engine reads each model's config from upstream's pocket_tts/config/<name>.yaml
(embedded at build time) and resolves the hf:// paths in it to files in a model
directory. This script records, for each of those files and for each stock voice,
the URL to fetch it from and its sha256, so the app and `pocket-speak install`
download exactly what the engine will load and refuse anything else.

sha256s come from Hugging Face's API (the LFS oid). A file not stored in LFS has
no sha256 there, so it is downloaded and hashed.

    uv run pocket-speak/tools/pin-models.py            # the offered models
    uv run pocket-speak/tools/pin-models.py NAME ...   # others

Rerun it when upstream moves a model's files; the engine's tests fail until then.
"""

import ast
import hashlib
import json
import pathlib
import sys
import urllib.parse
import urllib.request
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "pocket_tts" / "config"
OUT = ROOT / "pocket-speak" / "models.json"

# English largest first, then the other languages. The app lists them in this order.
OFFERED = [
    "english_2026-10",
    "english_18m_2026-10",
    "english_7m_2026-10",
    "english_2m_2026-10",
    "french",
    "german",
    "spanish",
    "italian",
    "portuguese",
    "dutch",
]

# The BCP-47 tag of each language upstream names its configs after.
LANGUAGES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
}

# Upstream's value when a config does not set default_temperature.
DEFAULT_TEMPERATURE = 0.3

# The stock voices' precomputed states: the revision pocket_tts/utils/utils.py
# pins in get_predefined_voice (upstream's rust/src/defaults.rs pins an older
# one, which lacks the 2026-10 models).
VOICE_REPO = "kyutai/pocket-tts-without-voice-cloning"
VOICE_REVISION = "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"


def upstream_defaults() -> dict[str, Any]:
    """DEFAULT_VOICE_FOR_LANGUAGE and DEFAULT_TEXT_FOR_LANGUAGE from default_parameters.py.

    Read as literals rather than imported: importing it imports the package, and torch.
    """
    source = (ROOT / "pocket_tts" / "default_parameters.py").read_text(encoding="utf-8")
    wanted = {"DEFAULT_VOICE_FOR_LANGUAGE", "DEFAULT_TEXT_FOR_LANGUAGE", "DEFAULT_VOICE_FALLBACK"}
    found: dict[str, Any] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - found.keys()
    if missing:
        sys.exit(f"default_parameters.py no longer defines {', '.join(sorted(missing))}")
    return found


def by_language(table: dict[str, str], name: str, fallback: str) -> str:
    """Upstream's lookup: the first key that is a substring of the config name."""
    return next((value for key, value in table.items() if key in name), fallback)


def parse_hf(path: str) -> tuple[str, str, str]:
    """hf://owner/repo/some/file@rev -> (owner/repo, some/file, rev)."""
    assert path.startswith("hf://"), path
    rest, _, revision = path[len("hf://") :].partition("@")
    owner, repo, file = rest.split("/", 2)
    if not revision:
        sys.exit(f"{path} is not pinned to a revision")
    return f"{owner}/{repo}", file, revision


def paths_info(repo: str, revision: str, paths: list[str]) -> dict[str, dict]:
    data = urllib.parse.urlencode([("paths", p) for p in paths] + [("expand", "true")]).encode()
    url = f"https://huggingface.co/api/models/{repo}/paths-info/{revision}"
    with urllib.request.urlopen(urllib.request.Request(url, data=data)) as response:
        return {entry["path"]: entry for entry in json.load(response)}


def list_dir(repo: str, revision: str, path: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}/{path}"
    with urllib.request.urlopen(url) as response:
        return [entry["path"] for entry in json.load(response) if entry["type"] == "file"]


def resolve_url(repo: str, file: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file}"


def sha256_of(url: str) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        while chunk := response.read(1 << 16):
            digest.update(chunk)
    return digest.hexdigest()


def pin(repo: str, file: str, revision: str, info: dict) -> dict:
    entry = info.get(file)
    if entry is None:
        sys.exit(f"hf://{repo}/{file}@{revision} does not exist")
    url = resolve_url(repo, file, revision)
    lfs = entry.get("lfs")
    sha256 = lfs["oid"] if lfs else sha256_of(url)
    return {"name": file.rsplit("/", 1)[-1], "url": url, "sha256": sha256, "size": entry["size"]}


def model_entry(name: str, defaults: dict[str, Any]) -> dict:
    config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    language = by_language(LANGUAGES, name, "")
    if not language:
        sys.exit(f"{name}: no language in its name; add it to LANGUAGES")
    # The weights without the voice-cloning encoder: public, smaller, and all a
    # model needs with precomputed voices.
    sources = [
        config["weights_path_without_voice_cloning"],
        config["flow_lm"]["lookup_table"]["tokenizer_path"],
    ]
    files = []
    for source in sources:
        repo, file, revision = parse_hf(source)
        files.append(pin(repo, file, revision, paths_info(repo, revision, [file])))

    voice_dir = f"languages/{name}/embeddings"
    voice_files = sorted(list_dir(VOICE_REPO, VOICE_REVISION, voice_dir))
    info = paths_info(VOICE_REPO, VOICE_REVISION, voice_files)
    voices = {}
    for file in voice_files:
        if file.endswith(".safetensors"):
            pinned = pin(VOICE_REPO, file, VOICE_REVISION, info)
            voices[pinned["name"].removesuffix(".safetensors")] = {
                "url": pinned["url"],
                "sha256": pinned["sha256"],
                "size": pinned["size"],
            }
    print(f"{name}: {len(files)} files, {len(voices)} voices", file=sys.stderr)
    default_voice = by_language(
        defaults["DEFAULT_VOICE_FOR_LANGUAGE"], name, defaults["DEFAULT_VOICE_FALLBACK"]
    )
    if default_voice not in voices:
        sys.exit(f"{name}: no embedding for its default voice {default_voice}")
    texts = defaults["DEFAULT_TEXT_FOR_LANGUAGE"]
    return {
        "name": name,
        "language": language,
        "default_temperature": config.get("default_temperature", DEFAULT_TEMPERATURE),
        "default_voice": default_voice,
        "sample_text": by_language(texts, name, texts["english"]),
        "files": files,
        "voices": voices,
    }


def main() -> None:
    names = sys.argv[1:] or OFFERED
    defaults = upstream_defaults()
    manifest = {
        "generated_by": "pocket-speak/tools/pin-models.py",
        "models": [model_entry(name, defaults) for name in names],
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
