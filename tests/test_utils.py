from pathlib import Path

import pytest

from pocket_tts.utils import utils


def test_download_http_cache_suffix_ignores_query_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class Response:
        content = b"data"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(utils, "make_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(utils.requests, "get", lambda url: Response())

    cached_file = utils.download_if_necessary("https://example.com/audio.wav?download=true")

    assert cached_file.parent == tmp_path
    assert cached_file.suffix == ".wav"
    assert "?" not in cached_file.name
    assert cached_file.read_bytes() == b"data"


@pytest.mark.parametrize(
    "config_path", sorted((Path(__file__).parent.parent / "pocket_tts" / "config").glob("*.yaml"))
)
def test_shipped_configs_sample_at_the_tuned_temperature(config_path: Path):
    # Configs without an explicit value used to fall back to 0.7 silently (#322).
    from pocket_tts.utils.config import load_config

    assert load_config(config_path).default_temperature == 0.3
