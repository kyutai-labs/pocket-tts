# Tests (tests/)

Comprehensive test suite ensuring TTS generation quality and API reliability.

## OVERVIEW
Test suite covering Python API, CLI integration, documentation examples, and WebSocket communication.

## TEST PATTERNS

**Cache Management**: Clear voice prompt cache after each test to prevent state pollution:
```python
model._cached_get_state_for_audio_prompt.cache_clear()
```

**Audio Validation**: Verify output files via structural checks:
- File existence: `assert output_file.exists()`
- Non-zero size: `assert output_file.stat().st_size > 0`
- Valid audio: `audio, sample_rate = audio_read(str(output_file))`
- Mono channel: `assert audio.shape[0] == 1`
- Has samples: `assert audio.shape[1] > 0`
- Sample rate: `assert sample_rate == 24000`

**Documentation Tests**: Run code examples from docs in `test_documentation_examples.py` to ensure they remain functional.

## CONVENTIONS

**EOS Enforcement**: `conftest.py` sets `POCKET_TTS_ERROR_WITHOUT_EOS=1` - tests must fail when End-of-Speech detection doesn't trigger. This ensures generation doesn't hang or produce incomplete audio.

**Integration Tests**: Use Typer's `CliRunner` for CLI testing. Tests cover basic usage, custom voices, custom parameters, verbose mode, default text, long text, and multiple consecutive runs.

**Sentence Splitting**: Mock tokenizer behavior in unit tests to verify `split_into_best_sentences()` preserves all content including comma-separated clauses (addresses Issue #3).

**WebSocket Tests**: Async tests using `websockets` library to verify streaming audio generation.
