# Notices

## speech-kit-obsidian-plugin

`src/model.rs` is ported from `native/src/adapters/pocket_tts.rs` in
[speech-kit-obsidian-plugin](https://github.com/brittain9/speech-kit-obsidian-plugin),
used under the MIT License:

```
MIT License

Copyright (c) 2026 Alexander Brittain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`src/text.rs`'s trailing-short-sentence join follows this repository's
`android/app/src/main/java/org/pockettts/android/speech/TextChunker.kt`.

## Models

The engine runs models it does not ship. `pocket-speak install` fetches them:

- ONNX export: [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx),
  `onnx/english_2026-04`, CC-BY-4.0.
- Voice embeddings: [kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning),
  `languages/english_2026-04/embeddings`, CC-BY-4.0.

Pocket TTS is by [Kyutai](https://kyutai.org).
