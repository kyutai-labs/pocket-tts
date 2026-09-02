"""WER of the ASR on the eval set's real audio -- the floor a TTS can reach.

librispeech.py scores a TTS by transcribing what it generated and comparing to
the reference text. On a corpus whose "reference" text was itself produced by
ASR or by subtitles (LEMAS, YODAS, GigaSpeech), that reference is not clean, so
a WER of, say, 18% says nothing on its own: the same ASR on the *original*
recording may already score 15%. This measures that floor with the same ASR,
normalizer and item list, so the TTS number can be read as a gap.

Usage:
    python -m training.eval.asr_floor data/eval_id/pairs.lst \
        --librispeech-root data/eval_id/audio --asr openai/whisper-large-v3 \
        --normalizer basic
"""

import argparse

import jiwer
import torch

from training.eval.librispeech import build_transcriber, load_16k, read_lst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lst")
    ap.add_argument("--librispeech-root", required=True)
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--normalizer", choices=("english", "basic"), default="basic")
    ap.add_argument("--num-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.normalizer == "english":
        from whisper_normalizer.english import EnglishTextNormalizer as Normalizer
    else:
        from whisper_normalizer.basic import BasicTextNormalizer as Normalizer
    normalize = Normalizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read_lst resolves item["ref"] to the *target* utterance's audio, which is
    # exactly the recording whose transcript the TTS is asked to reproduce.
    items = read_lst(args.lst, args.librispeech_root, args.num_items, prompt_root=None)
    transcribe = build_transcriber(args.asr, device)

    refs, hyps = [], []
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        # build_transcriber feeds these straight to the ASR pipeline, which
        # wants host arrays -- load on CPU and let it do its own placement.
        wavs = [load_16k(item["ref"], torch.device("cpu")).numpy() for item in chunk]
        for item, hyp in zip(chunk, transcribe(wavs), strict=True):
            refs.append(normalize(item["text"]))
            hyps.append(normalize(hyp))
        print(f"{len(refs)}/{len(items)}", flush=True)

    wer = jiwer.wer(refs, hyps)
    print(f"\nASR floor on real audio: WER {wer:.2%} over {len(refs)} items ({args.asr})")
    for r, h in list(zip(refs, hyps, strict=True))[:3]:
        print(f"  ref: {r[:100]}\n  hyp: {h[:100]}")


if __name__ == "__main__":
    main()
