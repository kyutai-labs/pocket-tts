Output of `agg_seeds.py` over `seeds/*/judge.json` (those files are not in the repository). Mode names are defined in `synth.py` and `ref_synth.py`, `ps_synth.py`: `whole` is one generation per chunk with the raw text (the app after PR #5); `sherpa` is sherpa-onnx's own sentence split, which PR #5 turned off; `ref` applies the reference's text preparation (`prepare_text_prompt`) in one pass; `refsherpa` is the same with sherpa's split; `refpad` adds the reference's space padding and EOS frame count; `lines` and `linessherpa` read each line as a sentence; `perunit@P` makes one generation per sentence with a P-second pause; `reference` is upstream pocket-tts 3.3.0 in PyTorch; `fp32-` is the fp32 sherpa-onnx model; `pocketspeak` is the desk build of pocket-speak; `phone-pocketspeak` is `pocket-bench` on the S25.

| mode, steps | corpus | WER (5 seeds) | worst seed WER | items with WER>=0.3 | wps | boundary quiet median s | boundaries <100 ms |
|---|---|---|---|---|---|---|---|
| whole_st1 | TalkBack | 53.0% | 57.1% | 37/45 | 3.02 | 0.13 | 24/57 |
| whole_st1 | ReadAloud | 3.4% | 7.0% | 0/25 | 4.38 | 0.17 | 8/40 |
| sherpa_st1 | TalkBack | 53.0% | 57.1% | 37/45 | 3.02 | 0.13 | 24/57 |
| sherpa_st1 | ReadAloud | 4.1% | 4.7% | 0/25 | 4.43 | 0.15 | 11/40 |
| whole_st5 | TalkBack | 59.0% | 79.4% | 38/45 | 2.44 | 0.15 | 27/57 |
| whole_st5 | ReadAloud | 2.3% | 2.8% | 0/25 | 4.24 | 0.17 | 2/40 |
| sherpa_st5 | TalkBack | 59.0% | 79.4% | 38/45 | 2.44 | 0.15 | 27/57 |
| sherpa_st5 | ReadAloud | 3.4% | 4.7% | 0/25 | 4.34 | 0.14 | 9/40 |
| ref_st1 | TalkBack | 23.8% | 25.4% | 22/45 | 3.98 | 0.09 | 36/65 |
| ref_st1 | ReadAloud | 3.4% | 7.0% | 0/25 | 4.38 | 0.17 | 8/40 |
| refsherpa_st1 | TalkBack | 23.8% | 25.4% | 22/45 | 3.98 | 0.09 | 36/65 |
| refsherpa_st1 | ReadAloud | 4.1% | 4.7% | 0/25 | 4.43 | 0.15 | 11/40 |
| lines_st1 | TalkBack | 24.8% | 30.2% | 23/45 | 3.96 | 0.10 | 29/60 |
| lines_st1 | ReadAloud | 3.4% | 7.0% | 0/25 | 4.38 | 0.17 | 8/40 |
| linessherpa_st1 | TalkBack | 27.0% | 33.3% | 27/45 | 3.95 | 0.11 | 28/61 |
| linessherpa_st1 | ReadAloud | 4.1% | 4.7% | 0/25 | 4.43 | 0.15 | 11/40 |
| perunit_st1 | TalkBack | 32.7% | 46.0% | 29/45 | 4.00 | 0.13 | 13/64 |
| perunit_st1 | ReadAloud | 4.1% | 5.1% | 0/25 | 4.43 | 0.15 | 7/40 |
| perunit@0.15_st1 | TalkBack | 34.6% | 50.8% | 31/45 | 3.59 | 0.28 | 5/65 |
| perunit@0.15_st1 | ReadAloud | 3.6% | 4.2% | 0/25 | 4.33 | 0.30 | 0/40 |
| refpad_st1 | TalkBack | 28.6% | 33.3% | 22/45 | 3.90 | 0.09 | 36/65 |
| refpad_st1 | ReadAloud | 3.4% | 7.0% | 0/25 | 4.38 | 0.17 | 8/40 |
| reference_st1 | TalkBack | 10.2% | 19.0% | 8/45 | 2.61 | 0.21 | 27/63 |
| reference_st1 | ReadAloud | 2.1% | 2.8% | 0/25 | 3.60 | 0.39 | 0/40 |
| fp32-whole_st1 | TalkBack | 54.6% | 71.4% | 35/45 | 2.45 | 0.17 | 20/55 |
| fp32-whole_st1 | ReadAloud | 1.8% | 2.3% | 0/25 | 4.29 | 0.20 | 4/40 |
| fp32-ref_st1 | TalkBack | 20.0% | 31.7% | 21/45 | 3.70 | 0.07 | 37/65 |
| fp32-ref_st1 | ReadAloud | 1.8% | 2.3% | 0/25 | 4.29 | 0.20 | 4/40 |
| pocketspeak_st1 | TalkBack | 3.5% | 6.3% | 5/45 | 2.45 | 0.09 | 34/65 |
| pocketspeak_st1 | ReadAloud | 1.5% | 1.9% | 0/25 | 3.55 | 0.34 | 4/40 |
| phone-pocketspeak_st1 | TalkBack | 6.7% | 17.5% | 7/45 | 2.45 | 0.09 | 34/65 |
| phone-pocketspeak_st1 | ReadAloud | 1.2% | 1.9% | 0/25 | 3.59 | 0.35 | 0/40 |
