# Commands used (2026-10-03). Not a script to run blind: the reference venv and fp32 model were deleted afterwards to free tmpfs.
uv venv venv && . venv/bin/activate && uv pip install sherpa-onnx==1.13.6 faster-whisper soundfile numpy jiwer
# model + voice pulled read-only from the S25: adb -s RFCY81H5HEB exec-out run-as org.pockettts.android cat files/pocket-tts/<...> > model/<file>
# matrix (speed has no effect; see report)
for m in whole sherpa; do for sp in 1.0 2.0; do for st in 1 5; do python synth.py out/${m}_sp${sp}_st${st} --mode $m --speed $sp --steps $st; done; done; done
# seeds 1-5, modes: whole sherpa ref refsherpa lines linessherpa perunit perunit@0.15 refpad (steps 1); whole sherpa (steps 5)
python synth.py seeds/<mode>_st<steps>_s<seed> --mode <mode> --steps <steps> --seed <seed>
MODEL=fp32 python synth.py seeds/fp32-<mode>_st1_s<seed> ...   # sherpa-onnx-pocket-tts-2026-01-26 (fp32), same encoder.onnx
# reference: uv venv refvenv; uv pip install pocket-tts==3.3.0 soundfile --extra-index-url https://download.pytorch.org/whl/cpu; python ref_synth.py seeds/reference_st1_s<seed> <seed>
HF_HOME=$PWD/hf python judge.py <celldir>...   # faster-whisper small.en int8 CPU, beam 5, word timestamps
python summary.py out/*_sp1.0_* ; python agg_seeds.py > agg_seeds.md
# Added after the swap (2026-10-03):
# desk pocket-speak (Linux build of pocket-speak/crates/cli, model in ~/.cache/pocket-tts-rs/english_2026-04):
python ps_synth.py seeds/pocketspeak_st1_s<seed> <seed>
# the same engine on the S25's CPU, via pocket-speak/crates/android's pocket-bench over adb, then judged here:
#   adb push pocket-bench libonnxruntime.so (onnxruntime-android 1.24.2, arm64-v8a) phone-lines.txt model/ to /data/local/tmp/pb
#   adb shell "cd /data/local/tmp/pb && ./pocket-bench ./libonnxruntime.so model model/embeddings/alba.safetensors out<seed> <seed> 2 < phone-lines.txt"
#   pull out<seed>/NNN.wav, rename to <item>.cNN.wav with phone-meta order, write meta.json, then judge.py
