#!/bin/bash
# Keep the Indonesian run alive on a shared GPU.
#
# The T4 is shared with other jobs and peak usage leaves ~1.3 GB of headroom, so
# a neighbour growing means CUDA OOM. train.py resumes from the newest
# checkpoint in run_dir on its own, so a restart costs at most ckpt_freq steps.
# 40 consecutive failures means the cause is not transient -- stop and look.
set -u
cd "$(dirname "$0")/../.." || exit 1
CONFIG=training/configs/finetune_indonesian.yaml
LOG=runs/indonesian/launch.log
mkdir -p runs/indonesian

for attempt in $(seq 40); do
    echo "=== attempt $attempt: $(date -Is) ===" >>"$LOG"
    if .venv/bin/python training/train.py "$CONFIG" >>"$LOG" 2>&1; then
        echo "=== finished: $(date -Is) ===" >>"$LOG"
        exit 0
    fi
    echo "=== attempt $attempt failed: $(date -Is), retrying in 180s ===" >>"$LOG"
    sleep 180
done
echo "=== gave up after 40 attempts: $(date -Is) ===" >>"$LOG"
exit 1
