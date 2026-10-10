#!/usr/bin/env bash
# LR sweep for the fine-tuned heads; each config writes its own results dir, pick by val macro-F1.
set -e
cd "$(dirname "$0")"
OUT=${1:-../../Tests/2026-10-05_in_distribution/sweep}
for LR in 1e-4 3e-4 1e-3; do
  ../../.venv/bin/python -u run_experiment.py --epochs 30 --lr $LR --out "$OUT/lr_$LR" 2>&1 \
    | grep -v -E "Warning|UNEXPECTED|Notes:|can be ignored" > "$OUT.lr_$LR.log" || true
done
