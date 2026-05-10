#!/usr/bin/env bash
# Exp01: Late Fusion CNN-BiLSTM baseline (text BiLSTM + audio CNN + vision BiLSTM).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=. python training/train.py \
  --model late_fusion \
  --data_path datasets/mosei_emotion_aligned_60.pkl \
  --exp_dir experiments/exp01_late_fusion \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.3 \
  --hidden_dim 128 \
  --pos_weight_mode sqrt \
  --monitor_metric weighted_f1 \
  --threshold_search \
  --patience 6
