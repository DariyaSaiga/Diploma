#!/usr/bin/env bash
# Exp02: MulT-like Cross-Modal Attention (six pairwise streams + concat + classifier).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=. python training/train.py \
  --model mult \
  --data_path datasets/mosei_emotion_aligned_60.pkl \
  --exp_dir experiments/exp02_mult_attention \
  --epochs 40 \
  --batch_size 32 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --hidden_dim 64 \
  --num_heads 4 \
  --num_layers 2 \
  --dropout 0.3 \
  --pos_weight_mode sqrt \
  --monitor_metric weighted_f1 \
  --threshold_search \
  --patience 7
