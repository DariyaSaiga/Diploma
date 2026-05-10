#!/usr/bin/env bash
# Exp03: Bottleneck Fusion with optional domain separation (proposed model).
# To run WITHOUT domain separation: drop --use_domain_sep below.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=. python training/train.py \
  --model bottleneck \
  --data_path datasets/mosei_emotion_aligned_60.pkl \
  --exp_dir experiments/exp03_bottleneck \
  --epochs 60 \
  --batch_size 16 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --hidden_dim 128 \
  --num_heads 8 \
  --num_layers 2 \
  --num_bottleneck_tokens 16 \
  --dropout 0.3 \
  --pos_weight_mode sqrt \
  --use_domain_sep \
  --alpha_sep 0.03 \
  --alpha_inv 0.005 \
  --alpha_rec 0.001 \
  --monitor_metric weighted_f1 \
  --threshold_search \
  --patience 8
