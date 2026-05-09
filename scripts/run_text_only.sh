#!/bin/bash
# Baseline 1: Text-only BERT
# Target: accuracy ~70-75%, macro F1 >= 0.55

echo "TEXT-ONLY BERT BASELINE"

python training/train.py \
  --model text_only \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-4 \
  --lr_bert 2e-5 \
  --freeze_bert partial \
  --patience 5 \
  --label_smoothing 0.1 \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/text_only_bert

echo "Done. Results: cat experiments/text_only_bert/metrics.txt"
