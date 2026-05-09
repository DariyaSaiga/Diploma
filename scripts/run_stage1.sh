#!/bin/bash
# Stage 1: Fresh training — frozen BERT, domain separation warm-up
# Expected: ~74-76% accuracy

echo "STAGE 1: Fresh Training (frozen BERT)"

python training/train.py \
  --model bottleneck \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --lr_bert 5e-5 \
  --freeze_bert full \
  --num_bottleneck_tokens 16 \
  --patience 5 \
  --label_smoothing 0.1 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage1_domain_sep

echo "Stage 1 done. Results: cat experiments/stage1_domain_sep/metrics.txt"
