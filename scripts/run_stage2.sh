#!/bin/bash
# Stage 2: Fine-tune from Stage 1 — lower LR, frozen BERT, stronger domain separation
# Expected: ~76-78% accuracy

echo "STAGE 2: Fine-tuning from Stage 1"

python training/train.py \
  --model bottleneck \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --lr_bert 5e-5 \
  --freeze_bert full \
  --num_bottleneck_tokens 16 \
  --patience 5 \
  --label_smoothing 0.15 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --pretrained_path experiments/stage1_domain_sep/best_model.pt \
  --data_path "/content/drive/MyDrive/Дипломка_правильная/mosei_bottleneck.pkl" \
  --exp_dir experiments/stage2_domain_sep

echo "Stage 2 done. Results: cat experiments/stage2_domain_sep/metrics.txt"
