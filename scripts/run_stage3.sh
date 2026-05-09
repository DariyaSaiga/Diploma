#!/bin/bash
# Stage 3: Unfreeze last 4 BERT layers — end-to-end fine-tuning begins
# Expected: ~77-79% accuracy

echo "STAGE 3: Partial BERT unfreezing"

python training/train.py \
  --model bottleneck \
  --epochs 15 \
  --batch_size 32 \
  --lr 5e-4 \
  --lr_bert 5e-5 \
  --freeze_bert partial \
  --num_bottleneck_tokens 16 \
  --patience 5 \
  --label_smoothing 0.15 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --pretrained_path experiments/stage2_domain_sep/best_model.pt \
  --data_path "/content/drive/MyDrive/Дипломка_правильная/mosei_bottleneck.pkl" \
  --exp_dir experiments/stage3_domain_sep

echo "Stage 3 done. Results: cat experiments/stage3_domain_sep/metrics.txt"
