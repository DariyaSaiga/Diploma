#!/bin/bash
# Stage 4: Final optimization — higher regularization, stronger label smoothing
# Expected: ~79-82% accuracy

echo "STAGE 4: Final optimization"

python training/train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-4 \
  --lr_bert 2e-5 \
  --freeze_bert partial \
  --num_bottleneck_tokens 16 \
  --patience 5 \
  --label_smoothing 0.2 \
  --dropout 0.15 \
  --use_domain_sep \
  --alpha_sep 0.15 \
  --alpha_inv 0.08 \
  --alpha_rec 0.02 \
  --pretrained_path experiments/stage3_domain_sep/best_model.pt \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage4_domain_sep

echo "Stage 4 done. Results: cat experiments/stage4_domain_sep/metrics.txt"
