#!/bin/bash

# Stage 1: Fresh Training с Domain Separation
# Learning Rate: 1e-3
# Epochs: 20
# BERT: Frozen
# Domain Separation: Enabled
# Expected: 74-76% accuracy

echo "🚀 STAGE 1: DOMAIN-SEPARATED BOTTLENECK TRAINING (Fresh)"
echo "=========================================================="

python train.py \
  --model bottleneck \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_bottleneck_tokens 16 \
  --num_bottleneck_layers 2 \
  --freeze_bert full \
  --patience 5 \
  --label_smoothing 0.1 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage1_domain_sep

echo ""
echo "✅ Stage 1 completed!"
echo "📊 Results saved to: experiments/stage1_domain_sep/"
echo ""
echo "📈 Check results:"
echo "   cat experiments/stage1_domain_sep/metrics.txt"
