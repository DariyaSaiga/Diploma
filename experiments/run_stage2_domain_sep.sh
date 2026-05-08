#!/bin/bash

# Stage 2: Fine-tuning с Domain Separation
# Learning Rate: 3e-4 (reduced 3.3x)
# Epochs: 20
# BERT: Frozen
# Domain Separation: Enabled (increased weights)
# Pretrained: stage1_domain_sep/best_model.pt
# Expected: 76-78% accuracy

echo "🚀 STAGE 2: DOMAIN-SEPARATED FINE-TUNING"
echo "========================================="

python train.py \
  --model bottleneck \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_bottleneck_tokens 16 \
  --num_bottleneck_layers 2 \
  --freeze_bert full \
  --patience 5 \
  --label_smoothing 0.15 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --data_path mosei_bottleneck.pkl \
  --pretrained_path experiments/stage1_domain_sep/best_model.pt \
  --exp_dir experiments/stage2_domain_sep

echo ""
echo "✅ Stage 2 completed!"
echo "📊 Results saved to: experiments/stage2_domain_sep/"
echo ""
echo "📈 Check results:"
echo "   cat experiments/stage2_domain_sep/metrics.txt"
echo ""
echo "🔥 Ready for Stage 3? Run: bash experiments/run_stage3_domain_sep.sh"
