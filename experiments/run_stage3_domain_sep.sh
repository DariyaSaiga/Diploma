#!/bin/bash

# Stage 3: BERT Unfreezing (Feature Adaptation)
# BERT LR: 5e-5 (very careful)
# Other LR: 5e-4
# Epochs: 15
# Domain Separation: Enabled
# Pretrained: stage2_domain_sep/best_model.pt
# Expected: 77-79% accuracy

echo "🚀 STAGE 3: BERT UNFREEZING + DOMAIN SEPARATION"
echo "==============================================="

python train.py \
  --model bottleneck \
  --epochs 15 \
  --batch_size 32 \
  --lr 5e-4 \
  --lr_bert 5e-5 \
  --num_bottleneck_tokens 16 \
  --num_bottleneck_layers 2 \
  --freeze_bert partial \
  --patience 5 \
  --label_smoothing 0.15 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --data_path mosei_bottleneck.pkl \
  --pretrained_path experiments/stage2_domain_sep/best_model.pt \
  --exp_dir experiments/stage3_domain_sep

echo ""
echo "✅ Stage 3 completed!"
echo "📊 Results saved to: experiments/stage3_domain_sep/"
echo ""
echo "📈 Check results:"
echo "   cat experiments/stage3_domain_sep/metrics.txt"
echo ""
echo "🔥 Ready for Stage 4? Run: bash experiments/run_stage4_domain_sep.sh"
