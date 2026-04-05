#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Запустить все эксперименты последовательно
#
# Использование:
#   cd Diploma/
#   bash experiments/run_all.sh
#
# Или только обязательные (без ablation):
#   bash experiments/run_exp01.sh
#   bash experiments/run_exp02.sh
#   bash experiments/run_exp03.sh
#   bash experiments/run_exp04.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # остановиться при первой ошибке

echo "======================================================"
echo "  Запускаем все эксперименты"
echo "======================================================"

echo ""
echo ">>> EXP 01: base bottleneck (n=8, lr=1e-3)"
mkdir -p experiments/exp_01_base_bottleneck
bash experiments/run_exp01.sh

echo ""
echo ">>> EXP 02: tokens=16, lr=1e-3"
bash experiments/run_exp02.sh

echo ""
echo ">>> EXP 03: tokens=8,  lr=5e-4"
bash experiments/run_exp03.sh

echo ""
echo ">>> EXP 04: tokens=16, lr=5e-4  (best combined)"
bash experiments/run_exp04.sh

echo ""
echo ">>> EXP 05: partial unfreeze BERT (tokens=16, lr=5e-4, lr_bert=2e-5)"
bash experiments/run_exp05.sh

echo ""
echo ">>> EXP 06A: ablation — no visual"
bash experiments/run_exp06A.sh

echo ""
echo ">>> EXP 06B: ablation — no audio"
bash experiments/run_exp06B.sh

echo ""
echo ">>> EXP 08: multi-layer MBT (layers=2, lr=3e-4, ep=20)"
bash experiments/run_exp08.sh

echo ""
echo ">>> EXP 09: multi-layer MBT + early stopping patience=3"
bash experiments/run_exp09.sh

echo ""
echo ">>> EXP 10: WeightedRandomSampler"
bash experiments/run_exp10.sh

echo ""
echo "======================================================"
echo "  Все эксперименты завершены!"
echo "  Результаты: experiments/exp_*/metrics.txt"
echo "  Сравнение:  python experiments/compare_results.py"
echo "======================================================"
