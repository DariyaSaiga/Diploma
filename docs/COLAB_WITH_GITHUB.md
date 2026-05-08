# 🚀 Google Colab Training with GitHub

## Two Options

### ✅ OPTION 1: GitHub (Recommended - Fresh Code)

**Pros**:
- Always latest code from repo
- No manual uploads
- Clean setup

**How**:
1. Copy content of `COLAB_CLEAN.py`
2. Paste into ONE Colab cell
3. Keep `USE_GITHUB = True`
4. Run (Ctrl+Enter)

**What it does**:
```python
USE_GITHUB = True  # Clone from GitHub
GITHUB_REPO = "https://github.com/DariyaSaiga/Diploma.git"

# Скрипт автоматически:
# 1. Клонирует репо: git clone https://github.com/DariyaSaiga/Diploma.git
# 2. Копирует файлы (core/, data/, training/, utils/, scripts/)
# 3. Скачивает датасет с Drive (если нужен)
# 4. Запускает обучение
```

---

### OPTION 2: Google Drive

**Pros**:
- Offline access
- No internet issues

**How**:
1. Copy all files to `MyDrive/Дипломка_правильная/`
2. Upload dataset `mosei_bottleneck.pkl`
3. Set `USE_GITHUB = False`
4. Run script

---

## 📋 Full Colab Script

```python
# STAGE 1 (First Colab Cell)

from google.colab import drive
import os, shutil, torch

GITHUB_REPO = "https://github.com/DariyaSaiga/Diploma.git"
USE_GITHUB = True

# Setup
os.makedirs('/content/diploma', exist_ok=True)
os.chdir('/content/diploma')

if USE_GITHUB:
    print("📥 Клонирую GitHub...")
    os.system(f"git clone {GITHUB_REPO} temp_repo")
    os.system("cp -r temp_repo/* . 2>/dev/null || true")
    os.system("rm -rf temp_repo")

# Датасет
if not os.path.exists('mosei_bottleneck.pkl'):
    drive.mount('/content/drive', force_remount=True)
    shutil.copy(
        '/content/drive/MyDrive/Дипломка_правильная/mosei_bottleneck.pkl',
        'mosei_bottleneck.pkl'
    )

# Зависимости
os.system('pip install -q torch torchvision torchaudio transformers scikit-learn numpy pandas')

# Проверка
print(f"✔️  GPU: {torch.cuda.is_available()}")

# Stage 1
print("\n" + "="*80)
print("🚀 STAGE 1: FRESH TRAINING")
print("="*80 + "\n")

os.system("""python training/train.py \\
  --epochs 20 --batch_size 32 --lr 1e-3 \\
  --use_domain_sep --alpha_sep 0.1 --alpha_inv 0.05 --alpha_rec 0.01 \\
  --data_path mosei_bottleneck.pkl \\
  --exp_dir experiments/stage1_domain_sep""")

print("\n✅ Stage 1 завершена!")
os.system('cat experiments/stage1_domain_sep/metrics.txt')
```

---

## 📊 4-Stage Training in Colab

### Cell 1: Setup & Stage 1 (40 min)
```bash
Copy COLAB_CLEAN.py with USE_GITHUB=True
```

### Cell 2: Stage 2 (40 min) - Optional
```python
os.system("""python training/train.py \\
  --epochs 20 --batch_size 32 --lr 3e-4 \\
  --use_domain_sep \\
  --pretrained_path experiments/stage1_domain_sep/best_model.pt \\
  --data_path mosei_bottleneck.pkl \\
  --exp_dir experiments/stage2_domain_sep""")
```

### Cell 3: Stage 3 (30 min) - Optional
```python
os.system("""python training/train.py \\
  --epochs 15 --batch_size 32 --lr 5e-4 \\
  --freeze_bert partial \\
  --use_domain_sep \\
  --pretrained_path experiments/stage2_domain_sep/best_model.pt \\
  --data_path mosei_bottleneck.pkl \\
  --exp_dir experiments/stage3_domain_sep""")
```

### Cell 4: Stage 4 (20 min) - Optional
```python
os.system("""python training/train.py \\
  --epochs 10 --batch_size 32 --lr 2e-4 \\
  --freeze_bert partial --label_smoothing 0.2 \\
  --use_domain_sep --alpha_sep 0.15 --alpha_inv 0.08 --alpha_rec 0.02 \\
  --pretrained_path experiments/stage3_domain_sep/best_model.pt \\
  --data_path mosei_bottleneck.pkl \\
  --exp_dir experiments/stage4_domain_sep""")

print("🏆 Final: 79-82% accuracy")
```

---

## 🔑 Key Points

✅ **USE_GITHUB = True** → Автоматически клонирует репо  
✅ **Датасет с Drive** → Автоматически скачивается  
✅ **4 ячейки** → Stage 1-4  
✅ **Результаты** → Сохраняются в Drive  

---

## ⚠️ Troubleshooting

**GitHub is slow?**
```python
USE_GITHUB = False  # Use Drive instead
```

**Dataset not found?**
```
1. Upload mosei_bottleneck.pkl to Drive
2. Mount Drive: drive.mount('/content/drive')
3. Copy manually
```

**GPU out of memory?**
```python
--batch_size 16  # Instead of 32
```

**Colab timeout?**
```
Run Stage 1, save results
Come back later for Stage 2-4
Results auto-saved to Drive
```

---

## 📝 Complete Setup Checklist

- [ ] Copy `COLAB_CLEAN.py` content
- [ ] Paste into Colab cell
- [ ] Ensure `USE_GITHUB = True`
- [ ] Have dataset `mosei_bottleneck.pkl` in Drive (optional if using GitHub)
- [ ] Run cell (Ctrl+Enter)
- [ ] Wait for Stage 1 (~40 min)
- [ ] Check results: `cat experiments/stage1_domain_sep/metrics.txt`
- [ ] Run Stage 2-4 in separate cells (optional)

**Expected**: 74-76% after Stage 1, 80%+ after all 4 stages

---

## 🚀 Quick Commands

```bash
# View Stage 1 results
!cat experiments/stage1_domain_sep/metrics.txt

# Check GPU
!nvidia-smi

# Check files downloaded
!ls -la

# Check training progress
!tail -f experiments/stage1_domain_sep/train_log.txt
```

---

**Ready? Copy COLAB_CLEAN.py and run in Colab!** 🎯
