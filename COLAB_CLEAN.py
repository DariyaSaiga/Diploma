from google.colab import drive
import os, shutil, torch, subprocess

# ⚠️  ВАРИАНТ 1: GitHub (рекомендуется - свежий код)
GITHUB_REPO = "https://github.com/DariyaSaiga/Diploma.git"
USE_GITHUB = True  # Set False если хочешь использовать Drive

# ⚠️  ВАРИАНТ 2: Google Drive (если нет GitHub доступа)
DATASET_PATH = "/content/drive/MyDrive/Дипломка_правильная/mosei_bottleneck.pkl"
DRIVE_CODE_PATH = "/content/drive/MyDrive/Дипломка_правильная"

# Setup
os.makedirs('/content/diploma', exist_ok=True)
os.chdir('/content/diploma')

if USE_GITHUB:
    print("📥 Клонирую репо из GitHub...")
    os.system(f"git clone {GITHUB_REPO} temp_repo")
    os.system("cp -r temp_repo/* . 2>/dev/null || true")
    os.system("rm -rf temp_repo")
    print("✅ Репо клонирован")
else:
    print("📂 Монтирую Google Drive...")
    drive.mount('/content/drive')

    # Копируем код
    print("📋 Копирую файлы из Drive...")
    for folder in ['core', 'data', 'training', 'utils', 'scripts']:
        src_folder = os.path.join(DRIVE_CODE_PATH, folder)
        if os.path.exists(src_folder):
            os.makedirs(folder, exist_ok=True)
            for f in os.listdir(src_folder):
                if f.endswith('.py'):
                    src = os.path.join(src_folder, f)
                    dst = os.path.join(folder, f)
                    shutil.copy(src, dst)
                    print(f"  ✅ {folder}/{f}")

    # Датасет с Drive
    if not os.path.exists('mosei_bottleneck.pkl'):
        shutil.copy(DATASET_PATH, 'mosei_bottleneck.pkl')
    print("✅ Датасет скопирован")

# Если датасета нет, скачиваем с Drive
if not os.path.exists('mosei_bottleneck.pkl'):
    print("\n⚠️  Датасет не найден! Монтирую Drive...")
    drive.mount('/content/drive', force_remount=True)
    dataset_path = "/content/drive/MyDrive/Дипломка_правильная/mosei_bottleneck.pkl"
    if os.path.exists(dataset_path):
        shutil.copy(dataset_path, 'mosei_bottleneck.pkl')
        print("✅ Датасет скопирован с Drive")
    else:
        print("❌ Датасет не найден! Положи mosei_bottleneck.pkl в Drive")

# Зависимости
print("\n🔧 Устанавливаю зависимости...")
os.system('pip install -q torch torchvision torchaudio transformers scikit-learn numpy pandas')

# Проверка
print(f"\n✔️  GPU: {torch.cuda.is_available()}")

# Stage 1
print("\n" + "="*80)
print("🚀 STAGE 1: FRESH TRAINING (20 epochs)")
print("="*80 + "\n")

os.system("""python training/train.py \\
  --epochs 20 --batch_size 32 --lr 1e-3 \\
  --use_domain_sep --alpha_sep 0.1 --alpha_inv 0.05 --alpha_rec 0.01 \\
  --data_path mosei_bottleneck.pkl \\
  --exp_dir experiments/stage1_domain_sep""")

# Результаты
print("\n✅ Stage 1 завершена!")
os.system('cat experiments/stage1_domain_sep/metrics.txt')

# Сохрани в Drive
drive_path = '/content/drive/MyDrive/Дипломка_правильная'
os.makedirs(f'{drive_path}/stage1_domain_sep', exist_ok=True)
os.system(f'cp -r experiments/stage1_domain_sep/* {drive_path}/stage1_domain_sep/')
print(f"\n📤 Результаты в Drive: stage1_domain_sep/")
