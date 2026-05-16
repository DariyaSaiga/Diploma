"""
debug_emotion_model.py
══════════════════════════════════════════════════════════════════════════════
Полный diagnostic toolkit для BottleneckFusionModel (BERT + COVAREP + OpenFace).

Запуск:
    python debug_emotion_model.py --checkpoint best_model.pt

Секции:
    1. Checkpoint inspection       — что внутри .pt файла
    2. Architecture mismatch       — совпадение весов с моделью
    3. Weight health check         — обучена ли модель или random init
    4. Forward pass debug          — логиты ДО softmax, по каждой ветке
    5. Modality ablation           — какая ветка вносит вклад
    6. Input distribution check    — правильный ли preprocessing
    7. Gradient flow               — BackProp проходит или нет
    8. Root cause diagnosis        — итоговый вердикт
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Debug uniform-prediction issue")
parser.add_argument("--checkpoint", default="best_model.pt", help="Путь к .pt файлу")
parser.add_argument("--device", default="cpu", help="cpu / cuda / mps")
parser.add_argument("--skip-bert", action="store_true", help="Не загружать BERT (быстрее)")
args = parser.parse_args()

DEVICE = torch.device(args.device)
PT_PATH = Path(args.checkpoint)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SEP = "═" * 72
sep = "─" * 72

def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):  print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def fail(msg): print(f"  ❌  {msg}")
def info(msg): print(f"  ℹ️   {msg}")

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 1: Checkpoint inspection
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 1 — Checkpoint inspection")

if not PT_PATH.exists():
    fail(f"Файл не найден: {PT_PATH}")
    sys.exit(1)

ok(f"Файл найден: {PT_PATH}  ({PT_PATH.stat().st_size / 1e6:.1f} MB)")

checkpoint = torch.load(PT_PATH, map_location="cpu", weights_only=False)

print(f"\n  Тип объекта: {type(checkpoint).__name__}")

if isinstance(checkpoint, dict):
    print(f"  Ключи верхнего уровня: {list(checkpoint.keys())}")

    # Определяем где лежит state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        info("Формат: {'model_state_dict': ..., ...} — стандартный training checkpoint")
        for k, v in checkpoint.items():
            if k != "model_state_dict":
                print(f"    meta  {k}: {v}")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        info("Формат: {'state_dict': ...}")
    elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
        info("Формат: чистый state_dict (только тензоры)")
    else:
        # Смешанный — ищем тензоры
        tensor_keys = [k for k, v in checkpoint.items() if isinstance(v, torch.Tensor)]
        if tensor_keys:
            state_dict = {k: checkpoint[k] for k in tensor_keys}
            warn(f"Смешанный формат — взяли {len(tensor_keys)} тензорных ключей")
        else:
            fail("Не удалось найти тензоры в checkpoint — неизвестный формат")
            sys.exit(1)
elif isinstance(checkpoint, nn.Module):
    fail("Checkpoint содержит объект модели (torch.save(model)), а не state_dict.")
    info("Нужно: torch.save(model.state_dict(), path) или загружать через model = torch.load(...)")
    sys.exit(1)
else:
    fail(f"Неизвестный тип checkpoint: {type(checkpoint)}")
    sys.exit(1)

print(f"\n  Всего ключей в state_dict: {len(state_dict)}")
print(f"\n  Первые 15 ключей:")
for i, (k, v) in enumerate(state_dict.items()):
    if i >= 15:
        print(f"    ... и ещё {len(state_dict) - 15} ключей")
        break
    print(f"    {k:<60s}  shape={tuple(v.shape)}")

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 2: Architecture mismatch
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 2 — Architecture mismatch (checkpoint vs model)")

try:
    from models import BottleneckFusionModel
    ok("models.py импортирован")
except ImportError as e:
    fail(f"Не удалось импортировать models.py: {e}")
    sys.exit(1)

if args.skip_bert:
    warn("--skip-bert: BERT не загружается. Mismatch по bert.* ключам ожидаем — игнорируем.")

model = BottleneckFusionModel()
model_sd = model.state_dict()

model_keys = set(model_sd.keys())
ckpt_keys  = set(state_dict.keys())

missing    = model_keys - ckpt_keys   # есть в модели, нет в checkpoint
unexpected = ckpt_keys - model_keys   # есть в checkpoint, нет в модели

print(f"\n  Ключей в модели    : {len(model_keys)}")
print(f"  Ключей в checkpoint: {len(ckpt_keys)}")

if not missing and not unexpected:
    ok("Полное совпадение ключей — архитектура и checkpoint идентичны")
else:
    if missing:
        warn(f"Missing keys ({len(missing)}) — в модели есть, в checkpoint нет:")
        bert_missing = [k for k in sorted(missing) if k.startswith("bert.")]
        other_missing = [k for k in sorted(missing) if not k.startswith("bert.")]
        if bert_missing:
            info(f"  BERT ключей нет в checkpoint: {len(bert_missing)} шт — "
                 "если обучали без fine-tune BERT, это ОК")
        for k in other_missing[:20]:
            print(f"    MISSING  {k}")
        if len(other_missing) > 20:
            print(f"    ... и ещё {len(other_missing) - 20}")

    if unexpected:
        warn(f"Unexpected keys ({len(unexpected)}) — в checkpoint есть, в модели нет:")
        for k in sorted(unexpected)[:20]:
            print(f"    UNEXPECTED  {k}")
        if len(unexpected) > 20:
            print(f"    ... и ещё {len(unexpected) - 20}")

# Shape mismatch — самая частая причина silent broken model
shape_mismatches = []
for k in model_keys & ckpt_keys:
    ms = tuple(model_sd[k].shape)
    cs = tuple(state_dict[k].shape)
    if ms != cs:
        shape_mismatches.append((k, ms, cs))

if shape_mismatches:
    fail(f"Shape mismatch в {len(shape_mismatches)} слоях — КРИТИЧНО:")
    for k, ms, cs in shape_mismatches[:15]:
        print(f"    {k}")
        print(f"      model: {ms}")
        print(f"      ckpt : {cs}")
else:
    ok("Shape mismatch: нет — все совпадающие ключи имеют одинаковые shape")

# Загружаем веса
load_result = model.load_state_dict(state_dict, strict=False)
ok(f"load_state_dict(strict=False) выполнен: "
   f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")

model.to(DEVICE).eval()

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 3: Weight health check — обучена ли модель?
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 3 — Weight health check (random init или обученная?)")

print()
# Ключевые не-BERT слои, которые должны быть обучены
key_layers = {
    "classifier.0.weight":  "Финальный classifier (Linear)",
    "classifier.3.weight":  "Финальный classifier head",
    "head_text.weight":     "Auxiliary text head",
    "head_audio.weight":    "Auxiliary audio head",
    "head_vision.weight":   "Auxiliary vision head",
    "bottleneck":           "Learnable bottleneck tokens",
    "audio_proj.proj.0.weight":  "Audio Conv1D projection",
    "vision_proj.proj.0.weight": "Vision Conv1D projection",
    "text_proj.weight":     "Text Linear projection",
}

all_random = True  # будем опровергать

for layer_key, description in key_layers.items():
    if layer_key not in state_dict:
        warn(f"  {description:<40s} — НЕТ в checkpoint")
        continue

    w = state_dict[layer_key].float()
    mean = w.mean().item()
    std  = w.std().item()
    absmax = w.abs().max().item()

    # Эвристика: random init от torch.randn имеет std≈1, обученный — std меньше
    # nn.Linear kaiming_uniform init: std ≈ sqrt(2/fan_in), обычно 0.05-0.5
    # Если std > 0.9 — скорее всего random
    is_random = (std > 0.8) or (absmax > 5.0 and std > 0.5)

    if is_random:
        fail(f"  {description:<40s}  mean={mean:+.4f}  std={std:.4f}  max={absmax:.4f}  "
             f"← похоже на random init!")
    else:
        ok(f"  {description:<40s}  mean={mean:+.4f}  std={std:.4f}  max={absmax:.4f}")
        all_random = False

print()
if all_random:
    fail("ВСЕ ключевые слои выглядят как random init — модель скорее всего НЕ ОБУЧЕНА")
    fail("или загружается неправильный checkpoint")
else:
    ok("Веса не похожи на random init — модель обучена")

# Проверяем norm по всем не-BERT параметрам
non_bert_norms = []
for k, v in state_dict.items():
    if not k.startswith("bert.") and v.dtype in (torch.float32, torch.float16, torch.bfloat16):
        non_bert_norms.append(v.float().norm().item())

if non_bert_norms:
    total_norm = sum(non_bert_norms)
    print(f"\n  Суммарная L2-норма не-BERT параметров: {total_norm:.2f}")
    if total_norm < 1.0:
        warn("Очень маленькая норма — возможно веса не загрузились или collapsed training")
    elif total_norm > 10000:
        warn("Очень большая норма — возможно gradient explosion при обучении")
    else:
        ok(f"Норма параметров в разумных пределах")

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 4: Forward pass debug — логиты ДО softmax
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 4 — Forward pass debug (логиты до softmax)")

# Создаём тестовые входы
B = 1
dummy_ids   = torch.zeros(B, 50, dtype=torch.long, device=DEVICE)
dummy_mask  = torch.ones(B, 50, dtype=torch.long, device=DEVICE)   # все токены "реальные"
dummy_audio = torch.zeros(B, 60, 74, device=DEVICE)
dummy_vision= torch.zeros(B, 60, 35, device=DEVICE)
dummy_amask = torch.ones(B, 60, dtype=torch.long, device=DEVICE)
dummy_vmask = torch.ones(B, 60, dtype=torch.long, device=DEVICE)

# Хук для захвата промежуточных значений
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach()
        elif isinstance(output, tuple):
            activations[name] = output[0].detach()
    return hook

# Регистрируем хуки на ключевых слоях
hooks = []
hooks.append(model.text_proj.register_forward_hook(make_hook("text_proj_out")))
hooks.append(model.audio_proj.register_forward_hook(make_hook("audio_proj_out")))
hooks.append(model.vision_proj.register_forward_hook(make_hook("vision_proj_out")))
hooks.append(model.classifier.register_forward_hook(make_hook("classifier_out")))

print("\n  [Режим: все входы = zeros/padding]\n")

with torch.no_grad():
    try:
        outputs = model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            audio=dummy_audio,
            vision=dummy_vision,
            audio_mask=dummy_amask,
            vision_mask=dummy_vmask,
        )
        logits_fuse, logits_text, logits_audio, logits_vision = outputs
    except Exception as e:
        fail(f"Forward pass упал: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

for hook in hooks:
    hook.remove()

def analyze_logits(name, logits):
    l = logits[0].cpu().float()
    probs = torch.softmax(l, dim=-1)
    entropy = -(probs * (probs + 1e-9).log()).sum().item()
    max_entropy = np.log(len(EMOTIONS))
    entropy_ratio = entropy / max_entropy  # 1.0 = полностью uniform

    print(f"\n  {name}:")
    print(f"    raw logits : {[f'{x:.4f}' for x in l.tolist()]}")
    print(f"    softmax    : {[f'{x:.4f}' for x in probs.tolist()]}")
    print(f"    логит range: min={l.min():.4f}, max={l.max():.4f}, std={l.std():.4f}")
    print(f"    entropy    : {entropy:.4f} / {max_entropy:.4f} (ratio={entropy_ratio:.3f})")

    if entropy_ratio > 0.99:
        fail(f"    → UNIFORM! Энтропия максимальная — логиты почти одинаковые")
    elif entropy_ratio > 0.90:
        warn(f"    → Почти uniform, очень неуверенная предсказание")
    else:
        pred_idx = l.argmax().item()
        ok(f"    → Предсказание: '{EMOTIONS[pred_idx]}' ({probs[pred_idx]*100:.1f}%)")

    return entropy_ratio

ratios = {}
ratios["fuse"]   = analyze_logits("logits_fuse   (основной)",  logits_fuse)
ratios["text"]   = analyze_logits("logits_text   (auxiliary)", logits_text)
ratios["audio"]  = analyze_logits("logits_audio  (auxiliary)", logits_audio)
ratios["vision"] = analyze_logits("logits_vision (auxiliary)", logits_vision)

print("\n  Промежуточные активации:")
for name, act in activations.items():
    a = act.float()
    print(f"    {name:<25s}  shape={tuple(a.shape)}  "
          f"mean={a.mean():.4f}  std={a.std():.4f}  "
          f"zeros={(a == 0).float().mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 5: Modality ablation
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 5 — Modality ablation (какая ветка влияет на результат?)")

print()

def run_forward_entropy(**kwargs) -> float:
    """Запустить forward и вернуть entropy ratio logits_fuse."""
    base = dict(
        input_ids=dummy_ids,
        attention_mask=dummy_mask,
        audio=dummy_audio,
        vision=dummy_vision,
        audio_mask=dummy_amask,
        vision_mask=dummy_vmask,
    )
    base.update(kwargs)
    with torch.no_grad():
        out = model(**base)
    l = out[0][0].cpu().float()
    p = torch.softmax(l, dim=-1)
    return (-(p * (p + 1e-9).log()).sum().item()) / np.log(len(EMOTIONS))

# Базовый (все нули)
e_base = run_forward_entropy()

# Случайный аудио
rand_audio  = torch.randn(B, 60, 74, device=DEVICE)
rand_vision = torch.randn(B, 60, 35, device=DEVICE)
# "Реальный" текст — слово "happy"
from transformers import BertTokenizer
try:
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tok("I am so happy today", max_length=50, padding="max_length",
              truncation=True, return_tensors="pt")
    real_ids  = enc["input_ids"].to(DEVICE)
    real_mask = enc["attention_mask"].to(DEVICE)
    has_bert = True
except Exception as e:
    warn(f"BERT tokenizer недоступен: {e}")
    real_ids, real_mask = dummy_ids, dummy_mask
    has_bert = False

conditions = [
    ("все нули (baseline)",        dict()),
    ("random audio, нулевые ост.", dict(audio=rand_audio)),
    ("random vision, нулевые ост.",dict(vision=rand_vision)),
    ("random audio + vision",      dict(audio=rand_audio, vision=rand_vision)),
]

if has_bert:
    conditions += [
        ("реальный текст 'happy'",     dict(input_ids=real_ids, attention_mask=real_mask)),
        ("всё реальное + rand av",     dict(input_ids=real_ids, attention_mask=real_mask,
                                            audio=rand_audio, vision=rand_vision)),
    ]

print(f"  {'Условие':<45s}  {'Entropy ratio':>14s}  {'Результат'}")
print(f"  {sep}")

for label, kwargs in conditions:
    e = run_forward_entropy(**kwargs)
    delta = e - e_base
    status_str = "UNIFORM" if e > 0.99 else ("почти uniform" if e > 0.90 else "✅ НЕ UNIFORM")

    # Предсказание
    base = dict(
        input_ids=real_ids if "реальный" in label else dummy_ids,
        attention_mask=real_mask if "реальный" in label else dummy_mask,
        audio=kwargs.get("audio", dummy_audio),
        vision=kwargs.get("vision", dummy_vision),
        audio_mask=dummy_amask, vision_mask=dummy_vmask,
    )
    with torch.no_grad():
        out = model(**base)
    pred_label = EMOTIONS[out[0][0].argmax().item()]
    conf = torch.softmax(out[0][0].float(), dim=-1).max().item() * 100

    print(f"  {label:<45s}  {e:>12.4f}    {status_str}  → {pred_label} ({conf:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 6: Input distribution check
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 6 — Input distribution check (правильный ли preprocessing?)")

print("""
  COVAREP (audio, 74-dim) — ожидаемые статистики тренировочного датасета CMU-MOSI:
  ─────────────────────────────────────────────────────────────────────────────
  • Данные должны быть Z-нормализованы по тренировочному set:
      mean ≈ 0.0  (после нормализации)
      std  ≈ 1.0
  • НЕ нормализованные COVAREP: pitch ≈ 100-300 Гц, energy ≈ большие значения
  • КРИТИЧНО: если модель обучена на нормализованных данных, а ты подаёшь сырые —
    распределение не совпадёт, и модель выдаст uniform

  OpenFace (vision, 35-dim AU признаки) — ожидаемые статистики:
  ─────────────────────────────────────────────────────────────────────────────
  • AU intensity: значения обычно в диапазоне [0, 5]
  • AU presence:  бинарные {0, 1}
  • После нормализации: mean ≈ 0, std ≈ 1
  • Если подавать zeros — все AU = 0 = "нейтральное лицо",
    но модель обучена на распределении, где 0 = нет активации мышцы

  Что проверить:
  ─────────────────────────────────────────────────────────────────────────────
  1. Посмотри как нормализованы данные в dataset.py (transform, scaler)
  2. Сохранены ли mean/std нормализации вместе с моделью?
  3. Применяешь ли ту же нормализацию на inference?
""")

# Проверяем есть ли нормализационные параметры в checkpoint
if isinstance(checkpoint, dict):
    norm_keys = [k for k in checkpoint.keys()
                 if any(x in k.lower() for x in ["norm", "mean", "std", "scaler", "stat"])]
    if norm_keys:
        ok(f"В checkpoint найдены нормализационные ключи: {norm_keys}")
        for k in norm_keys:
            v = checkpoint[k]
            print(f"    {k}: {v}")
    else:
        warn("Нормализационные параметры НЕ сохранены в checkpoint")
        warn("Убедись что используешь те же mean/std что при обучении")

# Симуляция правильных vs неправильных входов
print("\n  Симуляция: как меняется уверенность при разных масштабах входа:\n")
print(f"  {'Входные данные':<40s}  {'Max prob':>10s}  {'Entropy ratio':>14s}")
print(f"  {sep}")

test_inputs = [
    ("audio zeros (нет данных)",    torch.zeros(B, 60, 74)),
    ("audio N(0,1) — нормализован", torch.randn(B, 60, 74)),
    ("audio N(0,0.1) — сжатый",     torch.randn(B, 60, 74) * 0.1),
    ("audio N(0,10) — большой",     torch.randn(B, 60, 74) * 10),
    ("vision zeros",                torch.zeros(B, 60, 35)),
    ("vision N(0,1)",               torch.randn(B, 60, 35)),
    ("vision uniform [0,5] AU",     torch.rand(B, 60, 35) * 5),
]

for label, inp in test_inputs:
    is_audio = "audio" in label
    with torch.no_grad():
        out = model(
            input_ids=dummy_ids, attention_mask=dummy_mask,
            audio=inp.to(DEVICE) if is_audio else dummy_audio,
            vision=dummy_vision if is_audio else inp.to(DEVICE),
            audio_mask=dummy_amask, vision_mask=dummy_vmask,
        )
    p = torch.softmax(out[0][0].float(), dim=-1)
    e = (-(p * (p + 1e-9).log()).sum().item()) / np.log(len(EMOTIONS))
    print(f"  {label:<40s}  {p.max().item()*100:>9.2f}%  {e:>14.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 7: Gradient flow check
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 7 — Gradient flow check")

model.train()  # включаем grad
test_ids    = torch.zeros(B, 50, dtype=torch.long, device=DEVICE)
test_mask   = torch.ones(B, 50, dtype=torch.long, device=DEVICE)
test_audio  = torch.randn(B, 60, 74, device=DEVICE)
test_vision = torch.randn(B, 60, 35, device=DEVICE)

test_audio.requires_grad_(True)
test_vision.requires_grad_(True)

outputs = model(
    input_ids=test_ids, attention_mask=test_mask,
    audio=test_audio, vision=test_vision,
    audio_mask=dummy_amask, vision_mask=dummy_vmask,
)

# Loss на случайную метку
loss = nn.CrossEntropyLoss()(outputs[0], torch.tensor([0], device=DEVICE))
loss.backward()

print(f"\n  Loss value: {loss.item():.4f}")
if abs(loss.item() - np.log(len(EMOTIONS))) < 0.1:
    warn(f"Loss ≈ log({len(EMOTIONS)}) = {np.log(len(EMOTIONS)):.4f} — "
         f"типично для uniform prediction / random init")
else:
    ok(f"Loss не равен случайному — модель дискриминативна")

print("\n  Градиенты по ключевым параметрам:\n")
grad_issues = []
for name, param in model.named_parameters():
    if param.grad is None:
        if not name.startswith("bert."):
            grad_issues.append(name)
        continue
    gnorm = param.grad.norm().item()
    if gnorm == 0:
        warn(f"  {name:<55s}  grad_norm=0  ← МЁРТВЫЙ НЕЙРОН / нет потока")
    elif gnorm > 100:
        warn(f"  {name:<55s}  grad_norm={gnorm:.4f}  ← ВЗРЫВ ГРАДИЕНТА")
    elif name in ("bottleneck", "classifier.0.weight", "classifier.3.weight",
                  "head_audio.weight", "head_vision.weight", "head_text.weight"):
        ok(f"  {name:<55s}  grad_norm={gnorm:.6f}")

if grad_issues:
    warn(f"\n  {len(grad_issues)} параметров без градиента (не BERT):")
    for n in grad_issues[:10]:
        print(f"    {n}")

model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# СЕКЦИЯ 8: Root cause diagnosis
# ─────────────────────────────────────────────────────────────────────────────
header("СЕКЦИЯ 8 — Root cause diagnosis")

print()

causes = []
fixes  = []

# Диагноз 1: shape mismatch
if shape_mismatches:
    causes.append("SHAPE MISMATCH: архитектура не совпадает с checkpoint")
    fixes.append(
        "→ Убедись что models.py совпадает с той версией что была при обучении\n"
        "  → Проверь HIDDEN_DIM, N_HEADS, N_BOTTLENECK, N_LAYERS, N_EMOTIONS"
    )

# Диагноз 2: all uniform even with random input
all_uniform = all(r > 0.99 for r in ratios.values())
if all_uniform:
    causes.append("ALL UNIFORM: даже при случайных входах все логиты одинаковые")
    fixes.append(
        "→ Модель либо не обучена (random init), либо загружается неправильный checkpoint\n"
        "  → Проверь имя файла, путь, версию\n"
        "  → Попробуй: print(sum(p.numel() for p in model.parameters()))\n"
        "             print(len(state_dict)) — числа должны совпадать"
    )

# Диагноз 3: zeros input
if ratios["fuse"] > 0.99 and all_uniform:
    causes.append("ZERO INPUTS: все модальности получают нули")
    fixes.append(
        "→ Для frame endpoint: OpenFace и COVAREP не доступны из raw image/video\n"
        "  → Используй /api/analyze/multimodal с настоящими признаками\n"
        "  → Или добавь text='...' хотя бы для текстовой ветки"
    )

# Диагноз 4: weight norms
if all_random:
    causes.append("RANDOM WEIGHTS: ключевые слои выглядят как random init")
    fixes.append(
        "→ torch.load() мог загрузить неправильный файл\n"
        "  → Проверь: checkpoint['epoch'] или checkpoint['val_acc'] — есть ли metadata обучения?\n"
        "  → Попробуй другой .pt файл"
    )

# Диагноз 5: missing keys (не BERT)
non_bert_missing = [k for k in load_result.missing_keys if not k.startswith("bert.")]
if non_bert_missing:
    causes.append(f"MISSING KEYS: {len(non_bert_missing)} не-BERT слоёв нет в checkpoint")
    fixes.append(
        "→ Эти слои останутся с random init:\n  "
        + "\n  ".join(non_bert_missing[:5])
        + ("\n  ..." if len(non_bert_missing) > 5 else "")
    )

print(f"  Найдено возможных причин: {len(causes)}\n")
for i, (c, f) in enumerate(zip(causes, fixes), 1):
    fail(f"  [{i}] {c}")
    print(f"       {f}\n")

if not causes:
    ok("Явных проблем не обнаружено — модель загружена корректно")
    if all(r > 0.95 for r in ratios.values()):
        warn("Но предсказания всё равно uniform при нулевых входах")
        print("""
  ВЕРОЯТНАЯ ПРИЧИНА: input distribution mismatch

  Модель обучена на CMU-MOSI/MOSEI признаках:
  • COVAREP (74-dim): Z-нормализованные акустические признаки
  • OpenFace (35-dim): AU intensity + presence признаки

  При inference ты подаёшь нули — это не "нейтральное состояние",
  это OOD (out-of-distribution) вход, на котором модель не обучалась.

  РЕШЕНИЕ:
  1. Извлечь настоящие COVAREP признаки через opensmile / COVAREP toolkit
  2. Извлечь настоящие OpenFace признаки через OpenFace 2.0 / py-feat
  3. Применить ту же Z-нормализацию (mean/std) что использовалась при обучении
  4. Подать через /api/analyze/multimodal или /api/analyze/multimodal/csv
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Итоговый чеклист
# ─────────────────────────────────────────────────────────────────────────────
header("ИТОГОВЫЙ DEBUG CHECKLIST")

print("""
  □  1. Checkpoint format
         checkpoint = torch.load("best_model.pt", map_location="cpu")
         print(type(checkpoint))          # должен быть dict
         print(checkpoint.keys())         # ищи 'model_state_dict', 'epoch', 'val_acc'

  □  2. Shape совпадение
         model = BottleneckFusionModel()
         result = model.load_state_dict(state_dict, strict=False)
         print(result.missing_keys)       # должен быть пустым (кроме bert.* если не fine-tuned)
         print(result.unexpected_keys)    # должен быть пустым

  □  3. Веса не random
         w = state_dict["classifier.3.weight"]
         print(w.std())  # > 0.01 и < 0.8 — скорее всего обучено

  □  4. Логиты ДО softmax
         logits_fuse, *_ = model(...)
         print(logits_fuse)               # должны РАЗЛИЧАТЬСЯ между классами
         print(logits_fuse.std())         # если < 0.01 — collapsed

  □  5. Входы не нули
         print(audio.abs().mean())        # должно быть != 0
         print(vision.abs().mean())       # должно быть != 0

  □  6. Нормализация входов
         # Применяй те же mean/std что при обучении
         audio_norm  = (audio  - audio_mean)  / audio_std
         vision_norm = (vision - vision_mean) / vision_std

  □  7. Текст через tokenizer
         tok = BertTokenizer.from_pretrained("bert-base-uncased")
         enc = tok(text, max_length=50, padding="max_length", truncation=True)
         # НЕ подавай нулевые ids — подавай реальный текст

  □  8. Проверь entropy
         probs   = torch.softmax(logits_fuse, dim=-1)
         entropy = -(probs * probs.log()).sum()
         print(entropy)  # < log(6) = 1.79 означает уверенность модели

  □  9. Auxiliary heads
         # Если logits_text имеет уверенность, а logits_fuse — нет:
         # проблема в fusion, а не в text encoder
         # Если все 4 head uniform — проблема в weights/inputs

  □ 10. Один реальный пример из датасета
         batch = next(iter(val_loader))
         with torch.no_grad():
             out = model(**{k: v.to(device) for k, v in batch.items()
                           if k in ['input_ids','attention_mask','audio','vision',
                                    'audio_mask','vision_mask']})
         # Если на реальных данных из датасета тоже uniform — модель не обучена
         # Если нет — проблема только в inference preprocessing
""")

print(f"\n{SEP}")
print("  Debug завершён. Запусти с реальным .pt файлом: python debug_emotion_model.py --checkpoint best_model.pt")
print(SEP)