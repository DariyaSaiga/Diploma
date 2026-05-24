import pickle, numpy as np, os

# ✏️ МЕНЯЙ ЗДЕСЬ
PATH = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/mosei_finetune_bert.pkl"

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

# ─────────────────────────────────────────────────────────────────────────────

def sep(title): print(f"\n{'─'*20} {title} {'─'*20}")

def show(arr, name):
    if arr is None: print(f"  {name}: ❌ не найдено"); return
    arr = np.array(arr)
    nan = np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0
    print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}  NaN={nan}")

def find(d, keys):
    for k in keys:
        if k in d: return d[k]
    return None

# ─────────────────────────────────────────────────────────────────────────────

print(f"Файл: {PATH}  ({os.path.getsize(PATH)/1e6:.1f} MB)")

with open(PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# 1. Структура
sep("СТРУКТУРА")
if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  [{k}]  ->  keys: {list(v.keys())}")
        elif isinstance(v, np.ndarray):
            print(f"  [{k}]  ->  shape={v.shape}")
        else:
            print(f"  [{k}]  ->  {type(v).__name__}")
else:
    print(f"  тип: {type(data).__name__}  len={len(data)}")

# 2. Проверить каждый split
SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    if not isinstance(data, dict) or split not in data:
        continue

    sep(f"SPLIT: {split}")
    s = data[split]

    text   = find(s, ["text", "bert", "input_ids", "raw_text"])
    audio  = find(s, ["audio", "covarep", "acoustic"])
    vision = find(s, ["vision", "visual", "video", "facet"])
    labels = find(s, ["labels", "label", "emotions", "y"])

    show(text,   "text  ")
    show(audio,  "audio ")
    show(vision, "vision")
    show(labels, "labels")

    # Labels анализ
    if labels is not None:
        lb = np.array(labels)
        if lb.ndim == 2 and lb.shape[1] == 6:
            lb_bin = (lb > 0.5).astype(int) if lb.max() > 1 else lb.astype(int)
            N = len(lb_bin)
            print(f"\n  Per-class (N={N}):")
            for i, emo in enumerate(EMOTIONS):
                pos = lb_bin[:, i].sum()
                print(f"    {emo:<10}: {pos:>5}  ({100*pos/N:.1f}%)")

            # pos_weight для BCEWithLogitsLoss
            if split == "train":
                sep("pos_weight (BCEWithLogitsLoss)")
                pw = [(N - lb_bin[:,i].sum()) / max(lb_bin[:,i].sum(), 1)
                      for i in range(6)]
                vals = ", ".join(f"{w:.2f}" for w in pw)
                print(f"  pos_weight = torch.tensor([{vals}])")
                print(f"  # {', '.join(EMOTIONS)}")
        else:
            print(f"  ⚠️  labels shape {lb.shape} — не [N,6]")