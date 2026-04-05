"""
compare_results.py — читает metrics.txt из каждого эксперимента
и выводит сравнительную таблицу.

Использование:
    python experiments/compare_results.py
"""

import os
import re

EXPERIMENTS = [
    # ── Базовые эксперименты ─────────────────────────────────────────────
    ("exp_01_base_bottleneck",  "n=8,  lr=1e-3, layers=2, frozen",       "baseline"),
    ("exp_02_tokens16",         "n=16, lr=1e-3, layers=2, frozen",       "more tokens"),
    ("exp_03_lr5e4",            "n=8,  lr=5e-4, layers=2, frozen",       "lower lr"),
    ("exp_04_tokens16_lr5e4",   "n=16, lr=5e-4, layers=2, frozen",       "best combined"),
    ("exp_05_partial_unfreeze", "n=16, lr=5e-4, layers=2, partial BERT", "partial BERT"),
    # ── Ablation модальностей ────────────────────────────────────────────
    ("exp_06A_no_visual",       "n=8,  lr=1e-3, layers=2, no visual",    "ablation"),
    ("exp_06B_no_audio",        "n=8,  lr=1e-3, layers=2, no audio",     "ablation"),
    # ── Новые эксперименты (multi-layer MBT) ────────────────────────────
    ("exp_08_multilayer_lr3e4", "n=16, lr=3e-4, layers=2, ep=20",        "MBT main"),
    ("exp_09_early_stop_p3",    "n=16, lr=3e-4, layers=2, pat=3",        "early stop"),
    ("exp_10_sampler",          "n=16, lr=3e-4, layers=2, sampler",      "WRS sampler"),
]

BASE_DIR = os.path.join(os.path.dirname(__file__))


def parse_metrics(path: str) -> dict:
    result = {}
    try:
        with open(path) as f:
            text = f.read()
        m = re.search(r"Best epoch\s+:\s+(\d+)", text)
        if m:
            result["best_epoch"] = int(m.group(1))
        m = re.search(r"Best val macro F1\s+:\s+([\d.]+)", text)
        if m:
            result["val_f1"] = float(m.group(1))
        m = re.search(r"Test accuracy\s+:\s+([\d.]+)", text)
        if m:
            result["test_acc"] = float(m.group(1))
        m = re.search(r"Test macro F1\s+:\s+([\d.]+)", text)
        if m:
            result["test_f1"] = float(m.group(1))
        # per-class F1 — порядок совпадает с EMOTION_NAMES в train.py
        for emotion in ["happy", "sad", "anger", "surprise", "disgust", "fear"]:
            pattern = rf"{emotion}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            m = re.search(pattern, text)
            if m:
                result[f"{emotion}_f1"] = float(m.group(3))
    except FileNotFoundError:
        pass
    return result


def main():
    print("\n" + "=" * 100)
    print(f"{'Exp':<28} {'Config':<35} {'ValF1':>7} {'TestAcc':>8} {'TestF1':>8} "
          f"{'ep':>3}  {'Note'}")
    print("=" * 100)

    best_f1 = -1
    best_exp = ""

    rows = []
    for exp_id, config, note in EXPERIMENTS:
        metrics_path = os.path.join(BASE_DIR, exp_id, "metrics.txt")
        m = parse_metrics(metrics_path)

        if not m:
            rows.append((exp_id, config, note, None))
            print(f"{exp_id:<28} {config:<35}  {'—':>7}  {'—':>8}  {'—':>8}  {'—':>3}  {note} (не запущен)")
            continue

        val_f1   = m.get("val_f1",    0.0)
        test_acc = m.get("test_acc",  0.0)
        test_f1  = m.get("test_f1",   0.0)
        epoch    = m.get("best_epoch", 0)

        rows.append((exp_id, config, note, m))

        marker = " ← BEST" if test_f1 > best_f1 else ""
        if test_f1 > best_f1:
            best_f1  = test_f1
            best_exp = exp_id

        print(f"{exp_id:<28} {config:<35} {val_f1:>7.4f} {test_acc:>8.4f} "
              f"{test_f1:>8.4f} {epoch:>3}  {note}{marker}")

    print("=" * 100)
    print(f"\n🏆 Лучший по test macro F1: {best_exp}  (F1={best_f1:.4f})")

    # ── Per-class F1 по всем завершённым экспериментам ────────────────────
    emotions = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
    completed = [(e, c, n, m) for e, c, n, m in rows if m]

    if completed:
        print("\n── Per-class F1 ──────────────────────────────────────────────────────")
        header = f"{'Exp':<28}" + "".join(f"{em[:5]:>8}" for em in emotions)
        print(header)
        print("-" * (28 + 8 * len(emotions)))
        for exp_id, config, note, m in completed:
            row = f"{exp_id:<28}"
            for em in emotions:
                val = m.get(f"{em}_f1", 0.0)
                row += f"{val:>8.3f}"
            print(row)

    # ── Незавершённые эксперименты ─────────────────────────────────────────
    missing = [e for e, c, n, m in rows if m is None]
    if missing:
        print(f"\n⏳ Ещё не запущены / нет metrics.txt:")
        for e in missing:
            print(f"   - {e}")


if __name__ == "__main__":
    main()
