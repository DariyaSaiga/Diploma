"""
compare_baselines.py — сравнительная таблица baseline-моделей.

Структура сравнения (2 уровня):
  Уровень 1 — отдельные модальности:
    B01: Text-only BERT
    B02: Audio + Visual
  Уровень 2 — простой мультимодальный fusion:
    B03: SimpleFusion (concat без bottleneck)

Результаты Bottleneck-моделей смотри в: python experiments/compare_results.py

Использование:
    python experiments/compare_baselines.py
"""

import os
import re

# ── Описание моделей ──────────────────────────────────────────────────────────
BASELINES = [
    # (folder, short_name, description, level)
    ("B01_text_only",     "B01 Text-BERT",    "BERT-CLS → Linear(768→6)",              1),
    ("B02_audio_visual",  "B02 Audio+Visual", "MeanPool audio + visual → concat → MLP", 1),
    ("B03_simple_fusion", "B03 SimpleFusion", "BERT + audio + visual → concat → MLP",  2),
]

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

BASE_DIR = os.path.dirname(__file__)


# ── Парсинг metrics.txt ───────────────────────────────────────────────────────
def parse_metrics(path: str) -> dict:
    result = {}
    try:
        with open(path) as f:
            text = f.read()

        for key, pattern in [
            ("best_epoch", r"Best epoch\s+:\s+(\d+)"),
            ("val_f1",     r"Best val macro F1\s+:\s+([\d.]+)"),
            ("test_acc",   r"Test accuracy\s+:\s+([\d.]+)"),
            ("test_f1",    r"Test macro F1\s+:\s+([\d.]+)"),
            ("test_wa",    r"Test weighted F1\s+:\s+([\d.]+)"),
        ]:
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

        for emotion in EMOTIONS:
            m = re.search(rf"{emotion}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
            if m:
                result[f"{emotion}_p"]  = float(m.group(1))
                result[f"{emotion}_r"]  = float(m.group(2))
                result[f"{emotion}_f1"] = float(m.group(3))

    except FileNotFoundError:
        pass
    return result


# ── Форматирование ────────────────────────────────────────────────────────────
def fmt(val, fmt_str=".4f"):
    if val is None:
        return "  —   "
    return format(val, fmt_str)


def main():
    all_rows = []
    for folder, short_name, desc, level in BASELINES:
        metrics_path = os.path.join(BASE_DIR, folder, "metrics.txt")
        m = parse_metrics(metrics_path)
        all_rows.append({
            "folder":     folder,
            "short_name": short_name,
            "desc":       desc,
            "level":      level,
            "metrics":    m,
            "done":       bool(m),
        })

    # ── Главная таблица ───────────────────────────────────────────────────────
    W = 108
    print("\n" + "═" * W)
    print("  BASELINE COMPARISON — Multimodal Emotion Recognition (CMU-MOSEI)")
    print("═" * W)
    print(f"  {'Model':<22} {'Description':<40} {'ValF1':>7} {'TestAcc':>8} {'TestF1':>8} {'Ep':>3}  Level")
    print("─" * W)

    prev_level = 0
    best_f1, best_name = -1.0, ""

    for row in all_rows:
        lvl = row["level"]
        if lvl != prev_level:
            level_labels = {1: "── Уровень 1: отдельные модальности ──",
                            2: "── Уровень 2: простой concat-fusion ──"}
            print(f"\n  {level_labels[lvl]}")
            prev_level = lvl

        m = row["metrics"]
        name = row["short_name"]
        desc = row["desc"]

        if not row["done"]:
            print(f"  {name:<22} {desc:<40}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>3}  L{lvl}  (не запущен)")
            continue

        val_f1   = m.get("val_f1")
        test_acc = m.get("test_acc")
        test_f1  = m.get("test_f1")
        epoch    = m.get("best_epoch", 0)

        if test_f1 and test_f1 > best_f1:
            best_f1, best_name = test_f1, name

        marker = "  ← BEST" if test_f1 == best_f1 else ""
        print(f"  {name:<22} {desc:<40} {fmt(val_f1):>7} {fmt(test_acc):>8} {fmt(test_f1):>8} "
              f"{epoch:>3}  L{lvl}{marker}")

    print("\n" + "═" * W)
    if best_name:
        print(f"  🏆 Лучший по test macro F1: {best_name}  (F1 = {best_f1:.4f})")

    # ── Delta: SimpleFusion vs лучший из уровня 1 ────────────────────────────
    done = [r for r in all_rows if r["done"]]
    if len(done) >= 2:
        level_best: dict[int, float] = {}
        for row in done:
            lvl = row["level"]
            f1  = row["metrics"].get("test_f1", 0.0)
            if f1 > level_best.get(lvl, 0.0):
                level_best[lvl] = f1

        if 1 in level_best and 2 in level_best:
            delta = level_best[2] - level_best[1]
            sign  = "+" if delta >= 0 else ""
            print("\n" + "─" * W)
            print(f"  DELTA  L1 → L2  (SimpleFusion vs лучший unimodal):  {sign}{delta:.4f}")

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  PER-CLASS F1 (test set):\n")
    em_header = f"  {'Model':<22}" + "".join(f"{e[:5]:>8}" for e in EMOTIONS)
    print(em_header)
    print("  " + "-" * (22 + 8 * len(EMOTIONS)))

    for row in all_rows:
        if not row["done"]:
            row_str = f"  {row['short_name']:<22}" + "  —" * len(EMOTIONS)
        else:
            row_str = f"  {row['short_name']:<22}"
            for em in EMOTIONS:
                val = row["metrics"].get(f"{em}_f1")
                row_str += f"  {fmt(val, '.3f'):>6}"
        print(row_str)

    # ── Итоговый вывод ────────────────────────────────────────────────────────
    missing = [r["folder"] for r in all_rows if not r["done"]]
    if missing:
        print("\n" + "─" * W)
        print("  ⏳ Ещё не запущены:")
        for f in missing:
            print(f"     - experiments/{f}/  (нет metrics.txt)")
        print(f"\n  Запустить всё:  cd Diploma/ && bash experiments/run_baselines.sh")

    print("\n" + "═" * W + "\n")


if __name__ == "__main__":
    main()
