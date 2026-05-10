"""
Inspect mosei_emotion_aligned_60.pkl (CMU-SDK array format).

Structure expected:
    data[split] = {
        'vision':  ndarray (N, T, D_v),
        'audio':   ndarray (N, T, D_a),
        'text':    ndarray (N, T, D_t)  or list of strings,
        'labels':  ndarray (N, ...) ,
        'id':      list of str
    }

Usage:
    python scripts/inspect_dataset.py --path mosei_emotion_aligned_60.pkl
"""
import argparse
import pickle
import os
from collections import Counter

import numpy as np

# ── colours ───────────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def section(title):
    print(f"\n{BOLD}{CYAN}{'='*64}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*64}{RESET}")

def row(label, value, color=GREEN):
    print(f"  {BOLD}{label:<30}{RESET}{color}{value}{RESET}")

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


# ── load ──────────────────────────────────────────────────────────────────────
def load(path):
    size_gb = os.path.getsize(path) / 1e9
    print(f"\n{BOLD}File:{RESET} {path}  ({size_gb:.2f} GB)")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Top-level type : {type(data).__name__}")
    if isinstance(data, dict):
        print(f"  Top-level keys : {list(data.keys())}")
    return data


# ── inspect one split (dict-of-arrays format) ─────────────────────────────────
def inspect_split_arrays(split_data, split_name):
    section(f"Split: '{split_name}'")

    if not isinstance(split_data, dict):
        print(f"  Not a dict — type={type(split_data)}")
        print(f"  Value: {str(split_data)[:300]}")
        return

    print(f"  Keys: {list(split_data.keys())}\n")

    N = None  # number of samples

    for key, val in split_data.items():

        print(f"  {BOLD}── {key} ──{RESET}")

        # ── numpy array ───────────────────────────────────────────────────
        if isinstance(val, np.ndarray):
            N = val.shape[0]
            row("  shape",  str(val.shape))
            row("  dtype",  str(val.dtype))
            row("  N (samples)", str(val.shape[0]))

            if val.ndim >= 2:
                row("  T (time steps)", str(val.shape[1]))
            if val.ndim >= 3:
                row("  D (feature dim)", str(val.shape[2]))

            if np.issubdtype(val.dtype, np.number):
                flat = val.flatten()
                # skip padding zeros for stats
                nonzero = flat[flat != 0]
                row("  min",  f"{flat.min():.6f}")
                row("  max",  f"{flat.max():.6f}")
                row("  mean", f"{flat.mean():.6f}")
                row("  std",  f"{flat.std():.6f}")
                if len(nonzero) > 0:
                    row("  mean (non-zero)", f"{nonzero.mean():.6f}")
                nans  = int(np.isnan(flat).sum())
                infs  = int(np.isinf(flat).sum())
                zeros = int((flat == 0).sum())
                row("  NaN count",  str(nans),  RED if nans  > 0 else GREEN)
                row("  Inf count",  str(infs),  RED if infs  > 0 else GREEN)
                row("  Zero count", f"{zeros}  ({zeros/flat.size*100:.1f}%)")

            # sample values
            if val.ndim == 3 and val.shape[2] <= 12:
                print(f"  Sample[0, 0, :] = {val[0, 0, :]}")
            elif val.ndim == 3:
                print(f"  Sample[0, 0, :8] = {val[0, 0, :8]} ...")
            elif val.ndim == 2:
                print(f"  Sample[0, :8] = {val[0, :8]} ...")
            elif val.ndim == 1:
                unique_vals = np.unique(val)
                if len(unique_vals) <= 20:
                    print(f"  Unique values = {sorted(unique_vals.tolist())}")

        # ── list ──────────────────────────────────────────────────────────
        elif isinstance(val, list):
            N = len(val)
            row("  type",   f"list  len={len(val)}")

            if len(val) == 0:
                print()
                continue

            v0 = val[0]
            row("  element type", type(v0).__name__)

            # list of strings (text or IDs)
            if isinstance(v0, str):
                lens  = [len(x) for x in val]
                wlens = [len(x.split()) for x in val]
                row("  char len range",  f"{min(lens)}–{max(lens)}")
                row("  word len range",  f"{min(wlens)}–{max(wlens)}")
                row("  mean words",      f"{np.mean(wlens):.1f}")
                print(f"  Sample[0] = \"{str(v0)[:120]}\"")
                print(f"  Sample[1] = \"{str(val[1])[:120]}\"")

            # list of arrays
            elif isinstance(v0, np.ndarray):
                shapes = [x.shape for x in val[:200]]
                unique_shapes = set(shapes)
                row("  element shapes",
                    str(shapes[0]) if len(unique_shapes)==1
                    else f"varies — e.g. {sorted(unique_shapes)[:5]}")
                if len(unique_shapes) > 1:
                    T_vals = [s[0] for s in shapes if len(s)>=1]
                    row("  T range", f"{min(T_vals)}–{max(T_vals)}")
                # value stats
                flat = np.concatenate([x.flatten() for x in val[:100]])
                row("  min",  f"{flat.min():.6f}")
                row("  max",  f"{flat.max():.6f}")
                row("  mean", f"{flat.mean():.6f}")
                row("  std",  f"{flat.std():.6f}")

            # list of lists / tuples
            elif isinstance(v0, (list, tuple)):
                row("  element len", str(len(v0)))
                print(f"  Sample[0] = {str(v0)[:120]}")

            else:
                print(f"  Sample[0] = {str(v0)[:120]}")

        # ── other types ───────────────────────────────────────────────────
        else:
            row("  type",  type(val).__name__)
            row("  value", str(val)[:200])

        print()

    # ── label analysis ────────────────────────────────────────────────────
    labels_val = split_data.get("labels")
    if labels_val is not None:
        section(f"Labels analysis — '{split_name}'")

        if isinstance(labels_val, np.ndarray):
            arr = labels_val
        else:
            arr = np.array(labels_val)

        row("Full shape", str(arr.shape))
        row("dtype",      str(arr.dtype))

        # ── case 1: single integer per sample (N,) or (N,1) ──────────────
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            flat = arr.flatten().astype(int)
            cnt  = Counter(flat.tolist())
            total = len(flat)
            print(f"\n  {BOLD}Class distribution  (N={total}):{RESET}")
            print(f"  {'─'*52}")
            for lbl in sorted(cnt):
                name  = EMOTION_NAMES[lbl] if lbl < len(EMOTION_NAMES) else f"class_{lbl}"
                count = cnt[lbl]
                frac  = count / total
                bar   = "█" * max(1, int(frac * 34))
                color = RED if frac < 0.05 else (YELLOW if frac < 0.15 else GREEN)
                print(f"  {lbl} {name:<10} {count:>6}  {frac*100:5.1f}%  {color}{bar}{RESET}")
            imb = max(cnt.values()) / max(min(cnt.values()), 1)
            icolor = RED if imb > 20 else (YELLOW if imb > 5 else GREEN)
            print(f"\n  {BOLD}Num classes  :{RESET} {len(cnt)}")
            print(f"  {BOLD}Imbalance    :{RESET} {icolor}{imb:.1f}x{RESET}  (max / min class)")

        # ── case 2: multi-label / per-frame (N, T) or (N, T, C) ──────────
        elif arr.ndim == 2 and arr.shape[1] > 1:
            C = arr.shape[1]
            print(f"\n  Shape = (N={arr.shape[0]}, C={C})")
            if C <= 7:
                names = EMOTION_NAMES[:C] if C <= 6 else [f"col_{i}" for i in range(C)]
                print(f"  Columns: {names}")
                print(f"\n  {BOLD}{'Emotion':<12} {'min':>8} {'max':>8} {'mean':>8} "
                      f"{'std':>8} {'#unique':>8}{RESET}")
                print(f"  {'─'*58}")
                for i, name in enumerate(names):
                    col = arr[:, i]
                    u   = np.unique(col)
                    print(f"  {name:<12} {col.min():>8.3f} {col.max():>8.3f}"
                          f" {col.mean():>8.3f} {col.std():>8.3f} {len(u):>8}")
                    if len(u) <= 15:
                        print(f"  {'':12} unique={sorted(u.tolist())}")
            else:
                print(f"  {C} columns — printing stats for first 6:")
                for i in range(min(6, C)):
                    col = arr[:, i]
                    print(f"  col[{i}]: min={col.min():.3f}  max={col.max():.3f}"
                          f"  mean={col.mean():.3f}")

        elif arr.ndim == 3:
            print(f"\n  Shape = (N, T, C) = {arr.shape}")
            print(f"  This looks like per-frame labels for each of {arr.shape[2]} emotion classes.")
            for i in range(min(arr.shape[2], 6)):
                col = arr[:, :, i].flatten()
                name = EMOTION_NAMES[i] if i < len(EMOTION_NAMES) else f"class_{i}"
                print(f"  {name:<12} min={col.min():.3f}  max={col.max():.3f}"
                      f"  mean={col.mean():.3f}  std={col.std():.3f}")


# ── cross-split summary ────────────────────────────────────────────────────────
def cross_split_summary(data, splits):
    section("Cross-split summary")
    totals = {}
    for sp in splits:
        v = data[sp]
        # find N
        for key in ["labels", "audio", "vision", "text", "id"]:
            if key in v:
                val = v[key]
                n = val.shape[0] if isinstance(val, np.ndarray) else len(val)
                totals[sp] = n
                break

    grand = sum(totals.values())
    print(f"  {'Split':<8} {'N':>8}  {'%':>6}   Bar")
    print(f"  {'─'*46}")
    for sp in splits:
        n   = totals.get(sp, 0)
        pct = n / grand * 100 if grand else 0
        bar = "█" * int(pct / 2)
        print(f"  {sp:<8} {n:>8}  {pct:>5.1f}%   {GREEN}{bar}{RESET}")
    print(f"\n  {BOLD}Total:{RESET} {grand}")


# ── compare with mosei_bottleneck.pkl ─────────────────────────────────────────
def compare_existing(data, splits, ref_path="mosei_bottleneck.pkl"):
    if not os.path.exists(ref_path):
        return
    try:
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        section(f"Diff vs {ref_path}")
        print(f"  {'Split':<8} {'mosei_bottleneck':>18} {'this file':>12} {'delta':>8}")
        print(f"  {'─'*50}")
        for sp_new, sp_ref in [("train","train"), ("valid","val"), ("test","test")]:
            new_v = data.get(sp_new, {})
            ref_v = ref.get(sp_ref, [])
            # count new
            for key in ["labels","audio","vision","text","id"]:
                if key in new_v:
                    val = new_v[key]
                    new_n = val.shape[0] if isinstance(val, np.ndarray) else len(val)
                    break
            else:
                new_n = 0
            ref_n = len(ref_v)
            diff  = new_n - ref_n
            sign  = "+" if diff >= 0 else ""
            color = GREEN if diff >= 0 else RED
            print(f"  {sp_new:<8} {ref_n:>18} {new_n:>12}  "
                  f"{color}{sign}{diff:>6}{RESET}")
    except Exception as e:
        print(f"  Could not load {ref_path}: {e}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="mosei_emotion_aligned_60.pkl")
    args = parser.parse_args()

    data = load(args.path)

    if not isinstance(data, dict):
        print(f"\nUnexpected top-level type: {type(data)}")
        print(f"Value: {str(data)[:400]}")
        return

    splits = list(data.keys())

    cross_split_summary(data, splits)

    for sp in splits:
        inspect_split_arrays(data[sp], sp)

    compare_existing(data, splits)

    section("Inspection complete")
    print()


if __name__ == "__main__":
    main()