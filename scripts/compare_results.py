"""Compare metrics across experiments/.

Reads metrics.json (or falls back to metrics.txt) from every
experiments/exp* directory and prints a summary table.

Usage:
    PYTHONPATH=. python scripts/compare_results.py
    PYTHONPATH=. python scripts/compare_results.py --exp_root experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


METRIC_KEYS = ["micro_f1", "macro_f1", "weighted_f1", "hamming_loss", "subset_accuracy"]


def _from_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return None
    test = payload.get("test", {})
    return {
        "best_epoch": payload.get("best_epoch"),
        "threshold": payload.get("threshold"),
        **{k: test.get(k) for k in METRIC_KEYS},
    }


def _from_txt(path: Path) -> Optional[Dict]:
    """Fallback: parse metrics.txt key-value style."""
    if not path.exists():
        return None
    text = path.read_text()
    out: Dict[str, Optional[float]] = {"best_epoch": None, "threshold": None}
    for k in METRIC_KEYS:
        out[k] = None

    for line in text.splitlines():
        m = re.match(r"\s*([a-z_]+):\s*([0-9.+-eE]+)", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key == "best_epoch":
            out["best_epoch"] = int(float(val))
        elif key == "threshold":
            out["threshold"] = float(val)
        elif key in METRIC_KEYS and out.get(key) is None:
            # take only the first occurrence (validation), then test overrides
            try:
                out[key] = float(val)
            except ValueError:
                pass

    # The metrics.txt stores VALIDATION first, then TEST under "── TEST ──".
    # Re-parse the TEST block specifically:
    if "── TEST ──" in text:
        test_block = text.split("── TEST ──", 1)[1]
        for line in test_block.splitlines():
            m = re.match(r"\s*([a-z_]+):\s*([0-9.+-eE]+)", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key in METRIC_KEYS:
                try:
                    out[key] = float(val)
                except ValueError:
                    pass
    return out


def load_experiment(exp_dir: Path) -> Optional[Dict]:
    info = _from_json(exp_dir / "metrics.json") or _from_txt(exp_dir / "metrics.txt")
    if info is None:
        return None
    info["experiment"] = exp_dir.name
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="experiments")
    parser.add_argument("--out_csv", default="experiments/comparison_summary.csv")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    rows: List[Dict] = []
    for d in sorted(exp_root.glob("exp*")):
        if not d.is_dir():
            continue
        info = load_experiment(d)
        if info is None:
            print(f"  [skip] {d.name}: no metrics file")
            continue
        rows.append(info)

    if not rows:
        print("No experiments with metrics found.")
        return

    # ── pretty table ─────────────────────────────────────────────────────────
    cols = ["experiment", "best_epoch", "threshold"] + METRIC_KEYS
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in cols}

    def fmt_row(r):
        return "  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in cols)

    header_row = {c: c for c in cols}
    print(fmt_row(header_row))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print(fmt_row(r))

    # ── CSV ──────────────────────────────────────────────────────────────────
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _fmt(r.get(c)) for c in cols})
    print(f"\nSaved CSV → {out_csv}")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
