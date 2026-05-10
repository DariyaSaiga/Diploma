"""Small training utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Pick device. By default `auto` skips MPS because nn.LSTM is unstable on it.

    prefer ∈ {auto, cpu, cuda, mps}.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # auto: prefer CUDA, then CPU. MPS is opt-in due to LSTM instability.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(exp_dir: str | Path, config: Dict[str, Any]) -> None:
    exp_dir = Path(exp_dir)
    ensure_dir(exp_dir)

    serializable = {k: _coerce(v) for k, v in config.items()}
    with open(exp_dir / "config.json", "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
    with open(exp_dir / "config.txt", "w") as f:
        for k in sorted(serializable):
            f.write(f"{k}: {serializable[k]}\n")


def _coerce(v):
    if isinstance(v, (Path,)):
        return str(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, (torch.Tensor,)):
        return v.detach().cpu().tolist()
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


class AverageMeter:
    """Running average over a stream of scalar values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def append_log(path: str | Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")
