"""Losses for multi-label MOSEI emotion recognition.

Primary loss:  BCEWithLogitsLoss with optional pos_weight.
Auxiliary losses (only used by BottleneckFusion when --use_domain_sep):
    - separation:      invariant ⊥ private  (cosine^2 -> 0) per modality
    - invariant:       MSE alignment between modalities' invariant pools
    - reconstruction:  MSE between reconstructed features and original pooled
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pos_weight(
    pos_counts: np.ndarray,
    neg_counts: np.ndarray,
    mode: str = "sqrt",
    eps: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute pos_weight for BCEWithLogitsLoss according to `mode`.

    mode:
        none     -> None (no re-weighting)
        balanced -> neg / pos
        sqrt     -> sqrt(neg / pos), then normalized so mean == 1.0
    """
    if mode == "none":
        return None

    pos = pos_counts.astype(np.float64) + eps
    neg = neg_counts.astype(np.float64) + eps
    ratio = neg / pos

    if mode == "balanced":
        pw = ratio
    elif mode == "sqrt":
        pw = np.sqrt(ratio)
        pw = pw / pw.mean()  # keep loss scale stable
    else:
        raise ValueError(f"Unknown pos_weight_mode '{mode}'. "
                         "Use one of: none, balanced, sqrt.")

    return torch.tensor(pw, dtype=torch.float32)


def make_bce_criterion(pos_weight: Optional[torch.Tensor]) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


# ────────────────────────────── Auxiliary losses ──────────────────────────────

def compute_separation_loss(domain_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Push (invariant, private) toward orthogonality per modality."""
    losses = []
    for mod in ("text", "audio", "vision"):
        inv = domain_data.get(f"{mod}_inv_pool")
        priv = domain_data.get(f"{mod}_priv_pool")
        if inv is None or priv is None:
            continue
        cos = F.cosine_similarity(inv, priv, dim=-1)
        losses.append((cos ** 2).mean())
    if not losses:
        return torch.zeros((), device=_first_device(domain_data))
    return torch.stack(losses).mean()


def compute_invariant_loss(domain_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pairwise MSE between invariant pools (cross-modal alignment)."""
    pools = [
        domain_data[k]
        for k in ("text_inv_pool", "audio_inv_pool", "vision_inv_pool")
        if k in domain_data
    ]
    if len(pools) < 2:
        return torch.zeros((), device=_first_device(domain_data))
    total = pools[0].new_zeros(())
    count = 0
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            total = total + F.mse_loss(pools[i], pools[j])
            count += 1
    return total / count


def compute_reconstruction_loss(recon_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """MSE between reconstructed features and original pooled features."""
    losses = []
    for mod in ("text", "audio", "vision"):
        recon = recon_data.get(f"{mod}_recon")
        orig = recon_data.get(f"{mod}_original")
        if recon is None or orig is None:
            continue
        losses.append(F.mse_loss(recon, orig))
    if not losses:
        return torch.zeros((), device=_first_device(recon_data))
    return torch.stack(losses).mean()


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domain_data: Dict[str, torch.Tensor],
    recon_data: Dict[str, torch.Tensor],
    criterion: nn.Module,
    *,
    use_domain_sep: bool = False,
    alpha_sep: float = 0.03,
    alpha_inv: float = 0.005,
    alpha_rec: float = 0.001,
):
    """Return (total_loss, loss_dict_for_logging)."""
    task = criterion(logits, labels)
    log = {"task": float(task.detach())}

    if not use_domain_sep or not domain_data:
        return task, log

    L_sep = compute_separation_loss(domain_data)
    L_inv = compute_invariant_loss(domain_data)
    L_rec = compute_reconstruction_loss(recon_data)

    total = task + alpha_sep * L_sep + alpha_inv * L_inv + alpha_rec * L_rec
    log.update({
        "sep": float(L_sep.detach()),
        "inv": float(L_inv.detach()),
        "rec": float(L_rec.detach()),
    })
    return total, log


def _first_device(d: Dict[str, torch.Tensor]) -> torch.device:
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
