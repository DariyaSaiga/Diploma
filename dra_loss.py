"""
dra_loss.py — Dynamic Rate Adjustment Loss (из XMBT).

Идея:
  Каждая ветка (text, audio, visual, fused) получает свой loss.
  Итоговый лосс = взвешенная сумма с двумя механизмами взвешивания:
    1) Uncertainty weighting (Kendall et al.) — learnable α_m
       Веса выученные, большой α_m → модальность "неуверенна" → меньший вклад
    2) DWA (Dynamic Weight Averaging) — λ_m(τ)
       Веса динамические: если loss по ветке улучшается медленно — вес растёт
  + φ-регуляризация: ограничение на сумму α_m

Формула:
    L = Σ_m [ (1/α²_m + λ_m) * L_m + log(1 + α²_m) ] + |φ - Σ_m α_m|

Параметры:
    num_tasks   : 4  (text, audio, visual, fused)
    temperature : 2.0  (для softmax в DWA)
    phi         : 4.0  (≈ num_tasks, ограничение суммы α)

Использование:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    dra = DRALoss(num_tasks=4).to(device)
    optimizer = AdamW([...model params..., *dra.parameters()], lr=...)

    # в train loop:
    logits_f, logits_t, logits_a, logits_v = model.forward_dra(...)
    losses = [
        criterion(logits_t, labels),   # 0: text
        criterion(logits_a, labels),   # 1: audio
        criterion(logits_v, labels),   # 2: visual
        criterion(logits_f, labels),   # 3: fused (главный)
    ]
    total_loss = dra(losses)
    total_loss.backward()

    # после каждой эпохи:
    dra.update_history([L_text.item(), L_audio.item(), L_visual.item(), L_fused.item()])
"""

import torch
import torch.nn as nn


class DRALoss(nn.Module):
    """
    Dynamic Rate Adjustment Loss.

    Args:
        num_tasks   : число лоссов (обычно 4: text, audio, visual, fused)
        temperature : T для DWA softmax (2.0 из статьи)
        phi         : целевая сумма α_m (обычно = num_tasks)
    """

    def __init__(self, num_tasks: int = 4, temperature: float = 2.0, phi: float = None):
        super().__init__()
        self.num_tasks   = num_tasks
        self.temperature = temperature
        self.phi         = float(phi or num_tasks)

        # Learnable log(α_m) — инициализируем нулями (α_m = 1 изначально)
        self.log_alpha = nn.Parameter(torch.zeros(num_tasks))

        # История лоссов для DWA (буфер — не обучается, но сохраняется в checkpoint)
        self.register_buffer("prev_losses",      torch.ones(num_tasks))
        self.register_buffer("prev_prev_losses", torch.ones(num_tasks))

    # ─────────────────────────────────────────────────────────────────────────

    def _dwa_weights(self) -> torch.Tensor:
        """
        DWA: λ_m = M * softmax(r_m / T)
        где r_m = L_m(τ-1) / L_m(τ-2) — относительная скорость изменения лосса.
        Чем медленнее улучшается — тем выше вес.
        """
        r = self.prev_losses / (self.prev_prev_losses + 1e-8)        # (M,)
        weights = self.num_tasks * torch.softmax(r / self.temperature, dim=0)
        return weights.detach()   # DWA не дифференцируется через историю

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: list of M скалярных тензоров [L_text, L_audio, L_visual, L_fused]

        Returns:
            total_loss: скалярный тензор (дифференцируется по model params + log_alpha)
        """
        assert len(losses) == self.num_tasks, \
            f"DRALoss ожидает {self.num_tasks} лоссов, получено {len(losses)}"

        stacked = torch.stack(losses)             # (M,)
        alpha   = torch.exp(self.log_alpha)       # α_m > 0, (M,)

        # ── DWA веса (из истории, не обучаются) ──────────────────────────
        dwa_w = self._dwa_weights().to(stacked.device)   # (M,)

        # ── Uncertainty weighting + DWA ───────────────────────────────────
        # (1/α²_m + λ_m) * L_m
        inv_alpha_sq = 1.0 / (alpha ** 2 + 1e-8)         # (M,)
        weighted     = ((inv_alpha_sq + dwa_w) * stacked).sum()

        # ── Uncertainty regularization ───────────────────────────────────
        # log(1 + α²_m)
        reg = torch.log(1.0 + alpha ** 2).sum()

        # ── φ-регуляризация ───────────────────────────────────────────────
        # |φ - Σ α_m|  — ограничение на суммарный масштаб
        phi_reg = torch.abs(
            torch.tensor(self.phi, device=alpha.device, dtype=alpha.dtype) - alpha.sum()
        )

        return weighted + reg + phi_reg

    # ─────────────────────────────────────────────────────────────────────────

    def update_history(self, epoch_losses: list) -> None:
        """
        Обновить историю лоссов для следующей DWA-итерации.
        Вызывать ПОСЛЕ каждой эпохи.

        Args:
            epoch_losses: list of M float — средние лоссы за эпоху
                          порядок: [L_text, L_audio, L_visual, L_fused]
        """
        self.prev_prev_losses = self.prev_losses.clone()
        self.prev_losses = torch.tensor(
            epoch_losses, dtype=torch.float32, device=self.prev_losses.device
        )

    # ─────────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        alpha = torch.exp(self.log_alpha).detach().cpu().numpy().round(3)
        return (f"num_tasks={self.num_tasks}, T={self.temperature}, "
                f"phi={self.phi}, alpha={alpha}")
