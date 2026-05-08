import torch
import torch.nn as nn


class DRALoss(nn.Module):

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
        # ✅ Используем clamp для стабильности: избегаем деления на очень маленькие числа
        r = self.prev_losses / (self.prev_prev_losses.clamp(min=1e-4))  # (M,)
        weights = self.num_tasks * torch.softmax(r / self.temperature, dim=0)
        return weights.detach()   # DWA не дифференцируется через историю

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, losses: list) -> torch.Tensor:
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
        self.prev_prev_losses = self.prev_losses.clone()
        self.prev_losses = torch.tensor(
            epoch_losses, dtype=torch.float32, device=self.prev_losses.device
        )

    # ─────────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        alpha = torch.exp(self.log_alpha).detach().cpu().numpy().round(3)
        return (f"num_tasks={self.num_tasks}, T={self.temperature}, "
                f"phi={self.phi}, alpha={alpha}")
