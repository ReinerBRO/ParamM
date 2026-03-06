"""Torch MLP router model (3-way memory mix + confidence)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_mix_conf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        mix = torch.softmax(logits[..., :3], dim=-1)
        conf = torch.sigmoid(logits[..., 3])
        return mix, conf


def loss_fn(
    logits: torch.Tensor,
    target_mix: torch.Tensor,
    target_solved: torch.Tensor,
    mix_weight: float = 1.0,
    conf_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_mix = torch.softmax(logits[..., :3], dim=-1)
    mix_loss = F.mse_loss(pred_mix, target_mix)

    conf_logits = logits[..., 3]
    conf_loss = F.binary_cross_entropy_with_logits(conf_logits, target_solved.float())

    total = mix_weight * mix_loss + conf_weight * conf_loss

    with torch.no_grad():
        pred_conf = torch.sigmoid(conf_logits)
        conf_acc = ((pred_conf >= 0.5) == (target_solved >= 0.5)).float().mean().item()
        mix_mae = torch.abs(pred_mix - target_mix).mean().item()

    metrics = {
        "loss": float(total.item()),
        "mix_loss": float(mix_loss.item()),
        "conf_loss": float(conf_loss.item()),
        "mix_mae": float(mix_mae),
        "conf_acc": float(conf_acc),
    }
    return total, metrics
