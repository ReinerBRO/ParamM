"""Transformer-based router model for adaptive memory mixing.

This model uses self-attention to capture interactions between candidate
memories and outputs calibrated mix weights + confidence scores.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRouter(nn.Module):
    """Transformer-based router for memory mixing decisions.

    Architecture:
    1. Encode each candidate memory with its features
    2. Encode current context (problem state + candidate set stats)
    3. Use transformer to model candidate interactions
    4. Predict mix weights [w_prompt, w_reflection, w_negative]
    5. Predict calibrated confidence score
    """

    def __init__(
        self,
        candidate_feature_dim: int = 9,
        context_feature_dim: int = 22,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.candidate_feature_dim = candidate_feature_dim
        self.context_feature_dim = context_feature_dim
        self.d_model = d_model

        # Candidate encoder: maps per-candidate features to d_model
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Context encoder: maps context features to d_model
        self.context_encoder = nn.Sequential(
            nn.Linear(context_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder: models interactions between context and candidates
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Mix weight predictor: outputs 3 logits for [prompt, reflection, negative]
        self.mix_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # Confidence predictor: outputs calibrated confidence score
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        candidate_features: torch.Tensor,
        context_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            candidate_features: [batch, max_candidates, candidate_feature_dim]
            context_features: [batch, context_feature_dim]
            candidate_mask: [batch, max_candidates], True = padding

        Returns:
            mix_weights: [batch, 3] - softmax-normalized weights
            confidence: [batch] - calibrated confidence scores
        """
        batch_size = candidate_features.size(0)

        # Encode candidates
        cand_encoded = self.candidate_encoder(candidate_features)  # [B, N, d_model]

        # Encode context
        ctx_encoded = self.context_encoder(context_features).unsqueeze(1)  # [B, 1, d_model]

        # Concatenate: [context_token, candidate_1, ..., candidate_N]
        seq = torch.cat([ctx_encoded, cand_encoded], dim=1)  # [B, 1+N, d_model]

        # Prepare mask for transformer (True = ignore)
        if candidate_mask is not None:
            # Prepend False for context token (always valid)
            ctx_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=candidate_mask.device)
            full_mask = torch.cat([ctx_mask, candidate_mask], dim=1)  # [B, 1+N]
        else:
            full_mask = None

        # Apply transformer
        transformed = self.transformer(seq, src_key_padding_mask=full_mask)  # [B, 1+N, d_model]

        # Extract context representation (first token)
        context_repr = transformed[:, 0, :]  # [B, d_model]

        # Predict mix weights
        mix_logits = self.mix_head(context_repr)  # [B, 3]
        mix_weights = F.softmax(mix_logits, dim=-1)  # [B, 3]

        # Predict confidence with temperature scaling
        conf_logit = self.conf_head(context_repr).squeeze(-1)  # [B]
        confidence = torch.sigmoid(conf_logit / self.temperature.clamp(min=0.1))  # [B]

        return mix_weights, confidence

    @torch.no_grad()
    def predict(
        self,
        candidate_features: torch.Tensor,
        context_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference mode prediction."""
        self.eval()
        return self.forward(candidate_features, context_features, candidate_mask)


def router_loss_fn(
    mix_weights: torch.Tensor,
    confidence: torch.Tensor,
    target_solved: torch.Tensor,
    target_mix: Optional[torch.Tensor] = None,
    mix_weight: float = 1.0,
    conf_weight: float = 1.0,
    entropy_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss for router training.

    Args:
        mix_weights: [batch, 3] predicted mix weights
        confidence: [batch] predicted confidence scores
        target_solved: [batch] binary labels (1 = solved, 0 = failed)
        target_mix: [batch, 3] optional target mix weights (if available)
        mix_weight: weight for mix loss
        conf_weight: weight for confidence loss
        entropy_weight: weight for entropy regularization

    Returns:
        total_loss: scalar loss
        metrics: dictionary of metrics for logging
    """
    batch_size = mix_weights.size(0)

    # Confidence loss: binary cross-entropy
    conf_loss = F.binary_cross_entropy(confidence, target_solved.float())

    # Mix loss: if target_mix is provided, use MSE; otherwise use entropy regularization
    if target_mix is not None:
        mix_loss = F.mse_loss(mix_weights, target_mix)
    else:
        # Encourage sparse mix weights (low entropy)
        # Entropy = -sum(p * log(p))
        eps = 1e-8
        entropy = -(mix_weights * torch.log(mix_weights + eps)).sum(dim=-1).mean()
        mix_loss = entropy_weight * entropy

    # Total loss
    total_loss = mix_weight * mix_loss + conf_weight * conf_loss

    # Compute metrics
    with torch.no_grad():
        # Confidence accuracy
        conf_pred = (confidence >= 0.5).float()
        conf_acc = (conf_pred == target_solved).float().mean().item()

        # Mix weight statistics
        mix_entropy = -(mix_weights * torch.log(mix_weights + 1e-8)).sum(dim=-1).mean().item()
        mix_max = mix_weights.max(dim=-1)[0].mean().item()

        # Calibration error (Expected Calibration Error approximation)
        conf_bins = torch.linspace(0, 1, 11, device=confidence.device)
        ece = 0.0
        for i in range(10):
            bin_mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i + 1])
            if bin_mask.sum() > 0:
                bin_conf = confidence[bin_mask].mean()
                bin_acc = target_solved[bin_mask].float().mean()
                ece += bin_mask.float().mean() * torch.abs(bin_conf - bin_acc)
        ece = ece.item()

    metrics = {
        "loss": float(total_loss.item()),
        "mix_loss": float(mix_loss.item()),
        "conf_loss": float(conf_loss.item()),
        "conf_acc": float(conf_acc),
        "mix_entropy": float(mix_entropy),
        "mix_max": float(mix_max),
        "ece": float(ece),
    }

    return total_loss, metrics


class CalibratedTransformerRouter(nn.Module):
    """Wrapper that adds temperature scaling for confidence calibration."""

    def __init__(self, base_router: TransformerRouter):
        super().__init__()
        self.base_router = base_router
        # Override base router's temperature with a separate calibration parameter
        self.calibration_temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        candidate_features: torch.Tensor,
        context_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with calibrated confidence."""
        # Get base predictions
        mix_weights, _ = self.base_router(candidate_features, context_features, candidate_mask)

        # Recompute confidence with calibration temperature
        batch_size = candidate_features.size(0)
        cand_encoded = self.base_router.candidate_encoder(candidate_features)
        ctx_encoded = self.base_router.context_encoder(context_features).unsqueeze(1)
        seq = torch.cat([ctx_encoded, cand_encoded], dim=1)

        if candidate_mask is not None:
            ctx_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=candidate_mask.device)
            full_mask = torch.cat([ctx_mask, candidate_mask], dim=1)
        else:
            full_mask = None

        transformed = self.base_router.transformer(seq, src_key_padding_mask=full_mask)
        context_repr = transformed[:, 0, :]

        conf_logit = self.base_router.conf_head(context_repr).squeeze(-1)
        calibrated_conf = torch.sigmoid(conf_logit / self.calibration_temperature.clamp(min=0.1))

        return mix_weights, calibrated_conf

    def calibrate(self, val_loader: torch.utils.data.DataLoader, device: torch.device):
        """Calibrate temperature on validation set using LBFGS."""
        # Freeze base router
        for param in self.base_router.parameters():
            param.requires_grad = False

        self.calibration_temperature.requires_grad = True

        optimizer = torch.optim.LBFGS(
            [self.calibration_temperature],
            lr=0.01,
            max_iter=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            count = 0

            for batch in val_loader:
                cand_feats = batch["candidate_features"].to(device)
                ctx_feats = batch["context_features"].to(device)
                cand_mask = batch["candidate_mask"].to(device)
                target_solved = batch["target_solved"].to(device)

                _, conf = self.forward(cand_feats, ctx_feats, cand_mask)
                loss = F.binary_cross_entropy(conf, target_solved.float())
                total_loss += loss
                count += 1

            avg_loss = total_loss / max(count, 1)
            avg_loss.backward()
            return avg_loss

        optimizer.step(closure)

        # Restore base router gradients
        for param in self.base_router.parameters():
            param.requires_grad = True

        print(f"Calibration complete. Temperature: {self.calibration_temperature.item():.4f}")
