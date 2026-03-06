"""Train Transformer-based router with enhanced features.

This script trains the new router architecture using:
1. Enhanced per-candidate and context features
2. Transformer encoder for modeling candidate interactions
3. Confidence calibration via temperature scaling
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from memory_router.enhanced_features import (
    build_transformer_router_input,
    CANDIDATE_FEATURE_DIM,
    TOTAL_CONTEXT_DIM,
)
from memory_router.transformer_model import (
    TransformerRouter,
    CalibratedTransformerRouter,
    router_loss_fn,
)


class TransformerRouterDataset(Dataset):
    """Dataset for training Transformer router."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        max_candidates: int = 20,
    ):
        self.max_candidates = max_candidates
        self.samples: List[Dict[str, Any]] = []

        for record in records:
            # Extract state and candidates from record
            state = record.get("state", {})
            if not isinstance(state, Mapping):
                continue

            # Get similarity scores
            prompt_sims = record.get("prompt_sims", [])
            reflection_sims = record.get("reflection_sims", [])
            negative_sims = record.get("negative_sims", [])

            # Get candidates (mock if not available)
            candidates = record.get("candidates", [])
            if not candidates:
                # Create mock candidates from similarity scores
                n = max(len(prompt_sims), len(reflection_sims), len(negative_sims))
                candidates = [{"is_solved": False} for _ in range(n)]

            if not candidates:
                continue

            # Build input features
            try:
                cand_feats, ctx_feats, cand_mask = build_transformer_router_input(
                    state=state,
                    candidates=candidates,
                    prompt_sims=prompt_sims,
                    reflection_sims=reflection_sims,
                    negative_penalties=negative_sims,
                    max_candidates=max_candidates,
                )
            except Exception:
                continue

            # Get target
            target_solved = float(record.get("outcome_solved", 0))
            target_mix = record.get("target_mix", [1.0 / 3, 1.0 / 3, 1.0 / 3])
            if not isinstance(target_mix, list) or len(target_mix) < 3:
                target_mix = [1.0 / 3, 1.0 / 3, 1.0 / 3]

            self.samples.append({
                "candidate_features": cand_feats,
                "context_features": ctx_feats,
                "candidate_mask": cand_mask,
                "target_solved": target_solved,
                "target_mix": np.array(target_mix[:3], dtype=np.float32),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "candidate_features": torch.from_numpy(sample["candidate_features"]),
            "context_features": torch.from_numpy(sample["context_features"]),
            "candidate_mask": torch.from_numpy(sample["candidate_mask"]),
            "target_solved": torch.tensor(sample["target_solved"], dtype=torch.float32),
            "target_mix": torch.from_numpy(sample["target_mix"]),
        }


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _run_epoch(
    model: TransformerRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    """Run one epoch of training or validation."""
    model.train(train)
    step_metrics: List[Dict[str, float]] = []

    for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
        cand_feats = batch["candidate_features"].to(device)
        ctx_feats = batch["context_features"].to(device)
        cand_mask = batch["candidate_mask"].to(device)
        target_solved = batch["target_solved"].to(device)
        target_mix = batch["target_mix"].to(device)

        # Forward pass
        mix_weights, confidence = model(cand_feats, ctx_feats, cand_mask)

        # Compute loss
        loss, metrics = router_loss_fn(
            mix_weights=mix_weights,
            confidence=confidence,
            target_solved=target_solved,
            target_mix=target_mix,
            mix_weight=1.0,
            conf_weight=1.0,
            entropy_weight=0.05,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        step_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {}
    for key in step_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in step_metrics) / len(step_metrics)

    return avg_metrics


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Main training function."""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    print("Loading training data...")
    train_records = _read_jsonl(args.train_jsonl)
    val_records = _read_jsonl(args.val_jsonl)

    if not train_records:
        raise ValueError(f"No training records found in {args.train_jsonl}")
    if not val_records:
        print("Warning: No validation records, using training data for validation")
        val_records = train_records

    print(f"Loaded {len(train_records)} train, {len(val_records)} val records")

    # Create datasets
    train_ds = TransformerRouterDataset(train_records, max_candidates=args.max_candidates)
    val_ds = TransformerRouterDataset(val_records, max_candidates=args.max_candidates)

    print(f"Created {len(train_ds)} train, {len(val_ds)} val samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerRouter(
        candidate_feature_dim=CANDIDATE_FEATURE_DIM,
        context_feature_dim=TOTAL_CONTEXT_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1,
    )

    # Training loop
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = _run_epoch(model, train_loader, optimizer, device, train=True)

        # Validate
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, optimizer, device, train=False)

        # Update scheduler
        scheduler.step()

        # Log
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"  Train: loss={train_metrics['loss']:.4f}, "
            f"conf_acc={train_metrics['conf_acc']:.4f}, "
            f"ece={train_metrics['ece']:.4f}"
        )
        print(
            f"  Val:   loss={val_metrics['loss']:.4f}, "
            f"conf_acc={val_metrics['conf_acc']:.4f}, "
            f"ece={val_metrics['ece']:.4f}"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"  → New best model (val_loss={best_val_loss:.4f})")

    # Save checkpoint
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "state_dict": best_state,
        "model_config": {
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "context_feature_dim": TOTAL_CONTEXT_DIM,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "max_candidates": args.max_candidates,
        },
        "train_args": vars(args),
        "best_val_loss": best_val_loss,
    }

    torch.save(ckpt, args.output_ckpt)
    print(f"\nSaved checkpoint to {args.output_ckpt}")

    # Save metrics
    metrics_path = os.path.join(os.path.dirname(args.output_ckpt) or ".", "train_metrics.json")
    metrics_out = {
        "best_val_loss": best_val_loss,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {metrics_path}")

    return {
        "output_ckpt": args.output_ckpt,
        "train_metrics": metrics_path,
        "best_val_loss": best_val_loss,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Transformer router")
    parser.add_argument("--train_jsonl", required=True, help="Training data (JSONL)")
    parser.add_argument("--val_jsonl", required=True, help="Validation data (JSONL)")
    parser.add_argument("--output_ckpt", required=True, help="Output checkpoint path")

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_candidates", type=int, default=20, help="Max candidates per sample")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
