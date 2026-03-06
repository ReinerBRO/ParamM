"""Train direction3 memory router MLP and save checkpoint + metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from memory_router.feature_schema import FEATURE_ORDER
from memory_router.model import RouterMLP, loss_fn


class RouterDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        feature_order: Sequence[str],
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
    ):
        self.feature_order = list(feature_order)
        self.norm_mean = list(norm_mean)
        self.norm_std = list(norm_std)
        self.rows: List[Tuple[List[float], List[float], float]] = []

        for record in records:
            features = record.get("features")
            if not isinstance(features, list):
                continue
            vec = [float(features[i]) if i < len(features) else 0.0 for i in range(len(self.feature_order))]
            normalized = [
                (vec[i] - self.norm_mean[i]) / self.norm_std[i]
                for i in range(len(self.feature_order))
            ]
            target_mix = record.get("target_mix") or [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            if not isinstance(target_mix, list) or len(target_mix) < 3:
                target_mix = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            # Enforce non-negative normalized 3-way target mix.
            clipped = [max(0.0, float(target_mix[i] if i < len(target_mix) else 0.0)) for i in range(3)]
            denom = sum(clipped)
            if denom <= 1e-12:
                clipped = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            else:
                clipped = [v / denom for v in clipped]

            solved = float(record.get("outcome_solved", 0.0))
            self.rows.append((normalized, clipped, solved))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, m, s = self.rows[index]
        return (
            torch.tensor(f, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
        )


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
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


def _resolve_feature_order(records: Sequence[Mapping[str, Any]]) -> List[str]:
    for r in records:
        order = r.get("feature_order")
        if isinstance(order, list) and order:
            return [str(x) for x in order]
    return list(FEATURE_ORDER)


def _compute_norm(records: Sequence[Mapping[str, Any]], dim: int) -> Tuple[List[float], List[float]]:
    if not records:
        return [0.0] * dim, [1.0] * dim

    sums = [0.0] * dim
    sums_sq = [0.0] * dim
    n = 0
    for r in records:
        feats = r.get("features")
        if not isinstance(feats, list):
            continue
        n += 1
        for i in range(dim):
            v = float(feats[i]) if i < len(feats) else 0.0
            sums[i] += v
            sums_sq[i] += v * v

    if n == 0:
        return [0.0] * dim, [1.0] * dim

    mean = [s / n for s in sums]
    std = []
    for i in range(dim):
        var = max(sums_sq[i] / n - mean[i] * mean[i], 1e-8)
        std.append(var ** 0.5)
    return mean, std


def _aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"loss": 0.0, "mix_loss": 0.0, "conf_loss": 0.0, "mix_mae": 0.0, "conf_acc": 0.0}
    keys = list(metrics[0].keys())
    return {k: float(sum(m.get(k, 0.0) for m in metrics) / len(metrics)) for k in keys}


def _run_epoch(model: RouterMLP, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, train: bool) -> Dict[str, float]:
    model.train(train)
    step_metrics: List[Dict[str, float]] = []

    for x, target_mix, target_solved in loader:
        x = x.to(device)
        target_mix = target_mix.to(device)
        target_solved = target_solved.to(device)

        logits = model(x)
        loss, metrics = loss_fn(logits, target_mix, target_solved)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        step_metrics.append(metrics)

    return _aggregate(step_metrics)


def _safe_metric_value(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def train(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_records = _read_jsonl(args.train_jsonl)
    val_records = _read_jsonl(args.val_jsonl)

    if not train_records:
        raise ValueError(f"No training records found in {args.train_jsonl}")
    if not val_records:
        val_records = train_records

    feature_order = _resolve_feature_order(train_records)
    dim = len(feature_order)
    mean, std = _compute_norm(train_records, dim)

    train_ds = RouterDataset(train_records, feature_order, mean, std)
    val_ds = RouterDataset(val_records, feature_order, mean, std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterMLP(input_dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, train=True)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        if _safe_metric_value(val_metrics, "loss", float("inf")) < best_val:
            best_val = _safe_metric_value(val_metrics, "loss", float("inf"))
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"epoch={epoch} train_loss={_safe_metric_value(train_metrics, 'loss'):.6f} "
            f"val_loss={_safe_metric_value(val_metrics, 'loss'):.6f} "
            f"val_conf_acc={_safe_metric_value(val_metrics, 'conf_acc'):.4f}"
        )

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    final_ckpt_path = args.output_ckpt
    if os.path.isdir(final_ckpt_path) or not final_ckpt_path.endswith(".ckpt"):
        os.makedirs(final_ckpt_path, exist_ok=True)
        final_ckpt_path = os.path.join(final_ckpt_path, "router.ckpt")

    ckpt = {
        "state_dict": best_state,
        "feature_order": feature_order,
        "norm_mean": mean,
        "norm_std": std,
        "model_config": {
            "input_dim": dim,
            "hidden_dims": [128, 64],
            "dropout": 0.1,
        },
        "train_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
        },
        "best_val_loss": best_val,
    }

    torch.save(ckpt, final_ckpt_path)

    metrics_out = {
        "best_val_loss": best_val,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "history": history,
    }
    metrics_path = os.path.join(os.path.dirname(final_ckpt_path) or ".", "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    return {
        "output_ckpt": final_ckpt_path,
        "train_metrics": metrics_path,
        "best_val_loss": best_val,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train direction3 memory router.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
