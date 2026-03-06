"""Inference utility for direction3 router checkpoint."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping, Optional

import torch

from memory_router.feature_schema import vectorize_features
from memory_router.model import RouterMLP


def _read_state(state_json: Optional[str], state_json_str: Optional[str]) -> Dict[str, Any]:
    if state_json_str:
        try:
            data = json.loads(state_json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --state_json_str: {exc}") from exc
        if isinstance(data, Mapping):
            return dict(data)
        raise ValueError("--state_json_str must decode to a JSON object")

    if state_json:
        with open(state_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, Mapping):
            return dict(data)
        raise ValueError("--state_json must contain a JSON object")

    raise ValueError("Provide one of --state_json or --state_json_str")


def infer(ckpt_path: str, state: Mapping[str, Any]) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    feature_order = ckpt.get("feature_order") or []
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError("Checkpoint missing valid feature_order")

    mean = ckpt.get("norm_mean") or [0.0] * len(feature_order)
    std = ckpt.get("norm_std") or [1.0] * len(feature_order)
    model_cfg = ckpt.get("model_config") or {}

    input_dim = int(model_cfg.get("input_dim", len(feature_order)))
    hidden_dims = tuple(model_cfg.get("hidden_dims", [128, 64]))
    dropout = float(model_cfg.get("dropout", 0.1))

    model = RouterMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    features = vectorize_features(state, feature_order)
    if len(features) < input_dim:
        features = features + [0.0] * (input_dim - len(features))
    features = features[:input_dim]

    normed = []
    for i in range(input_dim):
        m = float(mean[i] if i < len(mean) else 0.0)
        s = float(std[i] if i < len(std) else 1.0)
        if abs(s) < 1e-12:
            s = 1.0
        normed.append((features[i] - m) / s)

    x = torch.tensor([normed], dtype=torch.float32)
    mix, conf = model.predict_mix_conf(x)

    return {
        "router_mix": [float(v) for v in mix[0].tolist()],
        "router_conf": float(conf[0].item()),
        "feature_order": feature_order,
        "features": features,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer router mix/conf from one state record.")
    parser.add_argument("--ckpt", required=True, help="Path to router.ckpt")
    parser.add_argument("--state_json", default=None, help="Path to JSON object for state")
    parser.add_argument("--state_json_str", default=None, help="Inline JSON string for state")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    state = _read_state(args.state_json, args.state_json_str)
    result = infer(args.ckpt, state)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
