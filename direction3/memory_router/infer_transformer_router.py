"""Inference utility for Transformer-based router checkpoint."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from memory_router.enhanced_features import build_transformer_router_input
from memory_router.transformer_model import TransformerRouter


def infer_transformer_router(
    ckpt_path: str,
    state: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    prompt_sims: List[float],
    reflection_sims: List[float],
    negative_penalties: List[float],
) -> Dict[str, Any]:
    """Run inference with Transformer router.

    Args:
        ckpt_path: Path to router checkpoint
        state: Current problem state dictionary
        candidates: List of candidate memory trajectories
        prompt_sims: Prompt similarity scores
        reflection_sims: Reflection similarity scores
        negative_penalties: Negative penalty scores

    Returns:
        Dictionary containing:
        - router_mix: [w_prompt, w_reflection, w_negative]
        - router_conf: confidence score
        - features: extracted features (for debugging)
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_cfg = ckpt.get("model_config", {})
    candidate_feature_dim = int(model_cfg.get("candidate_feature_dim", 9))
    context_feature_dim = int(model_cfg.get("context_feature_dim", 22))
    d_model = int(model_cfg.get("d_model", 256))
    nhead = int(model_cfg.get("nhead", 4))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))
    max_candidates = int(model_cfg.get("max_candidates", 20))

    # Create model
    model = TransformerRouter(
        candidate_feature_dim=candidate_feature_dim,
        context_feature_dim=context_feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Load weights
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    # Build input features
    cand_feats, ctx_feats, cand_mask = build_transformer_router_input(
        state=state,
        candidates=candidates,
        prompt_sims=prompt_sims,
        reflection_sims=reflection_sims,
        negative_penalties=negative_penalties,
        max_candidates=max_candidates,
    )

    # Convert to tensors
    cand_feats_t = torch.from_numpy(cand_feats).unsqueeze(0)  # [1, N, D]
    ctx_feats_t = torch.from_numpy(ctx_feats).unsqueeze(0)  # [1, D]
    cand_mask_t = torch.from_numpy(cand_mask).unsqueeze(0)  # [1, N]

    # Run inference
    with torch.no_grad():
        mix_weights, confidence = model.predict(cand_feats_t, ctx_feats_t, cand_mask_t)

    # Extract results
    mix = mix_weights[0].cpu().numpy().tolist()
    conf = float(confidence[0].cpu().item())

    return {
        "router_mix": mix,
        "router_conf": conf,
        "candidate_count": len(candidates),
        "features": {
            "context_features": ctx_feats.tolist(),
            "candidate_features_shape": cand_feats.shape,
        },
    }


def _read_state(state_json: Optional[str], state_json_str: Optional[str]) -> Dict[str, Any]:
    """Read state from file or string."""
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer Transformer router mix/conf from state.")
    parser.add_argument("--ckpt", required=True, help="Path to router checkpoint")
    parser.add_argument("--state_json", default=None, help="Path to JSON object for state")
    parser.add_argument("--state_json_str", default=None, help="Inline JSON string for state")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    state = _read_state(args.state_json, args.state_json_str)

    # Extract candidates and similarity scores from state
    # (In practice, these would be passed separately or extracted from state)
    candidates = state.get("candidates", [])
    prompt_sims = state.get("prompt_sims", [])
    reflection_sims = state.get("reflection_sims", [])
    negative_sims = state.get("negative_sims", [])

    if not candidates:
        # Create mock candidates if not provided
        n = max(len(prompt_sims), len(reflection_sims), len(negative_sims), 1)
        candidates = [{"is_solved": False} for _ in range(n)]

    result = infer_transformer_router(
        ckpt_path=args.ckpt,
        state=state,
        candidates=candidates,
        prompt_sims=prompt_sims,
        reflection_sims=reflection_sims,
        negative_penalties=negative_sims,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
