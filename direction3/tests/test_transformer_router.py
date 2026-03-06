#!/usr/bin/env python3
"""Quick test of Transformer router implementation."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from memory_router.enhanced_features import (
    extract_candidate_features,
    extract_candidate_set_features,
    extract_context_features,
    build_transformer_router_input,
)
from memory_router.transformer_model import TransformerRouter, router_loss_fn


def test_feature_extraction():
    """Test feature extraction functions."""
    print("Testing feature extraction...")

    # Mock state
    state = {
        "prompt": "Write a function to add two numbers",
        "reflections": ["Need to handle edge cases"],
        "test_feedback": ["AssertionError: expected 5, got 4"],
        "attempt_count": 2,
    }

    # Mock candidates
    candidates = [
        {"is_solved": True, "gen_solution": "def add(a, b): return a + b"},
        {"is_solved": False, "gen_solution": "def add(a, b): return a - b"},
    ]

    prompt_sims = [0.8, 0.6]
    reflection_sims = [0.5, 0.4]
    negative_penalties = [0.1, 0.3]

    # Test per-candidate features
    cand_feats = extract_candidate_features(candidates[0], prompt_sims[0], reflection_sims[0], negative_penalties[0])
    print(f"  Candidate features: {len(cand_feats)} dims")
    assert len(cand_feats) == 9, f"Expected 9 features, got {len(cand_feats)}"

    # Test candidate set features
    set_feats = extract_candidate_set_features(prompt_sims, reflection_sims, negative_penalties, candidates)
    print(f"  Candidate set features: {len(set_feats)} dims")
    assert len(set_feats) == 10, f"Expected 10 features, got {len(set_feats)}"

    # Test context features
    ctx_feats = extract_context_features(state)
    print(f"  Context features: {len(ctx_feats)} dims")
    assert len(ctx_feats) == 11, f"Expected 11 features, got {len(ctx_feats)}"

    # Test full input building
    cand_feats_arr, ctx_feats_arr, cand_mask = build_transformer_router_input(
        state, candidates, prompt_sims, reflection_sims, negative_penalties, max_candidates=20
    )
    print(f"  Candidate features array: {cand_feats_arr.shape}")
    print(f"  Context features array: {ctx_feats_arr.shape}")
    print(f"  Candidate mask: {cand_mask.shape}, valid={(~cand_mask).sum()}")

    assert cand_feats_arr.shape == (20, 9), f"Expected (20, 9), got {cand_feats_arr.shape}"
    assert ctx_feats_arr.shape == (21,), f"Expected (21,), got {ctx_feats_arr.shape}"
    assert cand_mask.shape == (20,), f"Expected (20,), got {cand_mask.shape}"
    assert (~cand_mask).sum() == 2, f"Expected 2 valid candidates, got {(~cand_mask).sum()}"

    print("  ✓ Feature extraction tests passed")


def test_model_forward():
    """Test Transformer router forward pass."""
    print("\\nTesting model forward pass...")

    # Create model
    model = TransformerRouter(
        candidate_feature_dim=9,
        context_feature_dim=21,
        d_model=64,  # Smaller for testing
        nhead=2,
        num_layers=1,
        dropout=0.1,
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create mock input
    batch_size = 4
    max_candidates = 10

    cand_feats = torch.randn(batch_size, max_candidates, 9)
    ctx_feats = torch.randn(batch_size, 21)
    cand_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    cand_mask[:, 5:] = True  # Mask last 5 candidates

    # Forward pass
    mix_weights, confidence = model(cand_feats, ctx_feats, cand_mask)

    print(f"  Mix weights shape: {mix_weights.shape}")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  Mix weights example: {mix_weights[0].tolist()}")
    print(f"  Confidence example: {confidence[0].item():.4f}")

    assert mix_weights.shape == (batch_size, 3), f"Expected (4, 3), got {mix_weights.shape}"
    assert confidence.shape == (batch_size,), f"Expected (4,), got {confidence.shape}"
    assert torch.allclose(mix_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "Mix weights should sum to 1"
    assert (confidence >= 0).all() and (confidence <= 1).all(), "Confidence should be in [0, 1]"

    print("  ✓ Model forward pass tests passed")


def test_loss_function():
    """Test loss function."""
    print("\\nTesting loss function...")

    batch_size = 8
    mix_weights = torch.softmax(torch.randn(batch_size, 3), dim=-1)
    confidence = torch.sigmoid(torch.randn(batch_size))
    target_solved = torch.randint(0, 2, (batch_size,)).float()
    target_mix = torch.softmax(torch.randn(batch_size, 3), dim=-1)

    loss, metrics = router_loss_fn(
        mix_weights=mix_weights,
        confidence=confidence,
        target_solved=target_solved,
        target_mix=target_mix,
        mix_weight=1.0,
        conf_weight=1.0,
        entropy_weight=0.1,
    )

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert "conf_acc" in metrics, "Metrics should contain conf_acc"
    assert "mix_entropy" in metrics, "Metrics should contain mix_entropy"
    assert "ece" in metrics, "Metrics should contain ece"

    print("  ✓ Loss function tests passed")


def test_inference():
    """Test inference mode."""
    print("\\nTesting inference mode...")

    model = TransformerRouter(
        candidate_feature_dim=9,
        context_feature_dim=21,
        d_model=64,
        nhead=2,
        num_layers=1,
        dropout=0.1,
    )

    cand_feats = torch.randn(1, 10, 9)
    ctx_feats = torch.randn(1, 21)
    cand_mask = torch.zeros(1, 10, dtype=torch.bool)
    cand_mask[:, 5:] = True

    # Test predict method
    mix_weights, confidence = model.predict(cand_feats, ctx_feats, cand_mask)

    print(f"  Predicted mix: {mix_weights[0].tolist()}")
    print(f"  Predicted confidence: {confidence[0].item():.4f}")

    assert mix_weights.shape == (1, 3), f"Expected (1, 3), got {mix_weights.shape}"
    assert confidence.shape == (1,), f"Expected (1,), got {confidence.shape}"

    print("  ✓ Inference tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Transformer Router Implementation Tests")
    print("=" * 60)

    try:
        test_feature_extraction()
        test_model_forward()
        test_loss_function()
        test_inference()

        print("\\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
