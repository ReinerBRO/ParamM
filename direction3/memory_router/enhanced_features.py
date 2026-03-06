"""Enhanced feature schema for Transformer-based router.

This module extracts richer features that capture:
1. Per-candidate quality metrics
2. Candidate set distribution statistics
3. Context-aware state features
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def extract_candidate_features(
    candidate: Mapping[str, Any],
    prompt_sim: float,
    reflection_sim: float,
    negative_penalty: float,
) -> Dict[str, float]:
    """Extract per-candidate quality features.

    Args:
        candidate: A single candidate memory trajectory
        prompt_sim: Similarity score with current prompt
        reflection_sim: Similarity score with current reflection
        negative_penalty: Negative sample penalty score

    Returns:
        Dictionary of candidate-level features
    """
    features = {
        # Similarity scores
        "prompt_sim_score": float(prompt_sim),
        "reflection_sim_score": float(reflection_sim),
        "negative_penalty": float(negative_penalty),

        # Candidate quality indicators
        "candidate_solved_flag": float(bool(candidate.get("is_solved", False))),
        "candidate_has_reflection": float(bool(candidate.get("reflection_embedding") is not None)),

        # Code quality metrics
        "candidate_code_len": float(len(str(candidate.get("gen_solution", "")))),
        "candidate_prompt_len": float(len(str(candidate.get("prompt", "")))),

        # Trajectory depth
        "candidate_reflection_depth": float(len(_safe_list(candidate.get("reflections", [])))),
        "candidate_attempt_count": float(_to_float(candidate.get("attempt_count", 0))),
    }

    return features


def extract_candidate_set_features(
    prompt_sims: List[float],
    reflection_sims: List[float],
    negative_penalties: List[float],
    candidates: List[Mapping[str, Any]],
) -> Dict[str, float]:
    """Extract aggregate features from the candidate set.

    Args:
        prompt_sims: List of prompt similarity scores
        reflection_sims: List of reflection similarity scores
        negative_penalties: List of negative penalties
        candidates: List of candidate trajectories

    Returns:
        Dictionary of aggregate features
    """
    n = len(candidates)
    if n == 0:
        return {
            "candidate_count": 0.0,
            "prompt_sim_mean": 0.0,
            "prompt_sim_std": 0.0,
            "prompt_sim_max": 0.0,
            "prompt_sim_top3_gap": 0.0,
            "reflection_sim_mean": 0.0,
            "reflection_sim_std": 0.0,
            "reflection_candidate_ratio": 0.0,
            "negative_candidate_ratio": 0.0,
            "solved_candidate_ratio": 0.0,
        }

    # Prompt similarity statistics
    prompt_arr = np.array(prompt_sims)
    prompt_mean = float(np.mean(prompt_arr))
    prompt_std = float(np.std(prompt_arr))
    prompt_max = float(np.max(prompt_arr))

    # Top-3 gap (measures concentration)
    sorted_prompt = sorted(prompt_sims, reverse=True)
    if len(sorted_prompt) >= 3:
        prompt_top3_gap = sorted_prompt[0] - sorted_prompt[2]
    else:
        prompt_top3_gap = 0.0

    # Reflection similarity statistics
    reflection_arr = np.array(reflection_sims)
    reflection_mean = float(np.mean(reflection_arr))
    reflection_std = float(np.std(reflection_arr))

    # Candidate type ratios
    reflection_count = sum(1 for c in candidates if c.get("reflection_embedding") is not None)
    negative_count = sum(1 for p in negative_penalties if p > 0.1)
    solved_count = sum(1 for c in candidates if bool(c.get("is_solved", False)))

    features = {
        "candidate_count": float(n),
        "prompt_sim_mean": prompt_mean,
        "prompt_sim_std": prompt_std,
        "prompt_sim_max": prompt_max,
        "prompt_sim_top3_gap": float(prompt_top3_gap),
        "reflection_sim_mean": reflection_mean,
        "reflection_sim_std": reflection_std,
        "reflection_candidate_ratio": float(reflection_count) / max(1, n),
        "negative_candidate_ratio": float(negative_count) / max(1, n),
        "solved_candidate_ratio": float(solved_count) / max(1, n),
    }

    return features


def extract_context_features(state: Mapping[str, Any]) -> Dict[str, float]:
    """Extract context features from current problem state.

    Args:
        state: Current problem state dictionary

    Returns:
        Dictionary of context features
    """
    prompt = str(state.get("prompt", ""))
    reflections = _safe_list(state.get("reflections", []))
    feedback_history = _safe_list(state.get("test_feedback", []))

    # Failure pattern analysis
    failure_texts = " ".join(str(f).lower() for f in feedback_history)
    syntax_fail = float("syntax" in failure_texts or "indent" in failure_texts)
    runtime_fail = float("runtime" in failure_texts or "exception" in failure_texts)
    assert_fail = float("assert" in failure_texts or "expected" in failure_texts)
    timeout_fail = float("timeout" in failure_texts or "time limit" in failure_texts)

    features = {
        # Problem characteristics
        "prompt_char_len": float(len(prompt)),
        "prompt_line_len": float(len([l for l in prompt.splitlines() if l.strip()])),

        # Current state
        "current_attempt_count": float(_to_float(state.get("attempt_count", 0))),
        "current_reflection_rounds": float(len(reflections)),
        "current_failure_count": float(len(feedback_history)),

        # Failure patterns
        "has_syntax_fail": syntax_fail,
        "has_runtime_fail": runtime_fail,
        "has_assert_fail": assert_fail,
        "has_timeout_fail": timeout_fail,

        # Availability flags
        "reflection_available": float(len(reflections) > 0),
        "has_feedback": float(len(feedback_history) > 0),
    }

    return features


def build_transformer_router_input(
    state: Mapping[str, Any],
    candidates: List[Mapping[str, Any]],
    prompt_sims: List[float],
    reflection_sims: List[float],
    negative_penalties: List[float],
    max_candidates: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build input tensors for Transformer router.

    Args:
        state: Current problem state
        candidates: List of candidate trajectories
        prompt_sims: Prompt similarity scores
        reflection_sims: Reflection similarity scores
        negative_penalties: Negative penalties
        max_candidates: Maximum number of candidates to consider

    Returns:
        Tuple of (candidate_features, context_features, candidate_mask)
        - candidate_features: [max_candidates, feature_dim] array
        - context_features: [feature_dim] array
        - candidate_mask: [max_candidates] boolean array (True = padding)
    """
    # Extract context features
    context_feats = extract_context_features(state)

    # Extract candidate set features
    set_feats = extract_candidate_set_features(
        prompt_sims, reflection_sims, negative_penalties, candidates
    )

    # Combine context and set features
    context_vector = []
    for key in sorted(context_feats.keys()):
        context_vector.append(context_feats[key])
    for key in sorted(set_feats.keys()):
        context_vector.append(set_feats[key])
    context_features = np.array(context_vector, dtype=np.float32)

    # Extract per-candidate features
    n_candidates = min(len(candidates), max_candidates)
    candidate_feature_list = []

    for i in range(n_candidates):
        cand_feats = extract_candidate_features(
            candidates[i],
            prompt_sims[i] if i < len(prompt_sims) else 0.0,
            reflection_sims[i] if i < len(reflection_sims) else 0.0,
            negative_penalties[i] if i < len(negative_penalties) else 0.0,
        )
        cand_vector = [cand_feats[key] for key in sorted(cand_feats.keys())]
        candidate_feature_list.append(cand_vector)

    # Pad to max_candidates
    feature_dim = len(candidate_feature_list[0]) if candidate_feature_list else 10
    candidate_features = np.zeros((max_candidates, feature_dim), dtype=np.float32)
    candidate_mask = np.ones(max_candidates, dtype=bool)  # True = padding

    for i, feats in enumerate(candidate_feature_list):
        candidate_features[i] = feats
        candidate_mask[i] = False  # False = valid candidate

    return candidate_features, context_features, candidate_mask


# Feature dimension constants
CANDIDATE_FEATURE_DIM = 9  # Number of per-candidate features
CONTEXT_FEATURE_DIM = 11  # Number of context features
SET_FEATURE_DIM = 10  # Number of candidate-set features
TOTAL_CONTEXT_DIM = CONTEXT_FEATURE_DIM + SET_FEATURE_DIM  # 21
