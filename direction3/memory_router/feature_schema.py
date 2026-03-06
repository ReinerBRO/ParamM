"""Feature schema and robust state feature extraction for direction3 router."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

FEATURE_ORDER: List[str] = [
    "prompt_char_len",
    "prompt_word_len",
    "prompt_line_len",
    "code_char_len",
    "code_line_len",
    "phase1_exists",
    "phase2_exists",
    "reflection_rounds",
    "attempt_count",
    "failure_count",
    "test_failed_count",
    "syntax_fail_count",
    "runtime_fail_count",
    "assert_fail_count",
    "timeout_fail_count",
    "prompt_sim_max",
    "prompt_sim_mean",
    "reflection_sim_max",
    "reflection_sim_mean",
    "negative_sim_max",
    "negative_sim_mean",
    "retrieval_candidate_count",
    "solved_history_flag",
]


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


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _get_nested(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _flatten_numeric(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for item in values:
        if isinstance(item, Mapping):
            candidate = _get_nested(
                item,
                [
                    "score",
                    "sim",
                    "similarity",
                    "value",
                    "distance",
                    "cosine",
                ],
            )
            out.append(_to_float(candidate, 0.0))
        else:
            out.append(_to_float(item, 0.0))
    return out


def _collect_similarity_values(state: Mapping[str, Any], prefixes: Iterable[str]) -> List[float]:
    values: List[float] = []
    mem_section = _safe_dict(_get_nested(state, ["memory", "memories", "memory_stats"], {}))

    for prefix in prefixes:
        direct = _get_nested(
            state,
            [
                f"{prefix}_sims",
                f"{prefix}_sim_list",
                f"{prefix}_scores",
                f"{prefix}_similarities",
                f"{prefix}_sim",
                f"{prefix}_similarity",
            ],
            None,
        )
        if isinstance(direct, (list, tuple)):
            values.extend(_flatten_numeric(direct))
        elif direct is not None:
            values.append(_to_float(direct, 0.0))

        mem_direct = _get_nested(
            mem_section,
            [
                f"{prefix}_sims",
                f"{prefix}_scores",
                f"{prefix}_sim",
                f"{prefix}_similarity",
            ],
            None,
        )
        if isinstance(mem_direct, (list, tuple)):
            values.extend(_flatten_numeric(mem_direct))
        elif mem_direct is not None:
            values.append(_to_float(mem_direct, 0.0))

    return [v for v in values if v == v]


def _max_and_mean(values: List[float]) -> List[float]:
    if not values:
        return [0.0, 0.0]
    max_v = max(values)
    mean_v = sum(values) / max(1, len(values))
    return [max_v, mean_v]


def extract_state_features(state: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Extract robust numeric features; missing fields fall back to safe defaults."""
    safe_state = dict(state or {})

    prompt = _to_text(_get_nested(safe_state, ["prompt", "question", "task_prompt", "description"], ""))
    prompt_words = [w for w in prompt.replace("\n", " ").split(" ") if w]
    prompt_lines = [line for line in prompt.splitlines() if line.strip()]

    reflections = _safe_list(_get_nested(safe_state, ["reflections", "reflection_history", "reflection_list"], []))
    attempts = _safe_list(_get_nested(safe_state, ["attempts", "runs", "executions", "implementations"], []))
    failures = _safe_list(_get_nested(safe_state, ["failures", "error_history", "failed_attempts"], []))

    code_text = _to_text(
        _get_nested(
            safe_state,
            ["solution", "code", "generated_code", "gen_solution", "current_solution"],
            "",
        )
    )

    failure_texts = [
        _to_text(x).lower()
        for x in (failures + _safe_list(_get_nested(safe_state, ["errors", "error_messages"], [])))
    ]

    syntax_count = sum(1 for t in failure_texts if "syntax" in t or "indent" in t)
    runtime_count = sum(1 for t in failure_texts if "runtime" in t or "exception" in t or "traceback" in t)
    assert_count = sum(1 for t in failure_texts if "assert" in t or "expected" in t)
    timeout_count = sum(1 for t in failure_texts if "timeout" in t or "time limit" in t)

    prompt_sims = _collect_similarity_values(safe_state, ["prompt", "query", "problem"])
    reflection_sims = _collect_similarity_values(safe_state, ["reflection", "self_reflection", "analysis"])
    negative_sims = _collect_similarity_values(safe_state, ["negative", "bad", "failure"])

    prompt_max, prompt_mean = _max_and_mean(prompt_sims)
    reflection_max, reflection_mean = _max_and_mean(reflection_sims)
    negative_max, negative_mean = _max_and_mean(negative_sims)

    retrieval_raw = _get_nested(
        safe_state,
        ["retrieval_candidates", "retrieved_memories", "memory_candidates"],
        None,
    )
    if retrieval_raw is None:
        mem_sec = _safe_dict(_get_nested(safe_state, ["memory", "memories", "memory_stats"], {}))
        retrieval_raw = _get_nested(
            mem_sec,
            ["retrieval_candidates", "candidate_count", "retrieval_candidate_count"],
            None,
        )

    if isinstance(retrieval_raw, (int, float, bool, str)):
        retrieval_count = max(0.0, _to_float(retrieval_raw, 0.0))
    else:
        retrieval_count = float(len(_safe_list(retrieval_raw)))

    test_feedbacks = _safe_list(_get_nested(safe_state, ["test_feedback", "tests_feedback", "feedback_history"], []))
    if not test_feedbacks:
        test_feedbacks = _safe_list(_get_nested(safe_state, ["phase1", "test_feedback"], [])) + _safe_list(
            _get_nested(safe_state, ["phase2", "test_feedback"], [])
        )
    test_failed_count = 0
    for fb in test_feedbacks:
        txt = _to_text(fb)
        if "Tests failed:" in txt:
            test_failed_count += txt.split("Tests failed:", 1)[1].count("assert")
        else:
            low = txt.lower()
            if "assert" in low:
                test_failed_count += low.count("assert")

    features: Dict[str, float] = {
        "prompt_char_len": float(len(prompt)),
        "prompt_word_len": float(len(prompt_words)),
        "prompt_line_len": float(len(prompt_lines)),
        "code_char_len": float(len(code_text)),
        "code_line_len": float(len([line for line in code_text.splitlines() if line.strip()])),
        "phase1_exists": float(bool(_get_nested(safe_state, ["phase1", "phase1_log", "first_stage"], None))),
        "phase2_exists": float(bool(_get_nested(safe_state, ["phase2", "phase2_log", "second_stage"], None))),
        "reflection_rounds": float(len(reflections)),
        "attempt_count": float(max(len(attempts), _to_float(_get_nested(safe_state, ["attempt_count"], 0.0)))),
        "failure_count": float(max(len(failures), len(failure_texts), test_failed_count)),
        "test_failed_count": float(test_failed_count),
        "syntax_fail_count": float(syntax_count),
        "runtime_fail_count": float(runtime_count),
        "assert_fail_count": float(assert_count),
        "timeout_fail_count": float(timeout_count),
        "prompt_sim_max": float(prompt_max),
        "prompt_sim_mean": float(prompt_mean),
        "reflection_sim_max": float(reflection_max),
        "reflection_sim_mean": float(reflection_mean),
        "negative_sim_max": float(negative_max),
        "negative_sim_mean": float(negative_mean),
        "retrieval_candidate_count": float(retrieval_count),
        "solved_history_flag": float(
            bool(
                _get_nested(
                    safe_state,
                    [
                        "solved_before",
                        "history_solved",
                        "history_passed",
                        "historical_pass",
                        "had_passed_before",
                    ],
                    False,
                )
            )
        ),
    }

    for name in FEATURE_ORDER:
        features.setdefault(name, 0.0)

    return features


def vectorize_features(feature_dict: Optional[Mapping[str, Any]], feature_order: Optional[List[str]] = None) -> List[float]:
    order = feature_order or FEATURE_ORDER
    candidate = dict(feature_dict or {})

    # Accept both raw state dict and pre-extracted feature dict.
    if any(name in candidate for name in order):
        return [_to_float(candidate.get(name, 0.0), 0.0) for name in order]

    features = extract_state_features(candidate)
    return [_to_float(features.get(name, 0.0), 0.0) for name in order]


def normalize_mix(raw_weights: Iterable[float]) -> List[float]:
    vals = [max(0.0, _to_float(v, 0.0)) for v in raw_weights]
    if len(vals) != 3:
        vals = (vals + [0.0, 0.0, 0.0])[:3]
    denom = sum(vals)
    if denom <= 1e-12:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [v / denom for v in vals]
