from typing import Iterable, Set

from .config import SearchConfig


def _tokenize(text: str) -> Set[str]:
    return {tok for tok in text.lower().split() if tok}


def jaccard_diversity(reflection: str, previous_reflections: Iterable[str]) -> float:
    """Return max diversity score against prior reflections using Jaccard distance."""

    tokens = _tokenize(reflection)
    if not tokens:
        return 0.0

    best_distance = 0.0
    has_previous = False
    for prev in previous_reflections:
        prev_tokens = _tokenize(prev)
        has_previous = True
        if not prev_tokens:
            best_distance = max(best_distance, 1.0)
            continue
        union = tokens | prev_tokens
        if not union:
            continue
        similarity = len(tokens & prev_tokens) / len(union)
        best_distance = max(best_distance, 1.0 - similarity)

    return best_distance if has_previous else 1.0


def complexity_penalty(code: str, config: SearchConfig) -> float:
    chars = len(code)
    lines = code.count("\n") + 1 if code else 0
    return (chars / config.complexity_char_scale) + (lines / config.complexity_line_scale)


def score_candidate(
    pass_prob: float,
    diversity: float,
    complexity_penalty_value: float,
    config: SearchConfig,
) -> float:
    return (
        pass_prob * config.weight_pass_prob
        + diversity * config.weight_diversity
        - complexity_penalty_value * config.weight_complexity
    )
