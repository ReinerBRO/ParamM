from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for the search engine."""

    beam_size: int = 3
    max_nodes: int = 50
    max_depth: int = 4
    early_stop: bool = True

    weight_pass_prob: float = 0.6
    weight_diversity: float = 0.25
    weight_complexity: float = 0.15

    complexity_char_scale: float = 1000.0
    complexity_line_scale: float = 100.0

    visible_test_weight: float = 0.6
    generated_test_weight: float = 0.3
    static_check_weight: float = 0.1
