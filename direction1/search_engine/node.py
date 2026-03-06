from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SearchNode:
    """A candidate node explored by the search engine."""

    code: str
    reflection: str = ""
    depth: int = 0
    parent_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    node_id: int = field(default=-1)
    pass_prob: float = 0.0
    diversity: float = 0.0
    complexity_penalty: float = 0.0
    score: float = float("-inf")

    def short_code(self, length: int = 80) -> str:
        return self.code[:length]
