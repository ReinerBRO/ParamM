from dataclasses import dataclass, field
from typing import List, Optional

from .node import SearchNode


@dataclass
class SearchState:
    """Mutable state maintained during a search run."""

    expanded_nodes: int = 0
    verifier_cost: float = 0.0
    next_node_id: int = 0

    best_node: Optional[SearchNode] = None
    best_score_trace: List[float] = field(default_factory=list)
    all_nodes: List[SearchNode] = field(default_factory=list)

    def register_node(self, node: SearchNode) -> SearchNode:
        node.node_id = self.next_node_id
        self.next_node_id += 1
        self.all_nodes.append(node)
        return node

    def update_best(self, node: SearchNode) -> None:
        if self.best_node is None or node.score > self.best_node.score:
            self.best_node = node
        self.best_score_trace.append(self.best_node.score if self.best_node else float("-inf"))
