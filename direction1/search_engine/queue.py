import heapq
from typing import List

from .node import SearchNode


class BeamQueue:
    """A max-score queue with optional beam truncation."""

    def __init__(self, beam_size: int) -> None:
        self.beam_size = beam_size
        self._heap: List[tuple[float, int, SearchNode]] = []

    def push(self, node: SearchNode) -> None:
        # Min-heap by score, then node id for stability.
        heapq.heappush(self._heap, (node.score, node.node_id, node))
        if len(self._heap) > self.beam_size:
            heapq.heappop(self._heap)

    def push_many(self, nodes: List[SearchNode]) -> None:
        for node in nodes:
            self.push(node)

    def pop_best(self) -> SearchNode:
        if not self._heap:
            raise IndexError("BeamQueue is empty")
        _, _, node = max(self._heap)
        self._heap.remove((node.score, node.node_id, node))
        heapq.heapify(self._heap)
        return node

    def drain_desc(self) -> List[SearchNode]:
        items = sorted(self._heap, reverse=True)
        self._heap.clear()
        return [node for _, _, node in items]

    def __len__(self) -> int:
        return len(self._heap)
