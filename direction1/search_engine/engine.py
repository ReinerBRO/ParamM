from dataclasses import dataclass
from typing import Iterable, List, Optional

from .candidate_generator import CandidateGeneratorAdapter
from .config import SearchConfig
from .node import SearchNode
from .queue import BeamQueue
from .scorer import complexity_penalty, jaccard_diversity, score_candidate
from .state import SearchState
from .verifier_adapter import VerifierAdapter


@dataclass
class SearchResult:
    best_node: Optional[SearchNode]
    nodes_expanded: int
    verifier_cost: float
    best_score_trace: List[float]


class SearchEngine:
    def __init__(
        self,
        config: SearchConfig,
        candidate_generator: CandidateGeneratorAdapter,
        verifier: VerifierAdapter,
    ) -> None:
        self.config = config
        self.candidate_generator = candidate_generator
        self.verifier = verifier

    def _score_node(self, node: SearchNode, previous_reflections: Iterable[str]) -> tuple[float, float, float, bool]:
        pass_prob, cost, all_passed = self.verifier.estimate_pass_probability(node.code)
        diversity = jaccard_diversity(node.reflection, previous_reflections)
        penalty = complexity_penalty(node.code, self.config)
        score = score_candidate(pass_prob, diversity, penalty, self.config)
        node.pass_prob = pass_prob
        node.diversity = diversity
        node.complexity_penalty = penalty
        node.score = score
        return score, cost, pass_prob, all_passed

    def run(self, initial_nodes: List[SearchNode] | None = None, **generation_kwargs: object) -> SearchResult:
        state = SearchState()
        frontier = BeamQueue(self.config.beam_size)
        reflection_history: List[str] = []

        seeds = initial_nodes if initial_nodes is not None else self.candidate_generator.generate(**generation_kwargs)
        for seed in seeds:
            state.register_node(seed)
            _, cost, _, all_passed = self._score_node(seed, reflection_history)
            state.verifier_cost += cost
            frontier.push(seed)
            reflection_history.append(seed.reflection)
            state.update_best(seed)

        while len(frontier) > 0 and state.expanded_nodes < self.config.max_nodes:
            parent = frontier.pop_best()
            if parent.depth >= self.config.max_depth:
                continue

            children = self.candidate_generator.generate(parent=parent, **generation_kwargs)
            if not children:
                continue

            state.expanded_nodes += 1
            for child in children:
                child.depth = parent.depth + 1
                child.parent_id = parent.node_id
                state.register_node(child)

                _, cost, _, all_passed = self._score_node(child, reflection_history)
                state.verifier_cost += cost
                reflection_history.append(child.reflection)

                frontier.push(child)
                state.update_best(child)

                if self.config.early_stop and all_passed:
                    return SearchResult(
                        state.best_node,
                        state.expanded_nodes,
                        state.verifier_cost,
                        state.best_score_trace,
                    )

        return SearchResult(state.best_node, state.expanded_nodes, state.verifier_cost, state.best_score_trace)
