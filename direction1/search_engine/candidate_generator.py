from typing import Any, Callable, Iterable, List

from .node import SearchNode


GenerateCandidatesFn = Callable[..., Iterable[SearchNode]]


class CandidateGeneratorAdapter:
    """Thin adapter layer around user-provided candidate generation."""

    def __init__(self, generator_fn: GenerateCandidatesFn) -> None:
        self._generator_fn = generator_fn

    def generate(self, *args: Any, **kwargs: Any) -> List[SearchNode]:
        # Keep call signature/prompting logic in caller; this only normalizes output type.
        return list(self._generator_fn(*args, **kwargs))
