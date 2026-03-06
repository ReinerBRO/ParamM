from dataclasses import dataclass
import hashlib
from typing import Callable, Dict, Tuple

from .config import SearchConfig


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    pass_rate: float
    cost: float
    details: str = ""


VerifierFn = Callable[[str], VerificationResult]


def _default_verifier(_: str) -> VerificationResult:
    return VerificationResult(passed=False, pass_rate=0.0, cost=0.0, details="default verifier")


class VerifierAdapter:
    """Adapter for verifier calls with code-hash based caching."""

    def __init__(
        self,
        config: SearchConfig,
        visible_tests_fn: VerifierFn | None = None,
        generated_tests_fn: VerifierFn | None = None,
        static_checks_fn: VerifierFn | None = None,
    ) -> None:
        self.config = config
        self._visible_tests_fn = visible_tests_fn or _default_verifier
        self._generated_tests_fn = generated_tests_fn or _default_verifier
        self._static_checks_fn = static_checks_fn or _default_verifier
        self._cache: Dict[Tuple[str, str], VerificationResult] = {}

    @staticmethod
    def _code_hash(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _run_cached(self, name: str, fn: VerifierFn, code: str) -> VerificationResult:
        key = (name, self._code_hash(code))
        if key in self._cache:
            return self._cache[key]
        result = fn(code)
        self._cache[key] = result
        return result

    def run_visible_tests(self, code: str) -> VerificationResult:
        return self._run_cached("visible", self._visible_tests_fn, code)

    def run_generated_tests(self, code: str) -> VerificationResult:
        return self._run_cached("generated", self._generated_tests_fn, code)

    def run_static_checks(self, code: str) -> VerificationResult:
        return self._run_cached("static", self._static_checks_fn, code)

    def estimate_pass_probability(self, code: str) -> tuple[float, float, bool]:
        """Return weighted pass probability, total verifier cost, and all-pass flag."""

        visible = self.run_visible_tests(code)
        generated = self.run_generated_tests(code)
        static = self.run_static_checks(code)

        weighted = (
            visible.pass_rate * self.config.visible_test_weight
            + generated.pass_rate * self.config.generated_test_weight
            + static.pass_rate * self.config.static_check_weight
        )
        cost = visible.cost + generated.cost + static.cost
        all_passed = visible.passed and generated.passed and static.passed
        return weighted, cost, all_passed
