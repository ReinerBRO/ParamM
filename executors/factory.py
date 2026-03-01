from .py_executor import PyExecutor
from .executor_types import Executor
from .leet_executor import LeetExecutor


def executor_factory(lang: str, is_leet: bool = False) -> Executor:
    if lang == "game24":
        from .game24_executor import Game24Executor
        return Game24Executor()
    if lang == "QA":
        from .MultihopQA_executor import MultiHopQAExecutor
        return MultiHopQAExecutor()
    if lang == "math":
        from .Math_executor import MathExecutor
        return MathExecutor()

    if lang == "py" or lang == "python":
        if is_leet:
            print("Using LeetCode Python executor")
            from .leetcode_env.types import ProgrammingLanguage
            from .leetcode_env.utils.formatting import PythonSubmissionFormatter as PySubmissionFormatter
            return LeetExecutor(
                ProgrammingLanguage.PYTHON3,
                PyExecutor(),
                PySubmissionFormatter,
            )
        return PyExecutor()

    if lang == "rs" or lang == "rust":
        from .rs_executor import RsExecutor
        if is_leet:
            from .leetcode_env.types import ProgrammingLanguage
            from .leetcode_env.utils.formatting import RustSubmissionFormatter as RsSubmissionFormatter
            return LeetExecutor(
                ProgrammingLanguage.RUST,
                RsExecutor(),
                RsSubmissionFormatter,
            )
        return RsExecutor()

    raise ValueError(f"Invalid language for executor: {lang}")
