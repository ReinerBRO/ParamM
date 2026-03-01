from .py_generate import PyGenerator
from .generator_types import Generator
from .model import (
    ModelBase,
    GPT4,
    GPT4turbo,
    GPT4o,
    GPT4oMini,
    GPT5Mini,
    o1,
    o1mini,
    GPT35,
    GPTDavinci,
    Llama3_1_405B,
    Llama3_1_70B,
    Llama3_1_8B,
    Llama2_7B,
    Mistral_7B,
    Qwen_7B,
    Qwen3_70B,
    Qwen_1dot5B,
    Qwen2_1dot5B,
    GPT_OSS_20B,
)


def generator_factory(lang: str) -> Generator:
    if lang == "game24":
        from .game24_generate import Game24Generator
        return Game24Generator()
    if lang == "math":
        from .MathQA_generate import MathQAGenerator
        return MathQAGenerator()
    if lang == "QA":
        from .MiltuhopQA_generate import MultiHopQAGenerator
        return MultiHopQAGenerator()
    if lang == "py" or lang == "python":
        return PyGenerator()
    if lang == "rs" or lang == "rust":
        from .rs_generate import RsGenerator
        return RsGenerator()
    raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        print("using GPT-4")
        return GPT4()
    if model_name == "gpt-4o":
        print("using GPT-4o")
        return GPT4o()
    if model_name == "gpt-4o-mini":
        print("using GPT-4o-mini")
        return GPT4oMini()
    if model_name == "gpt-5-mini":
        print("using GPT-5-mini")
        return GPT5Mini()
    if model_name == "gpt_oss_20b":
        return GPT_OSS_20B()
    if model_name == "o1":
        print("using o1")
        return o1()
    if model_name == "o1-mini":
        print("using o1-mini")
        return o1mini()
    if model_name == "gpt-4-turbo":
        print("using GPT-4-Turbo")
        return GPT4turbo()
    if model_name == "gpt-3.5-turbo":
        return GPT35()
    if model_name == "llama3_1_405b":
        print("using LLama 3.1 405B")
        return Llama3_1_405B()
    if model_name == "llama3_1_70b":
        print("using LLama 3.1 70B")
        return Llama3_1_70B()
    if model_name == "llama3_1_8b":
        print("using LLama 3.1 8B")
        return Llama3_1_8B()
    if model_name == "qwen_7b":
        print("using Qwen 7B")
        return Qwen_7B()
    if model_name == "qwen3_70b":
        print("using Qwen3 70B")
        return Qwen3_70B()
    if model_name == "qwen_1.5b":
        print("using Qwen 1.5B")
        return Qwen_1dot5B()
    if model_name == "mistral_7b":
        print("using Mistral 7B")
        return Mistral_7B()
    if model_name == "llama2_7b":
        print("using LLama 2 7B")
        return Llama2_7B()
    if model_name == "qwen2_1.5b":
        print("using Qwen2 1.5B (arize-ai)")
        return Qwen2_1dot5B()
    if model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    raise ValueError(f"Invalid model name: {model_name}")
