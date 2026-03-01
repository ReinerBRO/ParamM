# ParamAgent HumanEval 复现计划

> 本文档面向另一个 AI agent 执行。请严格按照顺序完成每个步骤，每步完成后执行验证命令，全部通过后再进入下一步。

## 目标

复现论文 Table 1 中 **HumanEval + Llama-3.1-8B** 的 ParamAgent 和 ParamAgent-plus 实验。

- **ParamAgent** (Phase 1): 论文报告 Pass@1 = **82.93%** (136/164)
- **ParamAgent-plus** (Phase 2): 论文报告约 **86.59%** (Table 2, 自改进版；GPT-4o-mini 标注版会更高)

使用预计算的 pitfall 文件（GPT-4o-mini 标注），无需本地 GPU。

---

## 前置条件

- Python 3.10+
- `TOGETHER_API_KEY`: Together AI API key（Llama-3.1-8B 推理用）
- `OPENAI_API_KEY`: OpenAI API key（text-embedding-3-small 用于 memory bank 检索）
- 如果使用中转 API，需配置 `OPENAI_BASE_URL`

---

## 步骤 1：修复 Python 依赖

### 问题

`requirements.txt` 缺少以下运行时必需包：
- `together`: `generators/model.py` 和 `generators/py_generate.py` 中 `from together import Together`
- `numpy`: `memory_utils.py` 中 `import numpy as np`
- `tqdm`: `paramAgent.py` 中 `from tqdm import tqdm`
- `dashscope`: `generators/model.py` 中 `import dashscope`（可选，仅 Qwen DashScope 后端用）
- `peft`, `transformers`, `accelerate`: 仅本地 LoRA 推理用，本实验不需要

### 操作

编辑 `requirements.txt`，添加：

```
together>=1.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

**不要**添加 `dashscope`（非必须，import 时不会立即触发）。

### 验证

```bash
pip install -r requirements.txt
python -c "import together; import numpy; import tqdm; print('deps OK')"
```

**通过标准**: 输出 `deps OK`，无报错。

---

## 步骤 2：创建 `generators/__init__.py`

### 问题

`paramAgent.py:12` 执行 `from generators import generator_factory, model_factory`，但 `generators/` 目录没有 `__init__.py`。

### 操作

创建文件 `generators/__init__.py`，内容：

```python
from .factory import generator_factory, model_factory
```

**注意**：不要直接写 `from .factory import *`，因为 `factory.py` 内部还 import 了大量缺失模块（下一步修复）。

### 验证

暂不验证（依赖步骤 3-5 完成后才能 import）。

---

## 步骤 3：创建 `executors/__init__.py`

### 问题

`paramAgent.py:11` 执行 `from executors import executor_factory`，但 `executors/` 目录没有 `__init__.py`。

### 操作

创建文件 `executors/__init__.py`，内容：

```python
from .factory import executor_factory
```

### 验证

暂不验证（依赖步骤 4 完成后才能 import）。

---

## 步骤 4：修复 `executors/factory.py` — 去掉缺失模块 import

### 问题

`executors/factory.py` 文件头 import 了 4 个不存在的模块：

```python
from .rs_executor import RsExecutor          # 文件不存在
from .game24_executor import Game24Executor   # 文件不存在
from .MultihopQA_executor import MultiHopQAExecutor  # 文件不存在
from .Math_executor import MathExecutor       # 文件不存在
```

这些 import 位于模块顶层，Python 加载 `executors` 包时会立即报 `ModuleNotFoundError`。

### 操作

将 `executors/factory.py` 修改为：

```python
from .py_executor import PyExecutor
from .executor_types import Executor
from .leet_executor import LeetExecutor

# 以下模块仅在对应 language 参数时才需要，延迟 import 避免缺失文件报错
# from .rs_executor import RsExecutor
# from .game24_executor import Game24Executor
# from .MultihopQA_executor import MultiHopQAExecutor
# from .Math_executor import MathExecutor

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
            return LeetExecutor(ProgrammingLanguage.PYTHON3,
                                PyExecutor(),
                                PySubmissionFormatter)
        else:
            return PyExecutor()
    elif lang == "rs" or lang == "rust":
        from .rs_executor import RsExecutor
        if is_leet:
            from .leetcode_env.types import ProgrammingLanguage
            from .leetcode_env.utils.formatting import PySubmissionFormatter
            from .leetcode_env.utils.formatting import RustSubmissionFormatter as RsSubmissionFormatter
            return LeetExecutor(ProgrammingLanguage.RUST,
                                RsExecutor(),
                                RsSubmissionFormatter)
        else:
            return RsExecutor()
    else:
        raise ValueError(f"Invalid language for executor: {lang}")
```

**核心改动**：将 `rs_executor`, `game24_executor`, `MultihopQA_executor`, `Math_executor` 从顶层 import 改为各分支内延迟 import。

### 验证

```bash
python -c "from executors import executor_factory; exe = executor_factory('py'); print(type(exe))"
```

**通过标准**: 输出 `<class 'executors.py_executor.PyExecutor'>`，无 ImportError。

---

## 步骤 5：修复 `generators/factory.py` — 去掉缺失模块和缺失类 import

### 问题

`generators/factory.py` 有两类问题：

**问题 A — 缺失模块文件**（顶层 import 直接 crash）：
```python
from .rs_generate import RsGenerator              # 文件不存在
from .game24_generate import Game24Generator       # 文件不存在
from .MiltuhopQA_generate import MultiHopQAGenerator  # 文件不存在
from .MathQA_generate import MathQAGenerator       # 文件不存在
```

**问题 B — model.py 中缺失的类**（factory.py 从 model.py import 了 4 个不存在的类）：
```python
from .model import CodeLlama, StarChat, Sonnet3, Sonnet35  # 这 4 个类在 model.py 中不存在
```

### 操作

将 `generators/factory.py` 修改为：

```python
from .py_generate import PyGenerator
from .generator_types import Generator
from .model import ModelBase, \
                    GPT4, \
                    GPT4turbo, \
                    GPT4o, \
                    GPT4oMini, \
                    GPT5Mini, \
                    o1, o1mini, \
                    GPT35, \
                    GPTDavinci, \
                    Llama3_1_405B, Llama3_1_70B, Llama3_1_8B, Llama2_7B, \
                    Mistral_7B, Qwen_7B, Qwen3_70B, Qwen_1dot5B, Qwen2_1dot5B, \
                    GPT_OSS_20B

# 以下模块/类缺失，延迟 import
# from .rs_generate import RsGenerator
# from .game24_generate import Game24Generator
# from .MiltuhopQA_generate import MultiHopQAGenerator
# from .MathQA_generate import MathQAGenerator
# from .model import CodeLlama, StarChat, Sonnet3, Sonnet35


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
    elif lang == "rs" or lang == "rust":
        from .rs_generate import RsGenerator
        return RsGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        print("using GPT-4")
        return GPT4()
    elif model_name == "gpt-4o":
        print("using GPT-4o")
        return GPT4o()
    elif model_name == "gpt-4o-mini":
        print("using GPT-4o-mini")
        return GPT4oMini()
    elif model_name == "gpt-5-mini":
        print("using GPT-5-mini")
        return GPT5Mini()
    elif model_name == "gpt_oss_20b":
        return GPT_OSS_20B()
    elif model_name == "o1":
        print("using o1")
        return o1()
    elif model_name == "o1-mini":
        print("using o1-mini")
        return o1mini()
    elif model_name == "gpt-4-turbo":
        print("using GPT-4-Turbo")
        return GPT4turbo()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name == "llama3_1_405b":
        print("using LLama 3.1 405B")
        return Llama3_1_405B()
    elif model_name == "llama3_1_70b":
        print("using LLama 3.1 70B")
        return Llama3_1_70B()
    elif model_name == "llama3_1_8b":
        print("using LLama 3.1 8B")
        return Llama3_1_8B()
    elif model_name == "qwen_7b":
        print("using Qwen 7B")
        return Qwen_7B()
    elif model_name == "qwen3_70b":
        print("using Qwen3 70B")
        return Qwen3_70B()
    elif model_name == "qwen_1.5b":
        print("using Qwen 1.5B")
        return Qwen_1dot5B()
    elif model_name == "mistral_7b":
        print("using Mistral 7B")
        return Mistral_7B()
    elif model_name == "llama2_7b":
        print("using LLama 2 7B")
        return Llama2_7B()
    elif model_name == "qwen2_1.5b":
        print("using Qwen2 1.5B (arize-ai)")
        return Qwen2_1dot5B()
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
```

**核心改动**：
1. 删除 `rs_generate`, `game24_generate`, `MiltuhopQA_generate`, `MathQA_generate` 的顶层 import，改为分支内延迟 import
2. 删除 `CodeLlama`, `StarChat`, `Sonnet3`, `Sonnet35` 的 import（这些类在 `model.py` 中不存在）
3. 删除 `model_factory` 中引用 `CodeLlama`/`StarChat` 的分支（`starchat` 和 `codellama` 分支）

### 验证

```bash
python -c "from generators import generator_factory, model_factory; gen = generator_factory('py'); model = model_factory('llama3_1_8b'); print(type(gen), type(model))"
```

**通过标准**: 输出含 `PyGenerator` 和 `Llama3_1_8B`，无 ImportError。

---

## 步骤 6：修复 `main_param.py` — 处理缺失的 `reflexion_parametric` import

### 问题

`main_param.py:5` 有：
```python
from reflexion_parametric import run_reflexion
```
但 `reflexion_parametric.py` 文件不存在。这是 Reflexion + ParamMem 变体，本实验用 `--strategy dot`，不会调用 `run_reflexion`，但 import 在模块顶层，Python 加载时直接 crash。

### 操作

编辑 `main_param.py` 第 5 行，将：

```python
from reflexion_parametric import run_reflexion
```

改为：

```python
try:
    from reflexion_parametric import run_reflexion
except ImportError:
    from reflexion import run_reflexion  # fallback to standard reflexion
```

### 验证

```bash
python -c "import main_param; print('main_param import OK')"
```

**通过标准**: 输出 `main_param import OK`，无 ImportError。（可能打印 model loading 信息，忽略即可）

---

## 步骤 7：生成 `benchmarks/humaneval_visible_tests.jsonl`

### 问题

`main_param.py:138` 在 HumanEval 数据集时执行：
```python
visible_tests = read_jsonl_map("benchmarks/humaneval_visible_tests.jsonl", primary_key='entry_point')
```

使用处 (`paramAgent.py:263`)：
```python
tests_i = visible_tests[identifier]['given_tests']
```

需要每个 HumanEval 题目有 `entry_point` 和 `given_tests`（assert 语句列表）。此文件不存在。

### 操作

创建脚本 `scripts/gen_visible_tests.py`：

```python
"""
从 HumanEval 的 docstring 中提取 >>> 示例，转为 assert 语句，
生成 benchmarks/humaneval_visible_tests.jsonl。

用法: python scripts/gen_visible_tests.py
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_jsonl


def extract_doctest_asserts(prompt: str, entry_point: str) -> list[str]:
    """从 prompt 的 docstring 中提取 >>> 行并转为 assert 语句。"""
    asserts = []
    lines = prompt.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>>>'):
            expr = line[3:].strip()
            # 查看下一行是否为预期输出（非 >>> 开头）
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith('>>>'):
                    # 构造 assert
                    asserts.append(f"assert {expr} == {next_line}")
                    i += 2
                    continue
        i += 1
    return asserts


def main():
    dataset = read_jsonl("benchmarks/humaneval_full.jsonl")
    output = []
    for item in dataset:
        entry_point = item["entry_point"]
        prompt = item["prompt"]
        given_tests = extract_doctest_asserts(prompt, entry_point)
        output.append({
            "entry_point": entry_point,
            "given_tests": given_tests,
        })

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/humaneval_visible_tests.jsonl", "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    # 统计
    total = len(output)
    with_tests = sum(1 for item in output if len(item["given_tests"]) > 0)
    print(f"Generated {total} entries, {with_tests} have visible tests")


if __name__ == "__main__":
    main()
```

然后执行：

```bash
mkdir -p scripts
# （将上面的脚本写入 scripts/gen_visible_tests.py 后）
python scripts/gen_visible_tests.py
```

### 验证

```bash
python -c "
from utils import read_jsonl_map
vt = read_jsonl_map('benchmarks/humaneval_visible_tests.jsonl', primary_key='entry_point')
print(f'entries: {len(vt)}')
# 检查第一个条目
sample = list(vt.values())[0]
print(f'sample keys: {list(sample.keys())}')
print(f'sample given_tests count: {len(sample[\"given_tests\"])}')
assert len(vt) == 164, f'Expected 164, got {len(vt)}'
assert all('given_tests' in v for v in vt.values()), 'Missing given_tests'
print('visible_tests OK')
"
```

**通过标准**:
- 输出 `entries: 164`
- `visible_tests OK`
- 大部分条目的 `given_tests` 非空（允许少量为空，因为某些 HumanEval 题目 docstring 无 `>>>` 示例）

### 备注

如果提取效果不理想（given_tests 大量为空），还有备选方案：
- 将 `main_param.py:138` 的 `visible_tests` 改为 `None`，让代码走 `gen.internal_tests()` 分支（由 LLM 生成合成测试用例）。这会增加 API 调用但不影响正确性。
- 从 DoT 原始仓库 (https://github.com/amazon-science/DiversityOfThoughts) 获取此文件。

---

## 步骤 8：处理 `dashscope` import 警告

### 问题

`generators/model.py:21-22` 有：
```python
import dashscope
from dashscope import Generation
```
如果未安装 `dashscope`，import `generators/model.py` 时 crash。但本实验只用 Together AI 后端，不需要 DashScope。

### 操作

编辑 `generators/model.py`，将第 21-22 行：

```python
import dashscope
from dashscope import Generation
```

改为：

```python
try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None
```

### 验证

```bash
python -c "from generators.model import Llama3_1_8B; m = Llama3_1_8B(); print(f'model: {m.name}')"
```

**通过标准**: 输出 `model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`。

---

## 步骤 9：配置 API Keys

### 操作

设置环境变量（或写入 `.env` 并确保代码加载 `.env`）：

```bash
export TOGETHER_API_KEY="你的 Together AI key"
export OPENAI_API_KEY="你的 OpenAI key"
# 如果用中转 API:
# export OPENAI_BASE_URL="https://api.zhizengzeng.com/v1"
```

### 验证

```bash
# 验证 Together AI
python -c "
from together import Together
import os
client = Together(api_key=os.environ['TOGETHER_API_KEY'])
resp = client.chat.completions.create(
    model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    messages=[{'role':'user','content':'Say hello'}],
    max_tokens=10
)
print('Together AI OK:', resp.choices[0].message.content[:30])
"

# 验证 OpenAI Embedding
python -c "
from openai import OpenAI
client = OpenAI()
resp = client.embeddings.create(model='text-embedding-3-small', input=['test'])
print('OpenAI Embedding OK, dim:', len(resp.data[0].embedding))
"
```

**通过标准**:
- Together AI 返回有效文本
- OpenAI Embedding 返回维度 1536 的向量

---

## 步骤 10：端到端冒烟测试（2 题）

先用 2 题快速验证流程跑通。

### 操作

```bash
python main_param.py \
    --run_name "smoke_test" \
    --root_dir ./results/humaneval/smoke/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k 1 \
    --max_iters 2 \
    --inner_iter 2 \
    --use_mistakes \
    --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
    --num_samples 2 \
    --verbose
```

### 验证

```bash
python -c "
from utils import read_jsonl
import os

root = './results/humaneval/smoke/smoke_test'

# 检查 first stage log
first = read_jsonl(os.path.join(root, 'first_stage_log.jsonl'))
print(f'First stage entries: {len(first)}')
assert len(first) == 2, f'Expected 2, got {len(first)}'

# 检查字段完整性
for item in first:
    assert 'is_solved' in item, 'Missing is_solved'
    assert 'solution' in item, 'Missing solution'
    assert 'cost' in item, 'Missing cost'
    assert 'original_index' in item, 'Missing original_index'

# 检查 second stage log（如果有失败题）
second_path = os.path.join(root, 'second_stage_log.jsonl')
if os.path.exists(second_path):
    second = read_jsonl(second_path)
    print(f'Second stage entries: {len(second)}')

# 检查 memory bank
mem_path = os.path.join(root, 'mem_bank.pkl')
assert os.path.exists(mem_path), 'Memory bank not created'

print('Smoke test PASSED')
"
```

**通过标准**:
- 无 Python 异常退出
- `first_stage_log.jsonl` 包含 2 条记录
- 每条记录有 `is_solved`, `solution`, `cost`, `original_index` 字段
- `mem_bank.pkl` 文件存在
- 如果有题目第一阶段失败，`second_stage_log.jsonl` 也应存在

---

## 步骤 11：运行完整 HumanEval 实验

### 操作

```bash
python main_param.py \
    --run_name "paramAgent_humaneval_llama8b" \
    --root_dir ./results/humaneval/paramAgent/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k 1 \
    --max_iters 5 \
    --inner_iter 5 \
    --use_mistakes \
    --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
    --verbose
```

预计运行时间：视 API 速率，数小时至一天。

### 验证

```bash
python -c "
from utils import read_jsonl
import os

run_dir = './results/humaneval/paramAgent/paramAgent_humaneval_llama8b'

# ===== Phase 1: ParamAgent =====
first = read_jsonl(os.path.join(run_dir, 'first_stage_log.jsonl'))
assert len(first) == 164, f'Expected 164, got {len(first)}'
solved_1 = sum(1 for x in first if x.get('is_solved', False))
pass1_1 = solved_1 / 164 * 100
print(f'Phase 1 (ParamAgent): {solved_1}/164 = {pass1_1:.2f}%')

# ===== Phase 2: ParamAgent-plus =====
second_path = os.path.join(run_dir, 'second_stage_log.jsonl')
if os.path.exists(second_path):
    second = read_jsonl(second_path)
    assert len(second) == 164, f'Expected 164, got {len(second)}'
    solved_2 = sum(1 for x in second if x.get('is_solved', False))
    pass1_2 = solved_2 / 164 * 100
    print(f'Phase 2 (ParamAgent-plus): {solved_2}/164 = {pass1_2:.2f}%')
else:
    print('No second stage log (all problems solved in Phase 1)')

# ===== Cost =====
total_cost = sum(x.get('cost', 0) for x in (second if os.path.exists(second_path) else first))
total_prompt = sum(x.get('prompt_tokens', 0) for x in (second if os.path.exists(second_path) else first))
total_completion = sum(x.get('completion_tokens', 0) for x in (second if os.path.exists(second_path) else first))
print(f'Total cost: \${total_cost:.4f}')
print(f'Prompt tokens: {total_prompt:,}')
print(f'Completion tokens: {total_completion:,}')
"
```

---

## 验收指标

### 必须通过（Hard Pass）

| 编号 | 指标 | 标准 |
|------|------|------|
| H1 | Phase 1 完整性 | `first_stage_log.jsonl` 包含 164 条完整记录 |
| H2 | Phase 2 完整性 | `second_stage_log.jsonl` 包含 164 条记录（含 Phase 1 结果） |
| H3 | 字段完整 | 每条记录含 `is_solved`, `solution`, `cost`, `prompt_tokens`, `completion_tokens`, `diverse_reflections`, `implementations` |
| H4 | Memory bank | `mem_bank.pkl` 存在且可 pickle.load，包含 `positive_trajectories` 和 `negative_trajectories` |
| H5 | 无异常退出 | 程序正常执行完毕（允许单条 API 超时重试，但不允许 Python 未捕获异常导致中途退出） |

### 应该通过（Soft Pass — 允许合理偏差）

| 编号 | 指标 | 论文值 | 可接受范围 | 说明 |
|------|------|--------|-----------|------|
| S1 | Phase 1 Pass@1 | 82.93% | **75%–90%** | ParamAgent。因 API 版本/温度采样随机性，±8% 合理 |
| S2 | Phase 2 Pass@1 | ~87% | **80%–95%** | ParamAgent-plus。Phase 2 在 Phase 1 基础上应有提升 |
| S3 | Phase 2 > Phase 1 | — | Phase 2 Pass@1 ≥ Phase 1 Pass@1 | 第二阶段（加入 memory bank）不应比第一阶段差 |
| S4 | Phase 1 > 60% | — | Pass@1 > 60% | 至少显著超过 Base (59.15%)，否则说明流程有误 |
| S5 | Token 用量量级 | ~815K prompt | 400K–2M prompt tokens | 与论文 Table 8 的量级一致 |

### 偏差分析指引

如果结果偏离预期，按以下顺序排查：

1. **Pass@1 < 60%**: 流程有 bug，检查 pitfall 文件是否正确加载、测试执行是否正常
2. **Pass@1 在 60%–75%**: 可能 visible_tests 提取有问题导致中间反馈不准确，或 Together AI 的 Llama-3.1-8B 版本与论文使用的不同
3. **Phase 2 < Phase 1**: memory bank 检索或 augmented prompt 构建有问题
4. **运行中途 crash**: 检查 API key 配额、网络连接、单条题目的 edge case

---

## 附录 A：文件修改清单

| 操作 | 文件路径 | 步骤 |
|------|---------|------|
| 编辑 | `requirements.txt` | 1 |
| 新建 | `generators/__init__.py` | 2 |
| 新建 | `executors/__init__.py` | 3 |
| 编辑 | `executors/factory.py` | 4 |
| 编辑 | `generators/factory.py` | 5 |
| 编辑 | `main_param.py` 第 5 行 | 6 |
| 新建 | `scripts/gen_visible_tests.py` | 7 |
| 新建 | `benchmarks/humaneval_visible_tests.jsonl`（由脚本生成） | 7 |
| 编辑 | `generators/model.py` 第 21-22 行 | 8 |

## 附录 B：不要修改的文件

以下文件是论文核心逻辑，**不要修改**（除非发现运行时 bug）：

- `paramAgent.py` — 两阶段 ParamAgent 核心算法
- `generators/py_generate.py` — Python 代码生成器和 prompt 模板
- `generators/generator_utils.py` — 反思生成函数
- `memory_utils.py` — embedding 和检索
- `executors/py_executor.py` — Python 代码执行和评估
- `benchmarks/humaneval_full.jsonl` — 评测数据集
- `benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl` — 预计算 pitfall

## 附录 C：论文参考数据

来自 Table 1, HumanEval, Llama-3.1-8B backbone:

| Method | Pass@1 |
|--------|--------|
| Base | 59.15% |
| Reflexion | 76.22% |
| DoT | 73.17% |
| DoT-bank | 79.56% |
| **ParamAgent** | **82.93%** |

来自 Table 8, HumanEval, Llama-3.1-8B backbone:

| Method | Prompt Tokens | Completion Tokens | Cost ($) | Pass@1 |
|--------|-------------|-----------------|---------|--------|
| Base | 37,463 | 13,506 | 0.00917 | 59.15% |
| ParamAgent | 814,627 | 163,257 | 0.17602 | 82.93% |
