# ParamMem 复现分析报告

## 论文概述

**ParamMem: Augmenting Language Agents with Parametric Reflective Memory**
(arXiv:2602.23320v1, Feb 2026)

提出参数化记忆模块 (Parametric Memory)，通过 LoRA 微调 Llama-3.1-8B 编码跨样本反思模式，推理时通过温度采样生成多样化反思信号，增强 agent 迭代推理能力。

### 核心流程 (Algorithm 1)

```
Phase 1 (ParamAgent):
  对每个任务 x:
    1. 从 ParamMem (LoRA模型) 采样 rg (T=0.2 第一次, T=1.0 后续)
    2. Actor 基于 (x, 历史反思 r1:k-1, rg) 生成解
    3. 评估 → 成功则存入 memory bank B，失败则生成自反思
    4. 重复 max_iters 次

Phase 2 (ParamAgent-plus):
  对 Phase 1 中失败的任务:
    1. 从 B 中检索最相似的成功轨迹 τ
    2. Actor 基于 (x, r1:k-1, rg, τ) 生成解 (三种记忆联合)
```

### 主实验覆盖范围

| 领域 | 数据集 | 规模 | 评估指标 |
|------|--------|------|---------|
| Code | HumanEval | 164 problems | Pass@1 |
| Code | MBPP | 397 problems | Pass@1 |
| Math | MATH | 278 sampled | 0-1 Accuracy |
| QA | HotpotQA | 300 sampled | 0-1 Accuracy |
| QA | 2WikiMultiHopQA | 300 sampled | 0-1 Accuracy |

### Backbone LLMs

- Llama-3.1-8B, Mistral-7B-v0.2, Qwen2-1.5B
- 附录含 70B 级别 (Llama-3.1-70B, Qwen2.5-72B)

### 关键超参

- 迭代次数: 5
- LoRA: r=128, α=32, lr=2e-5, 3 epochs
- 温度: T=0.2 (首轮), T=1.0 (后续)
- ParamMem 默认基于 Llama-3.1-8B-Instruct

---

## 代码完整性分析

### 已有组件

| 组件 | 文件 | 说明 |
|------|------|------|
| ParamAgent 核心 | `paramAgent.py` | 两阶段算法，Code generation |
| 基线策略 | `dot.py`, `reflexion.py`, `simple.py`, `dot_bank.py` | Reflexion/DoT/DoT-bank |
| Python 生成器 | `generators/py_generate.py` | 含 parametric 版本 prompt |
| LLM 后端 | `generators/model.py` | OpenAI/TogetherAI/DashScope/Anthropic |
| 代码执行器 | `executors/py_executor.py` | exec + subprocess 评估 |
| Embedding 检索 | `memory_utils.py` | OpenAI embedding + cosine similarity |
| LoRA 推理 | `LoRA_Llama3_Code_multigpu_inference.py` | 多卡 ParamMem 推理 |
| HumanEval 数据 | `benchmarks/humaneval_full.jsonl` | 164 题完整 |
| MBPP 数据 | `benchmarks/mbpp-py.jsonl` | 397 题完整 |
| HumanEval Pitfall | `benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl` | 164 条，含 pitfall + high_temp_pitfall(8条) |

### 缺失文件

#### 高优先级（阻塞 Code 实验运行）

| 缺失文件 | 引用位置 | 影响 |
|----------|---------|------|
| `generators/__init__.py` | 全局 import | 无法 `from generators import ...` (有 pyc 缓存可能临时可用) |
| `executors/__init__.py` | 全局 import | 同上 |
| `benchmarks/humaneval_visible_tests.jsonl` | `main_param.py:138` | HumanEval 运行时 crash |
| `generators/factory.py` 中 import 的缺失模块 | `factory.py:2-6` | `rs_generate`, `game24_generate`, `MiltuhopQA_generate`, `MathQA_generate` 全部缺失，import 直接报错 |

#### 中优先级（影响部分功能）

| 缺失文件 | 引用位置 | 影响 |
|----------|---------|------|
| `reflexion_parametric.py` | `main_param.py:5` | Reflexion + ParamMem 变体（主实验用 dot strategy 可绕过） |
| `LoRA_Llama3_Code_Inference.py` | `main_param.py:16` | 单卡 LoRA 推理（有 try/except 降级，用预计算 pitfall 替代） |
| `generators/livecodebench_utils.py` | `paramAgent.py` 等 | LiveCodeBench 支持（非主实验） |
| `benchmarks/code_pitfalls/mbpp_pitfalls.jsonl` | 运行 MBPP 实验 | MBPP pitfall 数据缺失 |

#### 低优先级（不影响 Python Code 实验）

| 缺失文件 | 说明 |
|----------|------|
| `generators/rs_generate.py` | Rust 生成器 |
| `generators/game24_generate.py` | Game24 生成器 |

### 缺失数据/模型

| 缺失项 | 说明 | 获取方式 |
|--------|------|---------|
| ParamMem LoRA 权重 | Code 领域 adapter | HuggingFace: `TianJun1/lora-llama3-8b-code` |
| LoRA 微调训练数据 | 4000 APPS + 4200 合成题 pitfall | 需自行构建或找作者 |
| LoRA 微调训练代码 | fine-tune 脚本 | 完全缺失，只有推理脚本 |
| MATH 数据集 | 278 sampled problems | 需从 MATH 原始数据采样 |
| HotpotQA / 2WikiMQA | 各 300 sampled | 需从原始数据集采样 |
| Math/QA ParamMem LoRA | 各领域独立 parametric module | 完全缺失 |
| Math/QA 生成器代码 | `MathQA_generate.py`, `MiltuhopQA_generate.py` | 完全缺失 |

---

## 复现路径

### 路径 A：最小复现 (仅 HumanEval)

所需步骤：

1. **修复 import 链**
   - 创建 `generators/__init__.py` 和 `executors/__init__.py`
   - 修改 `generators/factory.py` 注释掉缺失模块 import（rs_generate, game24, MathQA, MultiHopQA）

2. **补充 `humaneval_visible_tests.jsonl`**
   - 从 HumanEval 原始数据抽取 docstring 中的测试用例
   - 或从 DoT 原仓库获取

3. **配置 API keys**
   - `TOGETHER_API_KEY`: Llama-3.1-8B 推理 (via Together AI)
   - `OPENAI_API_KEY`: text-embedding-3-small (embedding 检索)

4. **运行**
   ```bash
   python main_param.py \
       --run_name "paramAgent_humaneval" \
       --root_dir ./results/humaneval/paramAgent/ \
       --dataset_path benchmarks/humaneval_full.jsonl \
       --strategy "dot" \
       --language "py" \
       --model "llama3_1_8b" \
       --pass_at_k 1 --max_iters 5 \
       --use_mistakes \
       --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
       --verbose
   ```

5. **预期结果**: first_stage_log = ParamAgent, second_stage_log = ParamAgent-plus

### 路径 B：Code 领域完整复现 (HumanEval + MBPP)

在路径 A 基础上额外需要：

6. **生成 MBPP pitfall 文件**
   - 下载 LoRA 权重: `TianJun1/lora-llama3-8b-code`
   - 运行 `LoRA_Llama3_Code_multigpu_inference.py` 对 MBPP 生成 pitfall
   - 或用 GPT-4o-mini API 直接生成

### 路径 C：全领域完整复现

在路径 B 基础上额外需要：

7. **补全 Math/QA 代码** — 编写或向作者索取 `MathQA_generate.py`, `MiltuhopQA_generate.py`
8. **构建 Math/QA 微调数据** — 按 Appendix B.2 用 GPT-4o-mini 生成
9. **编写 LoRA 微调脚本** — 标准 HuggingFace PEFT LoRA fine-tuning
10. **准备评估数据集** — MATH 采样 278 题, HotpotQA/2WikiMQA 各采样 300 题
11. **训练 Math/QA ParamMem** — 各训练一个独立 LoRA adapter

---

## 结论

当前代码仓库仅能支撑 **HumanEval Code 实验**（修复若干 import 问题后）。Math 和 QA 实验的生成器代码和数据均完全缺失。建议：

1. 优先走路径 A 复现 HumanEval 结果验证流程正确性
2. 联系论文作者获取完整代码仓库（特别是 Math/QA 部分）
3. LoRA 微调训练代码需自行编写（论文已给出足够超参信息）
