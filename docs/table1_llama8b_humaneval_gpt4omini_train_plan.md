# Table1 复现方案（Llama-3.1-8B + HumanEval，含 GPT-4o-mini 数据合成与 ParamMem 训练）

## 0. 结论先说（多专家并行评审后）
- **环境专家结论**：你当前 Phase1/Phase2 推理链路已经打通，下一步应补齐“数据合成 + LoRA训练”闭环。
- **算法专家结论**：要对齐 Table1 的 `ParamAgent=82.93`，重点是复现论文 Appendix B.2 的 `Mr` 训练流程，而不是继续改并发。
- **代码审查专家结论**：仓库中有 LoRA 推理脚本（`LoRA_Llama3_Code_multigpu_inference.py`），但**没有完整训练脚本**，需新增训练流水线。

---

## 1. 目标与对齐口径
- 目标：复现 **Table1 / HumanEval / Llama-3.1-8B / ParamAgent(Phase1)**。
- 论文对照值：`82.93% (136/164)`。
- 本方案不把 Table2（self-improvement）混进来，避免口径混淆。
- 说明：`ParamAgent-plus(phase2)` 不属于本次 Table1 主目标，仅作为后续可选扩展。

---

## 2. 论文与代码对齐点（必须锁定）
1. 论文 B.2：编程域 ParamMem 训练数据由两部分构成：
   - `APPs` 采样约 4000 题（intro-level）
   - `GPT-4o-mini` 合成约 4200 题
2. 对每题生成监督信号：**potential mistakes + buggy implementations**（论文 Figure 6 描述）。
3. 用这些监督做 LoRA 微调 `Llama-3.1-8B` 得到编程 ParamMem 模块 `Mr`。
4. 再用 `Mr` 对 HumanEval 164 题生成 pitfall（替代/重建 `humaneval_full_pitfalls.jsonl`）。
5. 运行 `main_param.py --strategy dot --phase1_only` 得到 Table1 对应的 ParamAgent 结果。

### 2.1 参数对齐清单（严格）
- **ParamMem训练数据规模**：`APPs 4000 + GPT-4o-mini 4200`（B.2）
- **ParamMem底座模型**：`Llama-3.1-8B-Instruct`（Implementation details）
- **LoRA超参**：`rank r=128`，`alpha=32`，`lr=2e-5`，`epochs=3`（Implementation details）
- **推理迭代数**：`max_iters=5`（Implementation details）
- **温度策略**：首轮 `T=0.2`，后续轮次 `T=1.0`（Implementation details）
- **评测集**：HumanEval `164` 题（Table 5）
- **指标**：`Pass@1`（Table 5）
- **embedding模型**：`text-embedding-3-small`（文中分析段）
- **pitfall来源口径**：由 `GPT-4o-mini` 生成监督数据后训练出的 `Llama-3.1-8B` ParamMem 模块来生成（B.2 + Figure 6）

> 说明：`dropout/max_length/weight_decay/batch_size` 等训练细节论文未明确给出，需在复现报告中标注为“实现假设”。

---

## 3. 当前仓库缺口
- 已有：
  - `LoRA_Llama3_Code_multigpu_inference.py`（LoRA 推理生成 pitfall）
  - ParamAgent 推理与评测链路（含并行版）
- 缺失：
  - **可直接复现论文 B.2 的 LoRA 训练脚本**
  - 训练数据构建脚本（APPs采样 + GPT-4o-mini合成 + 监督格式化）

---

## 4. 执行总流程（端到端）

### 阶段 A：数据合成（GPT-4o-mini）
1. 构建训练题集合：
   - `data/train/apps_4k.jsonl`
   - `data/train/synth_4k2.jsonl`
2. 用 GPT-4o-mini 为每题生成：
   - `pitfall`（核心错误模式）
   - `buggy_impls`（若干错误实现）
   - 输出 `data/train/parammem_code_supervision_8k2.jsonl`
3. 质检：字段完整率、空值率、重复率、长度分布、语法可解析率。

#### A.1 合成参数（论文未公开，需固定）
- 论文未明确：`temperature/top_p/max_tokens/seed`。
- 本次建议“基线默认值”（先固定再跑）：  
  - `temperature=0.7`  
  - `top_p=1.0`  
  - `max_tokens=1024`  
  - `seed=42`（若接口支持）
- 所有参数必须写入：`data/train/synthesis_meta.json`，确保可追溯。
- 为减少参数不确定性，建议加 1 组小样本对照（200 条）：  
  - A组：`temperature=0.7`  
  - B组：`temperature=1.0`  
  以 HumanEval phase1 的 early proxy（前 30 题）比较后再定全量。

### 阶段 B：LoRA 训练（ParamMem）
1. 新增训练脚本：`scripts/train/train_parammem_lora.py`
2. 论文对齐配置（必须）：
   - base: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - LoRA: `r=128, alpha=32`
   - epochs: `3`
   - lr: `2e-5`
3. 未公开细节（需固定并记录）：
   - `dropout`（建议先用 `0.05`）
   - `max_length`（建议 `2048`）
   - 全局 batch、grad_accum、weight_decay、warmup、scheduler
   - 随机种子（必须固定并写入日志）
4. 产物：`LoRA/code_r128/checkpoint-xxx/`

#### B.1 训练日志最小要求
- 记录：`global_step/train_loss/lr/epoch/seed/有效样本数`。
- 每个 epoch 保存 checkpoint；保留 best 与 last。
- 输出：`results/train_parammem_lora_<timestamp>/train_summary.json`。

### 阶段 C：LoRA 推理生成 HumanEval pitfall
1. 使用：`LoRA_Llama3_Code_multigpu_inference.py`
2. 输入：`benchmarks/humaneval_full.jsonl`
3. 输出：`benchmarks/code_pitfalls/humaneval_full_pitfalls_gpt4omini_lora.jsonl`
4. 校验：164 条、字段含 `pitfall` 与 `high_temp_pitfall`（若需要）。
5. 与仓库原文件做差异审计：
   - 覆盖率（164/164）
   - 平均长度
   - 空 pitfall 比例
   - `high_temp_pitfall` 的列表长度分布

### 阶段 D：Table1 目标评测（Phase1 only）
1. 用新 pitfall 文件运行：
   - `main_param.py --phase1_only --strategy dot --model llama3_1_8b --max_iters 5 --pass_at_k 1 --inner_iter 5`
2. 结果目录命名：`llama3_1_8b_humaneval_<timestamp>`
3. 输出：`first_stage_log.jsonl` + `results.txt`
4. 指标：`Pass@1 = solved/164`

---

## 5. 新增脚本清单（建议）
- `scripts/data/build_parammem_seedset.py`
  - 汇总 APPs 4k + 合成题 4k2，生成训练题清单
- `scripts/data/synthesize_gpt4omini_signals.py`
  - 调用 GPT-4o-mini 生成 mistakes/buggy implementations
  - 支持 24 key 轮询、断点续跑、失败重试
- `scripts/data/quality_report_parammem.py`
  - 数据质量报告（json + markdown）
- `scripts/train/train_parammem_lora.py`
  - QLoRA/SFT 训练入口
- `scripts/hpc4/run_table1_phase1_with_trained_pitfall.sh`
  - 统一执行 phase1 评测并写 `results.txt`

---

## 6. 并发与资源建议（HPC4/miku）
- 数据合成：可继续用你现有 24 key 并发策略（多worker+轮询）。
- 训练：独立 GPU 作业（不与 API 合成任务混跑）。
- 评测：沿用已验证的 24-worker phase1 并行（只跑 phase1）。
- 注意：并行加速不改变 Table1 口径，最终以合并后的 `first_stage_log.jsonl` 计算 Pass@1。

---

## 7. 风险与验证
1. **最大风险**：训练数据质量不足（不是并发问题）
   - 缓解：先小样本训练（1k）验证增益曲线，再扩到 8k2
2. LoRA 训练不稳定
   - 缓解：固定 seed + 保存每 epoch checkpoint + 早停
3. 结果波动
   - 缓解：同配置至少 3 次重复，报告均值/方差

---

## 8. 验收标准（针对 Table1）
- 硬条件：
  - `first_stage_log.jsonl` 164 条
  - `results.txt` 存在并记录 phase1 指标
- 软条件：
  - `Pass@1` 接近论文 82.93（建议目标带：`78%~86%`）
  - 三次运行标准差可接受（建议 `<2.5%`）
  - 训练与合成参数可完整回放（meta 文件与日志齐全）

---

## 9. 你下一步可以直接下发的执行指令
1. 先做数据链路：跑 `synthesize_parammem_code_dataset.py`（先 200 条小样本）。
2. 再做训练链路：实现并跑 `train_parammem_lora.py`（先 1 epoch 冒烟）。
3. 用 LoRA 产物生成 HumanEval pitfall 全量 164 条。
4. 跑 Table1 口径 phase1，并写入 `results.txt`。

### 9.1 已落地脚本与用法
- 数据合成主脚本：`scripts/data/synthesize_parammem_code_dataset.py`
- 质量报告脚本：`scripts/data/quality_report_parammem.py`
- HPC4 启动脚本：`scripts/hpc4/run_gpt4omini_synthesis_miku.sh`

启动命令（24 worker / 24 keys）：
```bash
./scripts/hpc4/run_gpt4omini_synthesis_miku.sh paramm table1_gpt4omini_data_<timestamp>
```

本地直跑（调试）：
```bash
python scripts/data/synthesize_parammem_code_dataset.py \
  --apps_samples 4000 \
  --synth_samples 4200 \
  --workers 24 \
  --model gpt-4o-mini \
  --temperature 0.7 \
  --max_tokens 1024 \
  --max_retries 4 \
  --seed 42 \
  --output_jsonl data/train/parammem_code_supervision_8200.jsonl \
  --work_dir data/train/parammem_build
```

质量检查：
```bash
python scripts/data/quality_report_parammem.py \
  --input data/train/parammem_code_supervision_8200.jsonl
```

---

## 10. 本次不做的事项（避免口径漂移）
- 不在本阶段把目标切到 Table2（self-improvement）。
- 不先改 ParamAgent 推理温度策略（先复现仓库实现，再做论文温度对照ablation）。
- 不把 phase2 分数当作 Table1 主对齐指标。
