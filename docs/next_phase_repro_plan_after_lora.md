# Next Phase Repro Plan (after LoRA train)

## 0) 训练结果快速结论（审查专家）
- 训练已正常完成，无异常退出。
- 日志：`logs/Llama-3.1-8B-Instruct_parammem_train_merged_clean_20260302_200901.log`
- 关键指标：
  - `train_loss` 从约 `1.31` 下降到后段约 `0.46~0.48`
  - 最终汇总：`train_loss=0.6059`，`epoch=2.9938`，`train_steps=180`
- 结论：曲线整体单调下降并后期收敛，属于正常训练形态。

## 1) 阶段目标（方案专家）
复现 Table1 中 Llama3.1-8B 在 HumanEval 的 ParamAgent结果：
1. 用本次 LoRA 权重生成 HumanEval pitfalls（164题）
2. 跑 ParamAgent Phase1（只做 HumanEval）
3. 与已跑结果/论文指标做对齐报告

## 2) 已产物（代码专家）
- LoRA输出目录：
  - `/data/user/user06/data/paramagent/lora_runs/table1_llama31_8b_parammem_20260302_200901`
  - 关键文件：`adapter_model.safetensors`、`adapter_config.json`、`checkpoint-180`
- 训练数据：
  - `/data/user/user06/data/paramagent/table1_lora_train/parammem_train_merged_clean.jsonl`

## 3) 下一步执行清单
### Step A: 生成 HumanEval pitfalls
- 输入：`benchmarks/humaneval/humaneval_visible_tests.jsonl`
- 模型：`/data/user/user06/cache/Models/Llama-3.1-8B-Instruct`
- LoRA：`/data/user/user06/data/paramagent/lora_runs/table1_llama31_8b_parammem_20260302_200901`
- 产物建议：`benchmarks/code_pitfalls/humaneval_pitfalls_lora_20260302.jsonl`
- 验收：164/164 题有 pitfall，JSON 可解析。

### Step B: 跑 ParamAgent Phase1 (HumanEval)
- 使用你现有的 24 key 并发逻辑（保持与已验证版本一致）
- 仅 Phase1，不改 prompt 主逻辑
- 输出目录命名：`llama3_1_8b_humaneval_<timestamp>`
- 验收：
  - `results.txt` 自动生成
  - `phase1_acc` 可复核（164题）

### Step C: 复核与对齐报告
- 输出一份对齐报告：
  - 训练参数对齐项（r/alpha/lr/epoch）
  - 不可避免偏差（APPS有效量、服务端模型版本）
  - 新结果 vs 之前串行/并行结果

## 4) 风险与控制（审查专家）
- 风险1：NPU多进程偶发僵死/端口占用
  - 控制：每次训练使用独立 `MASTER_PORT`；启动前检查旧 `torchrun`
- 风险2：pitfall JSON 质量不稳
  - 控制：生成后做 schema + parse 全量校验
- 风险3：结果目录被 root 写入导致本地不可改
  - 控制：统一落到仓库 `logs/` 和用户有权限目录

## 5) 建议的立即动作
1. 先执行 Step A（LoRA pitfalls 生成）
2. 通过校验后立刻执行 Step B（Phase1）
3. 当天出 Step C 对齐报告

---

## 执行状态更新（2026-03-02）
- ✅ Step A 已完成（8 chip 并行）
  - 原始输出：`benchmarks/code_pitfalls/humaneval_full_pitfalls_lora_20260302_211516.jsonl`
  - 清洗输出：`benchmarks/code_pitfalls/humaneval_full_pitfalls_lora_20260302_211516_clean.jsonl`
  - 样本数：`164/164`
  - 字段校验：`task_id/pitfall/high_temp_pitfall` 全部齐全，空值 0
- ▶ 下一步建议直接进入 Step B（Phase1 HumanEval），mistake 文件使用清洗版路径。
