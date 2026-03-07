# Read-Write Closed-Loop ParamMem ToDo List

## 研究目标

- 总目标：把当前 `ParamMem` 从 `offline train -> online read` 升级为 `online write explicit memory -> periodic consolidation -> online read updated parametric memory` 的闭环系统。
- 主验证任务：`Llama-3.1-8B-Instruct + HumanEval`。
- 最终目标：`ParamAgent-plus / Phase2 / Pass@1 > 89.0%`。

## 核心闭环

- `read memory`
- `route / select high-value trajectories`
- `verify / score`
- `write explicit memory buffer`
- `build consolidation dataset`
- `incremental LoRA consolidation`
- `A/B gate + publish / rollback`

## 计划总览

本 ToDo 不再按零散 Step 排列，而按 5 个研究里程碑推进：

- `Milestone 0`：固定基线与评测口径
- `Milestone 1`：完成 write-back 显式 memory 层
- `Milestone 2`：完成 consolidation 数据与增量训练链路
- `Milestone 3`：完成自动闭环与发布门控
- `Milestone 4`：完成最终研究验证并冲击目标指标

---

## Milestone 0：固定基线与评测口径

### 目标

建立一个后续所有闭环实验都必须复用的稳定对照组，避免因为评测口径变化导致结果不可比。

### 依赖

- 无。必须最先完成。

### 交付物

- 一份固定口径的基线记录文档
- 一组可重复对照 run
- 一份统一的实验命名与日志约定

### ToDo

1. 冻结当前 `main_param.py`、`paramAgent.py`、`scripts/hpc4/` 的基线提交状态。
2. 固定模型、pitfall 来源、visible tests、temperature、并发配置、merge 逻辑。
3. 记录当前 strongest baseline 的 run 路径、`results.txt`、`merge_summary.json`。
4. 统一实验命名规则：`模型_数据集_阶段_时间戳`。
5. 建立一份“禁止改口径项”清单，后续实验不得随意变更。
6. 复跑同一配置至少 2 次，记录波动范围。

### 验收标准

- 同口径重复 2 次，`Phase2 Pass@1` 波动 `<= ±0.8` 绝对点。
- 基线记录文档中必须明确写出：模型、数据、pitfall、phase1/phase2 指标、日志路径。
- 后续实验均能直接引用该基线，不需要重新解释口径。

---

## Milestone 1：完成在线写入显式 Memory Buffer

### 目标

把推理过程中产生的高价值经验，从“只存在日志里”升级成“结构化、可持续利用的显式 memory 层”。

### 依赖

- 必须完成 `Milestone 0`。
- 此阶段不要求重新训练。

### 交付物

- `memory_write_buffer` 数据格式
- 高价值轨迹导出脚本
- 质量评分与过滤脚本
- `clean_buffer.jsonl` 和质量报告

### ToDo

1. 设计 `memory_write_buffer` schema，至少包含：题目、初始失败代码、反思链、最终成功代码、测试反馈、检索上下文、cost、来源 run、lineage ID。
2. 在 `Phase2` 成功修复路径上新增自动写入逻辑。
3. 设定高价值样本筛选规则：首次修复成功、hard case 修复成功、search 相对 linear 有明显收益、多轮后成功等。
4. 增加唯一 ID、去重键、source run、adapter version 字段。
5. 实现 `export_memory_buffer.py`，从 run 日志稳定导出 buffer。
6. 实现 `score_memory_buffer.py`，计算质量分。
7. 实现 `filter_memory_buffer.py`，过滤模板泄漏、空样本、异常长度、重复样本、无效 patch。
8. 产出 `clean_buffer.jsonl` 与 `quality_report.json`。
9. 对样本做人工抽检，确认“写入的不是垃圾样本”。

### 验收标准

- 成功修复样本写入完整率 `= 100%`。
- `clean_buffer.jsonl` 无空关键字段、无解析错误、无模板污染。
- 去重后重复率 `< 5%`。
- 至少能从一个完整 run 中稳定导出 `>= 200` 条高质量轨迹。
- 人工抽检 50 条，质量通过率 `>= 90%`。
- `quality_score` 与最终修复收益的 Spearman 相关系数 `>= 0.3`。

---

## Milestone 2：完成 Consolidation 数据与增量训练链路

### 目标

把显式 write-back 经验转成真正可训练的 ParamMem 增量数据，并完成第一轮小步 consolidation 训练。

### 依赖

- 必须完成 `Milestone 1`。
- 这是本方向第一次必须重新训练的阶段。

### 交付物

- 版本化的 consolidation 数据集快照
- 增量 LoRA 训练模式
- `rw_parammem_v1` 与至少一个附加版本 checkpoint
- 训练元数据与 loss 曲线记录

### ToDo

1. 设计 consolidation 数据格式：输入题目状态，输出高价值 reflective signal。
2. 把 `clean_buffer` 转换为增量训练样本。
3. 支持三类数据混合：原始 GPT-4o-mini 合成数据、原始 LoRA 数据、write-back 高价值数据。
4. 加入 replay 机制，避免新数据覆盖旧分布。
5. 生成版本化数据快照并保存 lineage。
6. 在现有 LoRA 训练脚本上新增 `incremental_consolidation` 模式。
7. 支持 adapter 版本管理：`base -> rw_v1 -> rw_v2 ...`。
8. 运行第一轮 consolidation 训练，保存 loss、训练时长、checkpoint、数据快照编号。
9. 再运行至少一轮 follow-up consolidation，验证训练链路不是一次性跑通而已。

### 验收标准

- 数据集快照可完全追溯到源 buffer 和基线数据。
- write-back 样本占比可配置，默认控制在 `10% ~ 30%`。
- 不出现单一错误类型垄断训练集。
- consolidation 训练稳定完成，loss 无发散。
- 历史基线任务上的退化 `<= 0.5` 绝对点。
- 至少产出 `rw_parammem_v1` 和 `rw_parammem_v2` 两个可评测 adapter。

---

## Milestone 3：完成自动闭环与发布门控

### 目标

把 `write -> filter -> build dataset -> train -> A/B -> publish/rollback` 串成真正可循环运行的系统，而不是手工实验脚本。

### 依赖

- 必须完成 `Milestone 2`。

### 交付物

- 自动 A/B gate 脚本
- 自动发布/回滚逻辑
- 完整闭环调度脚本
- 完整 lineage 记录体系

### ToDo

1. 实现 `A/B gate`：每个新 adapter 自动跑固定 benchmark。
2. 评测至少覆盖 `HumanEval phase1/phase2`、`MBPP phase1/phase2`、成本与时延。
3. 定义发布规则：达到阈值才进入下一轮闭环。
4. 定义回滚规则：若指标退化或成本异常则自动回退。
5. 实现 `run_rw_parammem_loop.sh`，串起闭环全链路。
6. 为每一轮闭环生成 iteration 编号：`iter0/iter1/iter2/...`。
7. 记录完整 lineage：`run -> buffer -> clean_buffer -> dataset_snapshot -> adapter -> eval_result -> publish_decision`。
8. 增加恢复机制：中断后可从最近完成节点继续。
9. 连续运行至少 3 轮闭环，确认系统级稳定性。

### 验收标准

- 每次新 adapter 自动形成 A/B 报告。
- 发布门槛至少满足：
  - `HumanEval phase2` 不退化；
  - `MBPP phase2` 不显著退化（`<= 0.5` 绝对点）；
  - cost 增幅 `<= 20%`。
- 若不通过，系统自动回滚且回滚后仍可正常评测。
- 闭环调度可连续稳定执行 `>= 3` 轮。
- 任意一轮都能根据 lineage 回溯出数据、模型、结果来源。
- 中断恢复成功率 `= 100%`（在模拟中断测试中）。

---

## Milestone 4：完成最终研究验证并冲击目标指标

### 目标

证明 Read-Write Closed-Loop ParamMem 不只是工程闭环，而是能显著优于原始 read-only ParamMem，并达到目标指标。

### 依赖

- 必须完成 `Milestone 3`。

### 交付物

- `iter0/iter1/iter2/...` 全部评测结果
- `read-only vs read-write` 主表对照
- supporting analysis 图表和文字结论
- 最终研究报告/文档

### ToDo

1. 以 `Llama-3.1-8B-Instruct + HumanEval` 为主任务，固定口径跑 `iter0 -> iter1 -> iter2`。
2. 比较：原始静态 ParamMem、一次 consolidation 后、多轮 consolidation 后。
3. 汇总 phase1/phase2 指标、成本、样本量、adapter 版本、增量数据规模。
4. 形成 `read-only vs read-write` 的主结果表。
5. 做 supporting analysis，至少包括：
   - write-back 样本规模 vs 指标提升；
   - 质量分阈值 vs 指标；
   - replay 比例 vs 稳定性。
6. 复现最终最佳配置至少 2 次，验证不是偶然高点。
7. 对最终最佳 adapter 进行完整归档（checkpoint、数据快照、日志、评测结果）。

### 验收标准

- `iter1` 相对 `iter0`，`HumanEval phase2` 提升 `>= 1.5` 绝对点。
- `iter2` 相对 `iter1` 不退化，且最好继续提升 `>= 0.5` 绝对点。
- `read-write` 相对 `read-only`，`HumanEval phase2` 提升 `>= 3.0` 绝对点。
- 至少一个 supporting analysis 能明确解释增益来源，而不是只给最终分数。
- 最终最佳配置重复跑 2 次，`Phase2` 波动 `<= ±0.8`。

---

## 最终总验收目标

### 主指标

- `Llama-3.1-8B-Instruct + HumanEval + ParamAgent-plus (Phase2) Pass@1 > 89.0%`

### 次级指标

- `Phase1 Pass@1 >= 84.0%`
- 相比当前强基线，`Phase2` 绝对提升 `>= 6.0` 点。
- 闭环成功完成 `>= 3` 轮迭代，且后两轮无明显退化。

### 稳定性指标

- 同口径重复跑 2 次，`Phase2` 波动 `<= ±0.8` 绝对点。
- 历史 benchmark 无灾难退化，单数据集退化 `<= 1.0` 绝对点。

### 系统指标

- 全链路可复现：任意一次最终结果都能追溯到 `buffer / dataset / adapter / eval run`。
- 发布失败时可自动回滚到上一版可用 adapter。
- 调度中断后可恢复，不需要从头重来。

### 研究判断标准

只有当以下四点同时满足，才视为 `Read-Write Closed-Loop ParamMem` 成立：

1. 系统真实具备 write-back 与 consolidation 闭环，而不是只多了一个 buffer。
2. 新 adapter 的收益来自 write-back 高价值轨迹，而不是单纯训练噪声。
3. 提升是稳定、可复现、可解释的。
4. 最终 `HumanEval phase2 > 89.0%`。
