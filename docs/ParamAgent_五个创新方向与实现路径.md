# ParamAgent 五个创新方向与实现路径（基于代码+论文批判后的最终版）

## 基线与约束（统一前提）

- 基线代码：`main_param.py` + `paramAgent.py` 两阶段框架（Phase1 生成/反思，Phase2 失败题重试+memory）。
- 关键实现事实：
  - 检索目前是单点 `k=1`（`paramAgent.py:642`, `paramAgent.py:766`）。
  - 记忆库为 `positive_trajectories/negative_trajectories`（`paramAgent.py:155`）。
  - 迭代预算由 `max_iters/inner_iter` 控制（`main_param.py:38`, `main_param.py:40`）。
- 论文事实：
  - ParamAgent/ParamAgent-plus 两阶段定义（`docs/papers/ParamMem.txt:297-343`）。
  - 强调“多样性驱动”的反思收益，不是纯准确率优化（`docs/papers/ParamMem.txt:420-434`, `docs/papers/ParamMem.txt:1088-1093`）。
  - 固定 5 轮迭代、温度首轮 0.2 后续 1.0（`docs/papers/ParamMem.txt:410-415`）。
  - self-teaching 与 sample efficiency 是论文主叙事（`docs/papers/ParamMem.txt:1050-1060`, `docs/papers/ParamMem.txt:1102-1110`）。

---

## 方向1：D+ 多样性约束的验证驱动搜索（Phase2 范式重构）

### 这个方案在做什么
把当前 Phase2 的“线性反思迭代”改成“候选搜索 + 验证排序”：每题不再只沿一条链修复，而是在候选树里扩展多个修复分支，由 verifier 决定下一步。

### 端到端流程
1. 输入：Phase1 失败题 + 当前 memory 检索结果。  
2. 候选生成：每轮保留 `B` 个候选程序（beam）。  
3. 候选评估：每个候选跑可见测试 + 合成测试 + 复杂度/长度检查。  
4. 评分函数：`Score = pass_prob + λ1*reflection_diversity - λ2*complexity_penalty`。  
5. 搜索策略：Best-first/MCTS 扩展 top 候选，直到通过或预算耗尽。  
6. 输出：最佳候选写入 `second_stage_log`，并更新 memory bank。  

### 代码实现路径
- 新增 `search_engine.py`（候选池、扩展、剪枝、缓存）。
- 在 `paramAgent.py` Phase2 段替换线性循环为 `search_engine.run(...)`。
- `executors` 与现有 `gen.func_impl` 保持兼容，不改 API 层。

### 风险与控制
- 风险：token 成本上升。  
- 控制：引入 `max_nodes/max_depth/early_stop`，并保留线性模式作为 fallback。

### 详细分步 ToDo
1. 新建 `search_engine/` 模块骨架（state、node、queue、scorer、verifier_adapter）。
2. 在 `search_engine/config.py` 定义 `beam_size/max_nodes/max_depth/early_stop` 默认配置。
3. 抽离当前 Phase2 候选生成逻辑为 `candidate_generator` 适配层（不改生成 prompt 逻辑）。
4. 实现统一 verifier 接口：`run_visible_tests`、`run_generated_tests`、`run_static_checks`。
5. 实现候选缓存（同代码哈希不重复执行测试）。
6. 实现评分函数（通过率、多样性、复杂度惩罚）与可配置权重。
7. 在 `paramAgent.py` 增加 `--phase2_search_mode {linear,search}` 开关并默认保持 linear。
8. 完成最小 A/B：同一 run、同 pitfall 数据下对比 linear vs search（只改 Phase2）。
9. 输出新日志字段：`search_nodes_expanded`, `verifier_cost`, `best_score_trace`。
10. 加入失败回退：search 异常时自动切回原线性 Phase2。

### 验收标准（方向1）
- 功能验收：
  - `--phase2_search_mode search` 可稳定跑完 MBPP/HumanEval 全量，不中途崩溃。
  - search 模式日志完整包含 `search_nodes_expanded/verifier_cost/best_score_trace`。
- 指标验收：
  - 在同一 pitfall 数据、同一并发配置下，Phase2 Pass@1 相对 linear 提升 **≥ 1.5 绝对点**（至少一个数据集）。
  - 总 token 成本增长 **≤ 25%**；若超过需给出收益-成本曲线并通过审查。
- 回退验收：
  - 触发异常时可自动 fallback 到 linear，且结果文件完整产出。

---

## 方向2：C-lite 高频 RepairDSL（反思可执行化）

### 这个方案在做什么
把“自然语言反思”转换成“可执行修复动作”，避免反思文本含糊导致改写失控。

### 端到端流程
1. 从历史失败-成功轨迹抽取高频修复模式。  
2. 定义 15~20 个 DSL 原语（边界、索引、空输入、排序策略、状态重置等）。  
3. 模型先输出 DSL（可多条），再由编译器映射成代码 patch。  
4. patch 应用后立即执行测试；失败则回退并尝试下一条 DSL。  
5. 仅在 DSL 完全失败时才回退到原文本反思路径。  

### 代码实现路径
- 新增 `repair_dsl/schema.py`（语法与约束）。
- 新增 `repair_dsl/compiler.py`（DSL -> patch）。
- 在 `paramAgent.py` 反思更新点插入 `dsl_try -> fallback_text`。

### 风险与控制
- 风险：DSL 覆盖不全。  
- 控制：先做高频原语子集，逐步扩展；保留文本 fallback。

### 详细分步 ToDo
1. 从 `first_stage_log/second_stage_log` 抽取失败-修复样本，构建修复模式频次统计。
2. 定义 DSL v0 规范文档（15-20 原语，参数类型、约束、非法组合）。
3. 实现 `repair_dsl/schema.py`（语法校验、JSON schema 校验）。
4. 实现 `repair_dsl/compiler.py`（DSL 到代码 patch 的映射器）。
5. 实现 `repair_dsl/sandbox_apply.py`（临时应用 patch + 回滚）。
6. 在 `paramAgent.py` 插入执行链：`text_reflection -> dsl_parser -> compiler -> verify`。
7. 若 DSL 无法解析/执行失败，自动 fallback 到现有文本修复路径。
8. 记录可审计日志：`dsl_program`, `patch_diff`, `apply_result`。
9. 做 DSL 原语消融：逐类禁用，观察对 MBPP/HumanEval 的影响。
10. 固化“高风险原语黑名单”，防止破坏性 patch。

### 验收标准（方向2）
- 功能验收：
  - DSL 解析成功率 **≥ 90%**（在进入 DSL 路径的样本上）。
  - patch 应用与回滚流程正确率 **= 100%**（无脏写、无残留状态）。
- 指标验收：
  - 相比纯文本反思，Phase2 Pass@1 提升 **≥ 1.0 绝对点** 或等效成本下降 **≥ 10%**。
  - “同题重复试错轮次”中位数下降 **≥ 20%**。
- 质量验收：
  - 日志可追溯（每题可定位 `dsl_program` 与 `patch_diff`）。
  - 高风险原语黑名单命中后无破坏性提交。

---

## 方向3：B-lite 三记忆监督路由器（不是 RL 起步）

### 这个方案在做什么
让系统学习“当前题在当前状态下该用哪类记忆”，替代固定手工拼接顺序。

### 端到端流程
1. 用现有 logs 回放：构建样本 `(state, memory_mix, outcome)`。  
2. 训练轻量路由器：输入题目状态/失败类型/历史反馈，输出三记忆权重。  
3. 在线推理时先由路由器出权重，再执行检索与生成。  
4. 低置信度时回退固定策略，确保稳定性。  

### 代码实现路径
- 新增 `memory_router/train_router.py`、`memory_router/infer_router.py`。
- 在 `paramAgent.py` 检索前增加 `mix = router.predict(state)`。
- 记忆库结构不必一次性大改，先在读取层做加权融合。

### 风险与控制
- 风险：路由器过拟合单数据集。  
- 控制：跨 HumanEval/MBPP 分层验证 + 置信度门控回退。

### 详细分步 ToDo
1. 定义 `state` 特征：题目长度、历史失败类型、当前反思轮次、检索相似度统计。
2. 从历史日志构造监督样本：`state -> 最优 memory_mix -> outcome`。
3. 新建 `memory_router/dataset_builder.py` 生成训练集与验证集。
4. 实现轻量路由器模型（MLP 起步），输出三记忆权重和置信度。
5. 训练并保存 `router.ckpt`，记录跨数据集泛化指标。
6. 在 `paramAgent.py` 检索前读取 router 输出，执行加权融合检索。
7. 增加 `--router_enable` 与 `--router_conf_threshold` 参数。
8. 低置信样本自动回退固定策略，确保稳定。
9. 增加路由可解释日志：`router_mix`, `router_conf`, `fallback_flag`。
10. 做跨数据集评估（train on HumanEval, test on MBPP 反向也测）。

### 验收标准（方向3）
- 功能验收：
  - router 可在在线推理阶段稳定给出三记忆权重与置信度。
  - 低置信样本 fallback 比例可配置且生效。
- 指标验收：
  - 相对固定策略，Phase2 Pass@1 提升 **≥ 1.0 绝对点**（至少一个数据集）。
  - 不得出现两个主数据集同时退化（允许单数据集轻微波动 ≤0.5）。
- 泛化验收：
  - 跨数据集测试时，router 方案相对固定策略退化不超过 **1.0 绝对点**。

---

## 方向4：E-lite 自举闭环训练（在线高价值轨迹回灌 ParamMem）

### 这个方案在做什么
把推理阶段得到的高价值修复轨迹，持续变成下一轮 ParamMem 的训练数据，形成自增强循环。

### 端到端流程
1. 每次 run 后自动抽取高价值样本（难题、首次修复成功、跨轮修复成功）。  
2. 生成增量训练集（含质量标签与去重信息）。  
3. 执行小步 LoRA 训练（固定预算/固定 epoch）。  
4. A/B 评测通过阈值才切换新 adapter；不通过自动回滚。  

### 代码实现路径
- 新增 `scripts/data/export_bootstrap_tuples.py`。
- 复用现有 LoRA 训练脚本，增加“增量训练模式”。
- 新增 `scripts/hpc4/ab_gate_publish.sh` 做自动门控发布。

### 风险与控制
- 风险：灾难遗忘、数据漂移。  
- 控制：回放集混采 + 冻结策略 + 强制回归测试。

### 详细分步 ToDo
1. 新建 `scripts/data/export_bootstrap_tuples.py`，抽取高价值轨迹（含质量分）。
2. 定义增量训练数据格式（prompt、pitfall、quality、source_run、difficulty）。
3. 构建去重与污染检测（同题重复、模板泄漏、异常长度过滤）。
4. 复用 LoRA 训练脚本，新增 `--incremental_train` 模式。
5. 增加 replay 机制（旧高质量样本按比例混入）。
6. 训练后自动触发 A/B 评测（固定 benchmark + 固定随机种子设置）。
7. 定义发布门槛：至少一个主数据集提升且无主指标退化超阈值。
8. 若未达门槛，自动回滚到上一版 adapter。
9. 记录完整 lineage：`data_snapshot -> checkpoint -> eval_result -> publish_decision`。
10. 周期化调度（每 N 次 run 或每天固定窗口执行）。

### 验收标准（方向4）
- 功能验收：
  - 可自动完成 `导出样本 -> 增量训练 -> A/B -> 发布/回滚` 全链路。
  - lineage 元数据完整、可追溯到具体 run 与 checkpoint。
- 指标验收：
  - 连续 3 轮自举中，至少 2 轮带来正增益，且无单轮主指标退化超过 **1.0**。
  - 新模型发布门槛：至少一个主数据集提升 **≥ 1.0 绝对点** 且另一个不显著退化（≤0.5）。
- 稳定性验收：
  - 遗忘检测通过（历史基线集性能下降 ≤ 0.5）。

---

## 方向5：A-lite 半结构化图记忆（错误图+修复图）

### 这个方案在做什么
不直接上完整程序图状态机，而是先把错误与修复关系图结构化，作为检索与路由输入增强。

### 端到端流程
1. 把失败样本映射到错误 taxonomy（边界/类型/状态/复杂度等）。  
2. 建立“错误节点 -> 修复动作节点”的有向图。  
3. 检索时先按图邻域召回候选轨迹，再做语义重排。  
4. 召回结果进入 Phase2 作为条件信息。  

### 代码实现路径
- 新增 `memory_graph/build_graph.py` 与 `memory_graph/retrieve.py`。
- 在 `paramAgent.py` 当前检索前增加图召回结果拼接。
- 不破坏原 memory bank 文件格式，图索引单独存储。

### 风险与控制
- 风险：错误分类噪声传播。  
- 控制：分类器置信度阈值 + 弱监督纠错队列。

### 详细分步 ToDo
1. 定义错误 taxonomy v0（边界、类型、状态、复杂度、语义偏差等）。
2. 新建 `memory_graph/build_graph.py` 从历史日志构图（节点、边、权重）。
3. 新建 `memory_graph/retrieve.py`，支持“图邻域召回 + 语义重排”。
4. 将图索引存储为独立文件（不破坏现有 `mem_bank.pkl`）。
5. 在 `paramAgent.py` 增加图检索开关 `--graph_memory_enable`。
6. 在线流程中拼接图召回结果到 Phase2 检索候选池。
7. 记录图检索日志：`error_type`, `neighbor_hits`, `graph_weight`。
8. 实现弱监督纠错队列（低置信分类样本进入人工/规则复核）。
9. 与纯语义检索做对照实验，评估 hard case 提升。
10. 周期性重建图索引，防止陈旧拓扑影响检索。

### 验收标准（方向5）
- 功能验收：
  - 图索引构建成功率 **= 100%**（在有效日志输入下）。
  - 图检索可与现有语义检索并行工作，不破坏原流程。
- 指标验收：
  - hard case 子集（Phase1 失败题）上，召回命中率提升 **≥ 10%**（对比纯语义检索）。
  - 最终 Phase2 Pass@1 提升 **≥ 0.8 绝对点** 或等效成本下降 **≥ 8%**。
- 质量验收：
  - 错误分类低置信样本进入纠错队列比例可控（建议 5%~15%）。
  - 图索引定期重建后结果波动在可接受范围（±0.5 绝对点）。

---

## 五方向对比与优先级（批判后结论）

| 方向 | 可实现性 | 现实性 | 新颖性 | 收益性 | 建议优先级 |
|---|---:|---:|---:|---:|---:|
| 方向1 D+ 搜索 | 4 | 4 | 3 | 5 | 1 |
| 方向3 B-lite 路由 | 4 | 4 | 4 | 4 | 2 |
| 方向2 C-lite DSL | 3 | 4 | 4 | 4 | 3 |
| 方向4 E-lite 闭环 | 3 | 4 | 3 | 5 | 4 |
| 方向5 A-lite 图记忆 | 3 | 3 | 4 | 3 | 5 |

---

## 实施顺序（研究版，不是工程小修）

### 阶段 I（3-4 周）：拿到可发表增益
- 主线：方向1 + 方向3  
- 目标：在 MBPP/HumanEval 的 Phase2 取得稳定绝对提升，并报告成本变化。

### 阶段 II（3-4 周）：机制可解释
- 主线：方向2  
- 目标：把反思机制从“文本黑箱”变成“可执行修复”，产出错误类型-修复动作分析。

### 阶段 III（4 周）：持续学习闭环
- 主线：方向4 + 方向5（轻量接入）  
- 目标：验证 self-teaching 与 sample-efficiency 叙事，形成完整论文故事线。
