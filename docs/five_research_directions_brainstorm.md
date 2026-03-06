# ParamAgent 研究总方案（终稿）
# Final Comprehensive Research Agenda for ParamAgent

> 基于三位专家意见整合：优先执行 **#2 Surprisal-Bandit + #4 Plan-First PDL**，稳定后接入 **#1 Causal Event Memory**，并将 **#5 / #3** 作为二期高风险高回报方向。

---

## 1) Executive Summary（执行摘要）

本方案面向 ParamAgent 的下一阶段研究与工程落地，目标是在 **12 周内形成"可发表 + 可部署"的双闭环成果**。

核心策略是分两阶段推进：

- **Phase A（前6周）**：先做高可行性、可控性强的方向
  - 方向2：Surprisal-Bandit 测试时算力调度（最高 publishability）
  - 方向4：Plan-First PDL（最高 implementability）
- **Phase B（后6周）**：在系统稳定后做记忆增强与高风险探索
  - 方向1：Causal Event Memory（高新颖度）
  - 方向5：Disagreement-Driven Dual-Memory（需强消融）
  - 方向3：VB-LoRA Specialist Swarm（路由稳定性风险最高）

预期收益（综合专家区间）：
- 主线组合（#2 + #4）预计 **+6 至 +13 分**（MBPP Pass@1）
- 加入 #1 后整体有望达到 **+9 至 +20 分**（分阶段累计，非简单线性相加）
- #5 / #3 作为发表扩展线，重点贡献在机制创新与泛化鲁棒性。

---

## 2) Five Research Directions（五大研究方向）

### Direction 1: 因果事件记忆与反事实回放
### Causal Event Memory + Counterfactual Replay

**One-line narrative**  
将"发生了什么"升级为"为什么发生 + 若改决策会怎样"，把经验缓存转为因果可学习资产。  
*Turn episodic traces into causal assets: not just what happened, but why and what-if.*

**核心问题 / Core Problem**  
当前 ParamAgent 的 memory bank 使用 embedding-based 检索，仅能找到"表面相似"的轨迹，无法理解"哪个动作导致了成功/失败"。这导致：
- 检索到的轨迹可能包含无关操作
- 无法回答"如果当时选择了另一个工具会怎样"
- 跨任务迁移能力弱（相似问题但不同表述时检索失效）

**Detailed Method**  
1. **事件图抽取**：将每个迭代记录为 `(state, action, tool_result, reward, latency, failure_type)`
2. **因果边估计**：基于干预近似（do-like perturbation）与历史对照构建 causal score
3. **反事实回放**：对失败轨迹替换关键动作，重放评分，更新策略偏好
4. **记忆检索**：优先召回"高因果影响、同任务簇"的事件子图

**Expected Gains with Evidence**  
- 专家区间：**+3 ~ +7 MBPP points**，可行性 **7/10**，新颖性高
- 证据逻辑：相比纯向量记忆，因果记忆更利于错误纠正与跨任务迁移
- 支持文献：Event-sourced memory 架构在长期任务中降低成本 60%

**Implementation Phases (0-3)**  
- **Phase 0**: 定义事件schema、因果打分协议、离线回放评测集
- **Phase 1**: 实装事件日志 + 因果边估计 + 最小反事实回放
- **Phase 2**: 在线检索融合到 planner/scheduler
- **Phase 3**: 大规模消融（无因果/无反事实/随机回放）

**Phase 1 File-Level Changes**  
```
paramagent/memory/event_store.py          # 事件持久化
paramagent/memory/causal_graph.py         # 因果边构建
paramagent/training/counterfactual_replay.py  # 回放训练器
paramagent/schemas/event_schema.py        # 统一schema
configs/memory/causal_memory.yaml         # 开关与阈值
```

**Risk + Rollback Strategy**  
- 风险：伪因果、算力开销、错误强化
- 回滚：保留 `memory_mode=episodic`，反事实仅离线启用，设置因果置信阈值门控

**Publishability Analysis**  
- 强：机制新颖（agentic causal memory），可投 agent learning / decision-making track
- 适合会议：NeurIPS (Agent track), ICLR (RL/Planning), AAAI

---

### Direction 2: 惊讶度-多臂老虎机测试时算力调度
### Surprisal-Bandit Test-Time Compute Scheduler

**One-line narrative**  
把"每步都重算"改成"只在高不确定时花算力"。  
*Spend compute only when uncertainty/surprisal is high.*

**核心问题 / Core Problem**  
当前 ParamAgent 对所有问题使用固定 5 次迭代，导致：
- 简单问题浪费算力（1-2 次就能解决）
- 困难问题算力不足（5 次仍未解决）
- 无法根据实时反馈动态调整策略

**Detailed Method**  
1. **计算 surprisal 信号**：计划分支熵、工具失败先验、历史回报方差
2. **Bandit 动态分配预算**（arms = 推理深度/采样数/自检次数）
3. **奖励函数**：`task_success - λ*latency - μ*token_cost`
4. **上下文化 bandit**：按任务类型、难度、实时错误率自适应

**Expected Gains with Evidence**  
- 专家区间：**+4 ~ +9 MBPP points**，可行性 **8/10**，publishability 最高
- 证据逻辑：可直接优化成功率-时延-成本 Pareto 前沿，工程收益立竿见影
- 支持文献：
  - Surprisal-guided selection: 80% vs 50% success (Feb 2026 arXiv)
  - Horizontal scaling (8 parallel cheap runs) > single expensive run (AI21 Maestro)

**Implementation Phases (0-3)**  
- **Phase 0**: 定义预算动作空间与离线重放日志
- **Phase 1**: 上线 rule-based surprisal + UCB/Thompson baseline
- **Phase 2**: contextual bandit + online learning
- **Phase 3**: 多任务泛化与鲁棒性分析

**Phase 1 File-Level Changes**  
```
paramagent/scheduler/surprisal.py         # 惊讶度特征
paramagent/scheduler/bandit.py            # UCB/TS策略
paramagent/runtime/inference_budget.py    # 预算执行器
paramagent/eval/cost_latency_success.py   # 三目标评测
configs/scheduler/surprisal_bandit.yaml
```

**Risk + Rollback Strategy**  
- 风险：探索过度导致短期性能波动
- 回滚：固定预算fallback，bandit仅影子模式（shadow mode）先跑

**Publishability Analysis**  
- 很强：方法简洁、可复现、对工业系统价值高，最适合先出paper
- 适合会议：ICML (Systems track), NeurIPS (Optimization), MLSys

---

### Direction 3: 变分贝叶斯 LoRA 专家群
### VB-LoRA Specialist Swarm (Composable ParamMem)

**One-line narrative**  
让多个轻量专家在不确定性驱动下协同，而不是单一策略硬扛。  
*Use uncertainty-aware specialist LoRA swarm instead of one monolithic policy.*

**核心问题 / Core Problem**  
当前 ParamAgent 使用单一 LoRA adapter (r=128) 生成所有类型的反思，导致：
- 参数容量有限，难以同时处理语法修复、算法重写、测试修复等多种模式
- 部署成本高（128 rank LoRA 仍需较大显存）
- 无法针对不同错误类型使用专门策略

**Detailed Method**  
1. **多专家 LoRA**：规划、工具调用、纠错、压缩回答等专长头
2. **VB 后验近似**：输出不确定性用于路由与拒答
3. **路由器学习**：任务embedding + 当前状态 + 历史成功率
4. **聚合机制**：加权投票或稀疏门控（top-k experts）

**Expected Gains with Evidence**  
- 专家区间：**+3 ~ +8 points**，可行性 **4/10**（路由不稳定）
- 证据逻辑：理论上能提升复杂场景上限，但训练与路由稳定性是瓶颈
- 支持文献：VB-LoRA achieves 99.6% parameter reduction vs standard LoRA (NeurIPS 2024)

**Implementation Phases (0-3)**  
- **Phase 0**: 单专家LoRA基线与路由oracle上界
- **Phase 1**: 2-3专家最小系统 + 静态路由
- **Phase 2**: VB不确定性 + 动态路由
- **Phase 3**: 专家蒸馏回单模型（降低部署成本）

**Phase 1 File-Level Changes**  
```
paramagent/models/lora_experts.py
paramagent/router/expert_router.py
paramagent/training/expert_curriculum.py
paramagent/eval/router_stability.py
configs/models/vb_lora_swarm.yaml
```

**Risk + Rollback Strategy**  
- 风险：路由震荡、训练成本高、线上复杂度过高
- 回滚：只保留最佳单专家LoRA；路由改静态规则

**Publishability Analysis**  
- 中等偏强：若证明"高不确定任务收益显著 + 成本可控"则有亮点
- 适合会议：ICLR (PEFT track), EMNLP (Efficient NLP)

---

### Direction 4: 计划优先的程序合成 / PDL
### Plan-First Program Synthesis (PDL Compiler Path)

**One-line narrative**  
先产出可验证计划，再执行；把 agent 行为从"即兴"变为"可编译"。  
*Compile a verifiable plan first, then execute with guardrails.*

**核心问题 / Core Problem**  
当前 ParamAgent 直接生成代码，缺乏中间规划层，导致：
- 长链推理容易漂移（第3-5次迭代质量下降）
- 难以调试（无法知道agent"打算做什么"）
- 工具误用风险高（直接生成可能调用不存在的API）

**Detailed Method**  
1. **引入 PDL（Plan Description Language）中间层**
2. **流程**：任务 → PDL草案 → 静态检查 → 执行图
3. **约束检查**：依赖完整性、工具权限、循环上限、失败重试策略
4. **可解释性**：每一步可追溯到 plan node 与执行日志

**Expected Gains with Evidence**  
- 专家区间：**+2 ~ +6 points**，可行性 **9/10**，实现最稳
- 证据逻辑：结构化计划显著降低工具误用与长链路漂移
- 支持文献：AI21 Maestro PDL approach shows higher improvement rates when scaled horizontally

**Implementation Phases (0-3)**  
- **Phase 0**: 定义 PDL grammar + validator
- **Phase 1**: planner 输出 PDL，executor 支持最小节点集
- **Phase 2**: 计划修复（plan repair）与自检
- **Phase 3**: 与 scheduler 联动（PDL复杂度→预算）

**Phase 1 File-Level Changes**  
```
paramagent/planner/pdl_ast.py
paramagent/planner/pdl_parser.py
paramagent/planner/pdl_validator.py
paramagent/executor/pdl_runner.py
configs/planner/plan_first_pdl.yaml
```

**Risk + Rollback Strategy**  
- 风险：前期开发慢、语法过严导致召回下降
- 回滚：保留原 planner 路径，PDL 以可选模式逐步放量

**Publishability Analysis**  
- 强：工程可实现性高，适合 systems + agents 结合论文
- 适合会议：ICSE (Software Engineering), FSE, PLDI (Programming Languages)

---

### Direction 5: 分歧驱动的双记忆训练
### Disagreement-Driven Dual-Memory Training

**One-line narrative**  
让"短期情景记忆 vs 长期语义记忆"的分歧成为训练信号，而不是噪声。  
*Use disagreement between episodic and semantic memory as a supervision signal.*

**核心问题 / Core Problem**  
当前 ParamAgent Phase 1 使用 ParamMem（语义记忆），Phase 2 使用 memory bank（情景记忆），但两者独立运作：
- 当两个记忆给出冲突建议时，系统无法学习
- 无法识别哪种记忆在何种情况下更可靠
- 长期一致性差（相似问题在不同时间给出不同答案）

**Detailed Method**  
1. **双记忆**：episodic（近期轨迹）+ semantic（压缩知识）
2. **分歧检测**：结论冲突、工具选择冲突、置信度冲突
3. **训练策略**：高分歧样本加权、触发反思/重规划
4. **目标**：减少记忆污染，提高长期一致性

**Expected Gains with Evidence**  
- 专家区间：**+3 ~ +7 points**，可行性 **5/10**（需强消融）
- 证据逻辑：在分布漂移和长期任务中理论收益明显
- 支持文献：Three-layer memory systems reduce costs 60% for long-horizon tasks

**Implementation Phases (0-3)**  
- **Phase 0**: 定义分歧度量与标注协议
- **Phase 1**: 实现 dual-memory reader + disagreement scorer
- **Phase 2**: 训练时重加权与在线反思策略
- **Phase 3**: 长周期一致性基准验证

**Phase 1 File-Level Changes**  
```
paramagent/memory/episodic_memory.py
paramagent/memory/semantic_memory.py
paramagent/training/disagreement_sampler.py
paramagent/reasoning/reflection_trigger.py
configs/training/disagreement_dualmemory.yaml
```

**Risk + Rollback Strategy**  
- 风险：分歧指标不稳、训练信号噪声大
- 回滚：分歧仅用于分析，不进训练损失；保留单记忆主干

**Publishability Analysis**  
- 中等：若有长期一致性指标提升和扎实消融，可成为亮点副线
- 适合会议：ACL (Memory/Reasoning), EMNLP, CoRL (Continual Learning)

---

## 3) Recommended Execution Order（12周执行路线图）

| Week | Focus | Deliverable |
|---|---|---|
| 1-2 | #4 Plan-First PDL Phase 0-1 | PDL grammar + parser + validator + minimal runner |
| 3-4 | #2 Surprisal-Bandit Phase 0-1 | surprisal features + UCB/TS scheduler + shadow evaluation |
| 5-6 | #2 + #4 Integration | PDL complexity-driven budget allocation; first internal report |
| 7-8 | #1 Causal Memory Phase 0-1 | event schema + causal graph v1 + offline counterfactual replay |
| 9 | #1 + 主线融合 | memory-aware planning/scheduling ablation results |
| 10 | #5 feasibility sprint | dual-memory disagreement metric + pilot training run |
| 11 | #3 feasibility sprint | 2-3 expert LoRA + static router stability report |
| 12 | Paper package + release prep | full ablations, figures, reproducibility checklist, artifact draft |

**Execution Principle**  
- 主线：`#4 → #2 → (#2+#4) → #1`
- 二期探索：`#5/#3` 并行小步试错，不阻塞主线交付。

---

## 4) Experiment Design Table（实验设计表）

| Dimension | Primary Metrics | Baselines | Key Ablations | Success Criteria |
|---|---|---|---|---|
| Task Quality | Success@1, Pass@k, plan validity rate | Current ParamAgent main | -PDL, -Bandit, -Memory | 主线 +6 分以上 |
| Efficiency | p50/p95 latency, token/tool cost | Fixed-budget inference | UCB vs TS vs contextual | 成本下降 ≥15% 且成功率不降 |
| Robustness | OOD success, failure recovery rate | No-retry / fixed retry | +counterfactual replay on/off | OOD 提升显著（p<0.05） |
| Interpretability | Trace completeness, causal consistency | Free-form reasoning logs | no-causal-edge / random-edge | 可解释链路覆盖率提升 |
| Stability | Variance across seeds, router entropy | single-policy baseline | expert count, routing policy | 方差可控，不劣化主线 |

**Recommended Baselines**  
- ParamAgent 当前稳定版本（single planner + fixed compute）
- ReAct/Toolformer-style fixed policy baseline（若项目已有）

---

## 5) Resource Requirements（资源需求）

### Compute
- 主线（#2+#4+#1）
  - 训练/离线回放：4-8 × A100（或同级）持续 4-6 周
  - 在线评测：1-2 × A100 + CPU 工具集群
- 二期（#5/#3）
  - 额外 4 × A100（短周期冲刺）

### Data
- 任务轨迹日志（含工具调用、错误类型、时延、成本）
- 计划结构数据（PDL AST + validator结果）
- 反事实回放样本集（失败轨迹重写）
- OOD评测集（跨领域/跨工具组合）

### Personnel
- Research Lead ×1（总体设计 + 论文主笔）
- Applied Scientist ×2（#2/#4 主线；#1 记忆）
- ML Engineer ×2（训练管线、评测、infra）
- Data/Benchmark Engineer ×1（数据协议与评测治理）
- Part-time PM/TPM ×0.5（节奏与里程碑管理）

---

## 6) Final Recommendation（最终建议）

先用 **#4 + #2** 拿到"高确定性收益 + 可发表结果"，随后接入 **#1** 形成机制创新闭环；  
将 **#5 / #3** 放在二期作为论文扩展与长期上限探索。

这一路径在"影响力 × 可控性 / 成本"上与专家排序一致，且最适合 12 周内形成可复现成果。

---

## Appendix: Expert Review Summary

### Expert 1 (Proposer)
提出五个方向，强调创新性与实现路径的平衡。

### Expert 2 (Critical Reviewer)
- 可行性评分：#4 (9/10) > #2 (8/10) > #1 (7/10) > #5 (5/10) > #3 (4/10)
- Publishability排名：#2 > #1 > #5 > #4 > #3
- 建议组合：#2 + #4 先行，#1 跟进，#3/#5 作为二期

### Expert 3 (Synthesizer)
综合两位专家意见，形成本文档的最终方案。

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-03  
**Authors**: Expert Panel (via Codex MCP)  
**Status**: Ready for Implementation
