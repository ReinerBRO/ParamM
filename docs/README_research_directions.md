# ParamAgent 研究方向文档索引

## 文档列表

### 1. 执行摘要（推荐先读）
📄 **`five_directions_summary.md`**
- 五大方向速览
- 执行策略和时间表
- 预期成果和风险
- 下一步行动

### 2. 完整方案（详细实现）
📄 **`five_research_directions_brainstorm.md`** (364 lines)
- 每个方向的详细方法论
- Phase 0-3 实现路径
- 文件级代码变更清单
- 实验设计和消融策略
- 资源需求和人力配置

## 五大研究方向

| # | 方向 | 预期收益 | 可行性 | 发表价值 | 优先级 |
|---|------|---------|--------|---------|--------|
| 2 | Surprisal-Bandit 测试时算力调度 | +4~9分 | 8/10 | ⭐⭐⭐⭐⭐ | P0 (Week 3-4) |
| 4 | Plan-First PDL 程序合成 | +2~6分 | 9/10 | ⭐⭐⭐⭐ | P0 (Week 1-2) |
| 1 | Causal Event Memory 因果记忆 | +3~7分 | 7/10 | ⭐⭐⭐⭐ | P1 (Week 7-8) |
| 5 | Disagreement-Driven 双记忆训练 | +3~7分 | 5/10 | ⭐⭐⭐ | P2 (Week 10) |
| 3 | VB-LoRA Specialist Swarm 专家群 | +3~8分 | 4/10 | ⭐⭐⭐ | P2 (Week 11) |

## 执行时间线

```
Week 1-2:  #4 PDL Phase 0-1 (grammar + parser + validator)
Week 3-4:  #2 Bandit Phase 0-1 (surprisal + UCB/TS)
Week 5-6:  #2 + #4 Integration (PDL complexity → budget)
Week 7-8:  #1 Causal Memory Phase 0-1 (event store + replay)
Week 9:    #1 + 主线融合 (memory-aware planning)
Week 10:   #5 Feasibility Sprint (disagreement metric)
Week 11:   #3 Feasibility Sprint (expert LoRA + router)
Week 12:   Paper Package (ablations + figures + artifact)
```

## 关键文献支持

- **Surprisal-guided selection**: 80% vs 50% success (arXiv 2602.07670, Feb 2026)
- **AI21 Maestro**: Horizontal scaling + PDL approach (SWE-bench)
- **VB-LoRA**: 99.6% parameter reduction (NeurIPS 2024)
- **Event-sourced memory**: 60% cost reduction for long-horizon tasks

## 专家评审总结

### Expert 1 (Proposer)
提出五个方向，强调创新性与实现路径的平衡。

### Expert 2 (Critical Reviewer)
- 可行性评分：#4 (9/10) > #2 (8/10) > #1 (7/10) > #5 (5/10) > #3 (4/10)
- Publishability排名：#2 > #1 > #5 > #4 > #3
- 建议组合：#2 + #4 先行，#1 跟进，#3/#5 作为二期

### Expert 3 (Synthesizer)
综合两位专家意见，形成最终方案。

## 快速开始

1. **阅读执行摘要**：`five_directions_summary.md`
2. **深入了解方向**：`five_research_directions_brainstorm.md`
3. **查看当前 baseline**：`../README.md`, `reproduction_analysis.md`
4. **准备实验环境**：参考 `pathA_hpc4_miku_quickstart.md`

## 联系方式

如有问题或建议，请联系项目负责人或在项目仓库提 issue。

---

**文档生成日期**: 2026-03-03
**生成方式**: Codex MCP Expert Panel (3 experts)
**状态**: Ready for Implementation
