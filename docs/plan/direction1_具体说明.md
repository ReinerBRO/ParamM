# Direction1 到底在做什么（具体版）

> 文件目的：给你和老师看“Direction1 实际改了什么、为什么有效、和原版 ParamAgent 的边界在哪里”。
> 说明范围：基于你当前仓库 `direction1/` 的实现，不是只讲论文抽象。

---

## 1) 一句话定义

Direction1 的核心是：
**把 Phase2 的“线性反思重试”升级成“验证驱动的搜索式修复”**。

不是重写整个框架，而是主要改 **Phase2**：
- 原版：失败题按单轨迹线性迭代修
- Direction1：失败题先走 SearchEngine 做候选扩展与选择，异常时再回退线性流程

---

## 2) 原版 ParamAgent 在 Phase2 的问题（你这次改造的动机）

原版 Phase2 虽然有 cross-sample memory（会检索历史正轨迹），但核心修复策略仍是：
- 每轮最多试 2 条 reflection
- 线性推进到下一轮
- 缺少“系统性的多候选比较与预算控制”

这会带来两个问题：
1. 候选选择偏局部（更像边走边试）
2. 验证信号利用不够结构化（缺少统一评分/选择机制）

---

## 3) Direction1 的结构改动（你代码里真实发生的事）

### 3.1 新增 Phase2 模式开关

`phase2_search_mode` 支持：
- `linear`：保留原线性流程
- `search`：进入搜索流程

这保证了可回退、可对照。

### 3.2 SearchEngine 负责“扩展 + 打分 + 选择”

搜索配置里有显式预算：
- `beam_size`
- `max_nodes`
- `max_depth`
- `early_stop`

也就是说，Direction1 不是无限树搜索，而是受控搜索。

### 3.3 候选不是盲选，而是组合评分

对候选代码计算：
- 通过概率（pass_prob）
- 多样性收益（diversity）
- 复杂度惩罚（complexity_penalty）

总分：
`score = pass_prob * w1 + diversity * w2 - complexity_penalty * w3`

### 3.4 验证器不是单一信号，而是三路融合

VerifierAdapter 同时整合：
- visible tests
- generated tests
- static checks

并按权重融合为 pass probability，同时统计 verifier cost。

### 3.5 加入哈希缓存，降低重复验证开销

对代码做 hash，相同代码不会重复跑同一类验证。

### 3.6 搜索失败可回退线性流程

若 search 分支抛异常，不会直接中断任务，自动 fallback 到原线性 reflexion。

---

## 4) 运行时到底怎么走（Phase2）

以单题为例，Direction1 在 `search` 模式下的大流程是：

1. 先做一次初始实现与测试（沿用原框架）
2. 如果还没解出，进入 `_run_phase2_search(...)`
3. SearchEngine 从当前解作为初始节点开始扩展
4. 每个候选都跑多路验证并评分
5. 在预算内持续“扩展-打分-保留更优候选”
6. 提前满足 all_pass 可 early-stop
7. 输出 best_node 作为最终候选，再走最终评估
8. 若 search 抛异常，记录 fallback 标记并走线性流程

---

## 5) Direction1 没有改什么（边界很重要）

Direction1 **没有**把 ParamAgent 全部推倒重来，以下仍保留：
- Phase1 主逻辑
- memory bank 机制（positive/negative trajectories）
- pitfall/mistake_insights 信号注入链路
- 两阶段整体框架（Phase1 -> Phase2）

所以它是“**Phase2 策略升级**”，不是“全系统重构”。

---

## 6) 为什么你这次结果会涨（结合实现机制）

从机制看，收益来源主要是：
1. 候选选择从“局部线性”变成“预算内搜索比较”
2. 用统一评分把“可过测概率 + 多样性 + 复杂度”一起考虑
3. 验证器融合三路信号，减少单一反馈噪声
4. 缓存降低重复验证成本，使同等预算下可探索更多有效候选

这与“反思多样性 + 可验证性驱动修复”的目标一致。

---

## 7) 你可以怎么对老师讲（30秒版本）

我们没有改 ParamAgent 的整体两阶段框架，主要升级了 Phase2。
原来是线性重试，现在是受控搜索：在给定预算内扩展多个候选，
用“通过概率 + 多样性 - 复杂度”统一打分，并融合 visible/generated/static 三路验证。
这样做把修复从“边试边走”变成“验证驱动选择”，因此在 HumanEval/MBPP 都得到稳定提升。

---

## 8) 关键实现入口（便于你快速回看代码）

- `direction1/paramAgent.py`
  - `phase2_search_mode` 开关
  - `_run_phase2_search(...)` 调用点
  - fallback 到 linear 的逻辑
- `direction1/search_engine/engine.py`
  - 搜索主循环与预算控制
- `direction1/search_engine/scorer.py`
  - 候选评分函数
- `direction1/search_engine/verifier_adapter.py`
  - 多验证器融合与哈希缓存

