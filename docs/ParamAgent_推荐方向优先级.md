# ParamAgent 推荐方向优先级

## 结论

基于当前代码实现、`Direction1` 的真实形态、以及 `ParamMem` 论文的主叙事，我认为五个推荐方向的**研究主线优先级**应当写成：

1. **Read-Write Closed-Loop ParamMem**
2. **State-Adaptive Memory Router**
3. **Learned Search Value Model**
4. **RepairDSL / Executable Reflection**
5. **Error-Repair Graph Memory**

这个排序的核心原则不是“哪个最容易做”，而是：

- 是否真正贴合 `ParamMem` 的论文主轴；
- 是否能把当前系统从“read-only reflective prior”升级成更完整的 memory agent；
- 是否能和你现有 `Direction1`、router 线、LoRA 训练线形成闭环；
- 是否有足够强的论文故事，而不只是局部工程优化。

---

## 1. Read-Write Closed-Loop ParamMem

### 优先级：最高

### 推荐理由

- 这是五个方向里**最适合当整篇论文主故事**的一条。
- 现有 `ParamMem` 本质上是：先在辅助反思数据上离线训练一个 parametric reflective prior，再在推理时读取它生成 `r_t^g` 来条件化 actor；论文虽然展示了 self-improvement 潜力，但还没有把“在线经验写回 memory，并进一步固化到参数记忆里”真正做成主系统。
- 因此，把系统从 **read-only ParamMem** 升级到 **read-write closed-loop ParamMem**，是最自然、也最有叙事张力的扩展。

### 它在系统中的定位

- 它不是简单的“多加一个训练脚本”。
- 它应该成为整个新系统的总框架：
  - `read memory`
  - `route memory`
  - `search / repair`
  - `verify`
  - `write explicit memory`
  - `consolidate into parametric memory`

### 适合当主创新的原因

- 论文味最强。
- 和 `ParamMem` 原文最连续，不会让人觉得你换了题目。
- 可以自然吸收后面四个方向，形成统一闭环。

### 主要风险

- 如果把“write”写成逐题在线更新参数，会显得不现实且不稳定。
- 更合理的写法应是：**run-level 或 batch-level consolidation**，即先写入显式高价值 buffer，再周期性做小步 LoRA 更新。

---

## 2. State-Adaptive Memory Router

### 优先级：第二

### 推荐理由

- 这是五个方向里**最强的机制层创新**。
- ParamAgent/ParamAgent-plus 实际上已经在用三类 memory：episodic memory、cross-sample memory、parametric memory；但当前系统并没有真正回答“什么时候该更依赖哪一种”。
- 论文也已经给出信号：不同任务对 diversity 的收益不同，说明“固定 memory 配方”并不总是最优。

### 它在系统中的定位

- 它是闭环中的 `route memory` 模块。
- 主作用不是替代 ParamMem，而是决定：
  - 这道题当前更该看哪类 memory；
  - 每种 memory 的权重是多少；
  - 是否值得投入更深的 search budget。

### 为什么排第二

- 它和主线的耦合非常自然。
- 相比 Search 或 DSL，它更能解释“为什么同样的 memory 在不同题上效果不同”。
- 这是把三记忆统一成一个真正系统的关键，而不是简单并列堆叠。

### 主要风险

- 若一开始就上 RL/bandit，会显著增加不确定性。
- 因此更合理的起步是：**先做监督式 router，再逐步在线化**。

---

## 3. Learned Search Value Model

### 优先级：第三

### 推荐理由

- 这条线最适合接在你当前 `Direction1` 后面。
- 你现有的 search 方向，真实价值并不是“search 本身就是主创新”，而是它已经开始产出更丰富的 Phase2 过程信号：候选轨迹、verifier 成本、best-score trace、正负例路径。
- 这些信号天然可以反过来训练一个 value/scorer model，用来预测“哪个节点值得继续扩展”。

### 它在系统中的定位

- 它不是新的总故事，而是闭环中的 `search / repair` 关键增强件。
- 作用是把当前手工打分函数升级成 learned scorer。

### 为什么不是第一

- 它更像 `Direction1` 的升级版，而不是完整 memory 论文的总纲。
- 单独写成主故事，会比较像“Phase2 search engineering”，而不是“ParamMem 体系扩展”。

### 为什么仍然很重要

- 它很可能是最直接的涨点来源之一。
- 尤其对 HumanEval 这类更依赖 Phase2 修复质量的数据集，提升空间明显。

---

## 4. RepairDSL / Executable Reflection

### 优先级：第四

### 推荐理由

- 这条线的价值主要在于：把“反思”从自然语言提升成可执行修复动作。
- 它对系统可解释性很强，也能显著减少无效 edit variance。

### 它在系统中的定位

- 它属于闭环中的 `search / repair` 子模块。
- 更准确地说，它是“如何执行修复”的表达层升级。

### 为什么不更靠前

- 它和 memory 主线的连接没有前 3 个方向那么强。
- 单独写出来，更像 program repair / controllable editing 论文，而不是 memory agent 论文。

### 为什么仍然值得保留

- 它能让系统机制更清楚、更可分析。
- 对 MBPP 这类局部 bug 更多的数据集，可能特别有效。

---

## 5. Error-Repair Graph Memory

### 优先级：第五

### 推荐理由

- 这个方向学术上很好看，因为它把“错误类型 → 修复动作”结构化成图，天然符合 memory system 的直觉。
- 但从当前代码现实出发，它是五个方向里**工程跨度最大、最晚该做**的一条。

### 它在系统中的定位

- 它是闭环中的 `write explicit memory` 与 `read structured memory` 模块。
- 更适合作为后期增强：为 parametric memory 提供结构化显式先验。

### 为什么排最后

- 你当前代码没有图抽取、图索引、图检索的稳定基础。
- 如果过早推进，很容易把主线资源消耗在基础设施上，而不是先把主故事跑通。

### 为什么它不是没价值

- 它是后期最有可能提升 hard case 命中率的结构化组件。
- 更适合放在主系统稳定后，作为显式 memory 层的升级。

---

## 推荐的整体写法

如果要把这五个方向组织成一篇完整的研究路线，我建议统一成：

**从 read-only ParamMem 升级为 read-write closed-loop memory agent。**

对应关系如下：

- **主创新**：Read-Write Closed-Loop ParamMem
- **核心机制 1**：State-Adaptive Memory Router
- **核心机制 2**：Learned Search Value Model
- **机制增强**：RepairDSL / Executable Reflection
- **显式 memory 增强**：Error-Repair Graph Memory

这样写的好处是：

- `Direction1` 不再被错误地写成整篇论文的主轴；
- 你已有的 search 进展可以自然降级为“Phase2 搜索底座”；
- 五个方向都能被纳入同一个闭环系统，而不是五个松散模块。

---

## 实施顺序建议（工程启动顺序）

虽然研究主线优先级如上，但如果按**工程落地顺序**启动，我建议：

1. `State-Adaptive Memory Router`
2. `Learned Search Value Model`
3. `Read-Write Closed-Loop ParamMem`
4. `RepairDSL / Executable Reflection`
5. `Error-Repair Graph Memory`

原因很简单：

- Router 和 Search Value Model 更容易直接复用现有代码与日志；
- 主创新 `Read-Write ParamMem` 最好建立在前两者已经能稳定挖高价值轨迹的前提下；
- 图记忆最晚做，避免基础设施过早拖慢主线。

---

## 最终建议

如果你只允许保留一个最强论文故事，我建议选：

**Read-Write Closed-Loop ParamMem**

如果你要保留五个方向并形成完整路线，我建议的最终结构是：

- 第一层：`Read-Write Closed-Loop ParamMem`（总故事）
- 第二层：`Memory Router` + `Search Value Model`（最关键的系统机制）
- 第三层：`RepairDSL` + `Graph Memory`（解释性与结构化增强）

