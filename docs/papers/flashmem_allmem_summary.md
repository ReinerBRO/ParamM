# FlashMem 与 AllMem 总结

## 1) FlashMem 在做什么

- 目标：解决 Agent 长时任务中的“无状态”问题，避免每轮都重复重编码历史。
- 核心思路：把记忆生成直接耦合到主干模型推理过程，复用已有计算状态（computation reuse）。
- 关键模块：
  - Shared-KV Consolidator：使用最后隐状态作为历史摘要，从 backbone 的冻结 KV cache 中提炼 latent memory。
  - Cognitive Monitor：基于注意力熵判断不确定性，只在“模型困惑”时触发记忆整合，避免无效开销。
- 训练方式：训练轻量 consolidator（SFT），backbone 冻结；Cognitive Monitor 为无参数机制。
- 结果特点：在多任务上接近或达到重型 latent-memory 基线（如 MemGen），并显著降低长上下文时延（文中报告约 5x 相对 MemGen 的加速）。

## 2) AllMem 在做什么

- 目标：降低 Transformer 在超长上下文中的计算和缓存成本，同时尽量保持全注意力性能。
- 核心思路：在每层引入并行双分支记忆结构：
  - SWA（Sliding Window Attention）负责局部精细依赖。
  - 非线性 TTT memory 网络负责长期压缩记忆。
  - 通过可学习门控进行融合，实现短期/长期记忆协同。
- 训练方式：采用 memory-efficient fine-tuning + 蒸馏，把现有预训练 Transformer 转换为 AllMem 结构。
- 结果特点：在 LongBench/InfiniteBench/LV-Eval 上展示出高效长上下文能力；文中给出 37k 平均长度下接近全注意力表现、128k 上部分设置超过全注意力，并在长序列下显著节省计算与缓存。

## 3) 两者差异（如何理解）

- FlashMem 更偏“Agent 运行时记忆系统”：重点是何时触发记忆、如何低开销回忆。
- AllMem 更偏“LLM 主干架构改造”：重点是把模型层本身改造成长上下文友好的混合记忆-注意力结构。
- 一句话：FlashMem 主要优化“推理流程里的记忆注入”，AllMem 主要优化“模型结构里的记忆机制”。

