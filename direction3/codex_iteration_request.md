# Direction 3 架构改进请求

## 当前状态
- **验证结果**: phase2_acc = 73.55% (292/397)
- **目标**: phase2_acc > 75.31%
- **差距**: 1.76%
- **需要**: 再解决约7个问题 (397 * 0.0176 ≈ 7)

## 已实现架构
1. **Transformer Router** (刚完成训练)
   - 模型: d_model=256, nhead=4, num_layers=2
   - 训练数据: 318 train, 79 val
   - Best val_loss: 0.3102
   - 参数量: 1.1M

2. **Enhanced Features**
   - 21维上下文特征
   - 9维per-candidate特征
   - Self-attention建模候选交互

3. **三记忆融合**
   - Prompt memory (正例轨迹)
   - Reflection memory (反思改进)
   - Negative memory (负例惩罚)

## 问题分析
Router已经训练完成并集成，但phase2_acc仍未达标。可能原因：

1. **Router置信度阈值**: 当前0.6可能过高，导致很多情况fallback
2. **特征工程**: 21+9维特征可能不够充分
3. **训练数据**: 318个样本可能不足
4. **Router架构**: Transformer可能需要更深或更宽
5. **Memory质量**: 三个memory bank的质量和覆盖度

## 改进方向
请分析以下方面并提出具体改进方案：

1. **Router诊断**
   - 检查router实际使用率（多少问题使用了router vs fallback）
   - 分析router confidence分布
   - 查看router mix weights分布

2. **失败案例分析**
   - 找出phase2仍然失败的105个问题
   - 分析失败模式
   - 确定是router问题还是memory问题

3. **架构优化**
   - Router threshold调优
   - 特征增强
   - 模型架构改进
   - Memory bank扩充

## 约束条件
- **必须保持Direction 3核心**: 学习"在什么状态下使用哪种memory"
- **不能绕过router**: 不能退化成ensemble或multi-strategy
- **可用资源**: miku集群24 workers并发
- **时间限制**: 需要快速迭代

## 期望输出
1. 诊断分析报告
2. 具体改进方案（优先级排序）
3. 实施步骤
4. 预期提升幅度

## 相关文件
- Router实现: `memory_router/transformer_model.py`
- 特征提取: `memory_router/enhanced_features.py`
- 主逻辑: `paramAgent.py`
- 训练脚本: `memory_router/train_transformer_router.py`
- 最新结果: `results/mbpp_router_runs/dir3_router_mbpp_architer8_20260305_224152/`
