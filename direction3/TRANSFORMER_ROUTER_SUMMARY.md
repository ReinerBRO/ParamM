# Direction 3 Transformer Router - 实施总结

## 已完成工作

### 1. 根本原因诊断 ✓

**发现的核心问题**：
- **特征工程缺陷**：原有 23 个特征过于静态，缺少候选记忆质量和分布信息
- **训练数据质量差**：使用启发式规则生成标签，存在循环依赖
- **模型架构过简单**：3 层 MLP 无法捕捉候选记忆之间的交互关系
- **置信度总是 1.0**：因为训练标签是二值的，模型学会了"总是高置信度"
- **成功率仅 2.2%**：模型过拟合到启发式规则，泛化能力极差

详见：`/data/user/user06/ParamAgent/direction3/ROUTER_DIAGNOSIS_AND_FIX.md`

### 2. 新架构设计与实现 ✓

#### A. 增强特征工程 (`enhanced_features.py`)
- **Per-candidate features** (9 维)：每个候选记忆的质量指标
  - 相似度分数（prompt/reflection/negative）
  - 是否来自成功轨迹
  - 代码质量指标
  - Reflection 深度

- **Candidate-set features** (10 维)：候选集的分布统计
  - 相似度的均值/标准差/最大值
  - Top-3 gap（区分"都很像" vs "只有1个像"）
  - 候选类型比例（reflection/negative/solved）

- **Context features** (11 维)：当前问题状态
  - 问题特征（长度等）
  - 当前尝试次数、reflection 轮数
  - 失败模式（syntax/runtime/assert/timeout）

**总特征维度**：21 维 context + 9 维 per-candidate

#### B. Transformer Router 模型 (`transformer_model.py`)
```python
class TransformerRouter(nn.Module):
    # 架构：
    # 1. Candidate Encoder: 9 → d_model
    # 2. Context Encoder: 21 → d_model
    # 3. Transformer: 建模候选之间的交互
    # 4. Mix Head: d_model → 3 (prompt/reflection/negative 权重)
    # 5. Confidence Head: d_model → 1 (校准后的置信度)
```

**关键创新**：
- 使用 self-attention 建模候选记忆之间的竞争关系
- Temperature scaling 用于置信度校准
- Pre-LN Transformer 提升训练稳定性

#### C. 训练流程 (`train_transformer_router.py`)
- 支持从 phase1/phase2 logs 构建训练数据
- 使用 MSE loss 训练 mix weights
- 使用 BCE loss 训练 confidence
- 添加 entropy regularization 鼓励稀疏权重
- 计算 Expected Calibration Error (ECE) 监控校准质量

#### D. 推理接口 (`infer_transformer_router.py`)
- 兼容现有 `infer_router` 接口
- 输出格式：`{"router_mix": [w0, w1, w2], "router_conf": conf}`

### 3. 测试验证 ✓

**测试结果**：
```
✓ Feature extraction tests passed
✓ Model forward pass tests passed (49,093 parameters)
✓ Loss function tests passed
✓ Inference tests passed
```

所有单元测试通过，模型可以正常前向传播和推理。

---

## 文件清单

### 核心实现
1. `/data/user/user06/ParamAgent/direction3/memory_router/enhanced_features.py` - 增强特征提取
2. `/data/user/user06/ParamAgent/direction3/memory_router/transformer_model.py` - Transformer router 模型
3. `/data/user/user06/ParamAgent/direction3/memory_router/train_transformer_router.py` - 训练脚本
4. `/data/user/user06/ParamAgent/direction3/memory_router/infer_transformer_router.py` - 推理接口
5. `/data/user/user06/ParamAgent/direction3/memory_router/build_transformer_dataset.py` - 数据集构建

### 工具脚本
6. `/data/user/user06/ParamAgent/direction3/scripts/train_transformer_router.sh` - 一键训练脚本
7. `/data/user/user06/ParamAgent/direction3/tests/test_transformer_router.py` - 单元测试

### 文档
8. `/data/user/user06/ParamAgent/direction3/ROUTER_DIAGNOSIS_AND_FIX.md` - 诊断与方案文档
9. `/data/user/user06/ParamAgent/direction3/TRANSFORMER_ROUTER_SUMMARY.md` - 本文档

---

## 下一步：训练与验证

### Phase 1: 训练新 Router（预计 1-2 小时）

使用现有的 MBPP phase1/phase2 logs 训练：

```bash
cd /data/user/user06/ParamAgent/direction3

# 设置数据路径
export PHASE1_LOG="/data/user/user06/ParamAgent/results/mbpp/paramAgent/dir3_router_mbpp_20260303_223346/merged_phase1/phase1_merged_log.jsonl"
export PHASE2_LOG="/data/user/user06/ParamAgent/results/mbpp/paramAgent/dir3_router_mbpp_20260303_223346/merged_phase2/second_stage_log.jsonl"
export OUTPUT_DIR="/data/user/user06/ParamAgent/direction3/router_checkpoints/transformer_mbpp_v1"

# 运行训练
bash scripts/train_transformer_router.sh
```

**预期输出**：
- 训练数据：~400 samples (MBPP 有 500 题，部分可能缺失)
- 训练时间：30 epochs × ~1 min/epoch = 30 分钟
- 最终 val_loss：预计 < 0.5
- 最终 conf_acc：预计 > 0.7
- 最终 ECE：预计 < 0.15

### Phase 2: 集成到 paramAgent（预计 30 分钟）

修改 `paramAgent_architer9.py`，添加 Transformer router 支持：

```python
# 在文件开头添加
try:
    from memory_router.infer_transformer_router import infer_transformer_router
    TRANSFORMER_ROUTER_AVAILABLE = True
except ImportError:
    TRANSFORMER_ROUTER_AVAILABLE = False

# 在 select_stage2_traj 函数中
if router_enable and router_ckpt_path.endswith("transformer_router.ckpt"):
    # 使用新的 Transformer router
    router_output = infer_transformer_router(
        ckpt_path=router_ckpt_path,
        state=router_state,
        candidates=candidates,
        prompt_sims=prompt_sims,
        reflection_sims=reflection_sims,
        negative_penalties=negative_penalties,
    )
else:
    # 使用旧的 MLP router
    router_output = infer_router(router_ckpt_path, router_state)
```

### Phase 3: 24-Worker 全量验证（预计 2-3 小时）

在 miku 上启动 MBPP 全量测试：

```bash
# 创建新的运行脚本
cat > /data/user/user06/ParamAgent/direction3/scripts/hpc4/run_mbpp_transformer_router_miku.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=mbpp_transformer_router
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=logs/mbpp_transformer_router_%j.log

cd /data/user/user06/ParamAgent/direction3

python main_param.py \\
    --run_name "mbpp_transformer_router_$(date +%Y%m%d_%H%M%S)" \\
    --root_dir "." \\
    --dataset_type mbpp \\
    --dataset mbpp \\
    --model llama3_1_8b \\
    --pass_at_k 1 \\
    --max_iters 8 \\
    --num_workers 24 \\
    --verbose \\
    --router_enable \\
    --router_ckpt_path "/data/user/user06/ParamAgent/direction3/router_checkpoints/transformer_mbpp_v1/router.ckpt" \\
    --router_conf_threshold 0.6
EOF

chmod +x /data/user/user06/ParamAgent/direction3/scripts/hpc4/run_mbpp_transformer_router_miku.sh

# 提交任务
sbatch /data/user/user06/ParamAgent/direction3/scripts/hpc4/run_mbpp_transformer_router_miku.sh
```

---

## 预期效果

### Router 性能改进
| 指标 | 旧 MLP Router | 新 Transformer Router (预期) |
|------|---------------|------------------------------|
| 置信度分布 | 总是 ~1.0 | 0.3-0.9 合理分布 |
| 成功率 (conf > 0.7) | 2.2% | > 60% |
| Mix 权重多样性 | 固定 [0.33, 0.33, 0.33] | 动态调整 |
| ECE (校准误差) | > 0.5 | < 0.15 |

### 整体性能提升
| 数据集 | architer2-7 基线 | Transformer Router (预期) | 提升 |
|--------|------------------|---------------------------|------|
| MBPP Pass@1 | 74.81% | 78-80% | +3-5% |

**关键场景改进**：
1. **高质量 reflection 场景**：w_reflection 自动提升，充分利用 reflection 记忆
2. **候选质量差场景**：router 输出低置信度，触发 fallback 机制
3. **负样本多场景**：w_negative 自动增强，避免错误记忆干扰

---

## 技术亮点

1. **从"修补"到"重构"**：不是简单修复旧 router，而是基于根本原因分析重新设计
2. **特征工程创新**：从静态特征转向候选-query 交互特征
3. **架构创新**：使用 Transformer 建模候选记忆之间的竞争关系
4. **置信度校准**：Temperature scaling 解决过拟合问题
5. **可解释性**：可以通过 attention weights 分析 router 决策

---

## 风险与缓解

### 风险 1：训练数据不足
- **现状**：MBPP 只有 ~400 samples
- **缓解**：
  - 使用 data augmentation（随机 dropout 候选记忆）
  - 合并 HumanEval 数据进行联合训练
  - 使用较小的模型（d_model=256）避免过拟合

### 风险 2：推理速度
- **现状**：Transformer 比 MLP 慢
- **缓解**：
  - 模型很小（~50K 参数），推理时间 < 10ms
  - 可以使用 TorchScript 加速
  - Router 只在 phase2 调用，不是瓶颈

### 风险 3：集成复杂度
- **现状**：需要修改 paramAgent_architer9.py
- **缓解**：
  - 保持接口兼容，只需修改 10-20 行代码
  - 提供 fallback 到旧 router 的机制
  - 充分测试后再部署

---

## 立即行动

**现在可以开始训练**：

```bash
cd /data/user/user06/ParamAgent/direction3

# 检查数据是否存在
ls -lh /data/user/user06/ParamAgent/results/mbpp/paramAgent/dir3_router_mbpp_20260303_223346/merged_phase1/phase1_merged_log.jsonl
ls -lh /data/user/user06/ParamAgent/results/mbpp/paramAgent/dir3_router_mbpp_20260303_223346/merged_phase2/second_stage_log.jsonl

# 启动训练
export PHASE1_LOG="/data/user/user06/ParamAgent/results/mbpp/paramAgent/dir3_router_mbpp_20260303_223346/merged_phase1/phase1_merged_log.jsonl"
export PHASE2_LOG="/data/user/user06/ParamAgent/results/mbpp/paramAgent/dir3_router_mbpp_20260303_223346/merged_phase2/second_stage_log.jsonl"
export OUTPUT_DIR="/data/user/user06/ParamAgent/direction3/router_checkpoints/transformer_mbpp_v1"
export EPOCHS=30

bash scripts/train_transformer_router.sh
```

训练完成后，立即集成并启动 24-worker 验证。
