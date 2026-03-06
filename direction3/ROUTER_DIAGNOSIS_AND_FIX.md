# Direction 3 Router 深度诊断与架构创新方案

## 一、当前 Router 损坏的根本原因

### 1.1 特征工程缺陷

**问题**：`feature_schema.py` 中的 23 个特征存在严重的信息不对称：
- **静态特征过多**（prompt长度、代码长度等）：这些特征在问题确定后就固定了，无法反映"当前状态下该用哪类记忆"
- **动态特征缺失**：缺少"当前候选记忆与query的相似度分布"、"候选记忆的质量指标"等关键信息
- **相似度特征设计错误**：
  ```python
  # 当前实现：只统计 max/mean，丢失了分布信息
  "prompt_sim_max": float(prompt_max),
  "prompt_sim_mean": float(prompt_mean),
  ```
  这导致 router 无法区分"有1个高相似度候选"和"有10个中等相似度候选"的场景。

### 1.2 训练数据质量问题

**问题**：`dataset_builder.py` 中的 `_heuristic_mix` 函数使用启发式规则生成训练标签：
```python
def _heuristic_mix(...) -> List[float]:
    w_prompt = max(0.0, 0.30 + prompt_signal)
    w_reflection = max(0.0, 0.25 + reflection_signal + 0.06 * reflection_rounds)
    w_negative = max(0.0, 0.20 + negative_signal + failure_boost)

    if solved:
        w_prompt += 0.25
        w_reflection += 0.20
    else:
        w_negative += 0.35
```

**致命缺陷**：
- 标签是基于"是否最终solved"的粗粒度信号，而不是"哪种记忆真正帮助了解决"
- 启发式权重（0.30, 0.25, 0.20）没有理论依据
- **循环依赖**：router 应该学习"何时用哪种记忆"，但训练标签本身就是基于固定规则生成的

### 1.3 模型架构过于简单

**问题**：`model.py` 中的 RouterMLP 是一个简单的 3 层 MLP：
```python
class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        # 输入 23 维 → 128 → 64 → 4 维输出（3个mix权重 + 1个confidence）
```

**问题**：
- **表达能力不足**：23维特征 → 128 → 64 → 4，无法捕捉复杂的记忆选择模式
- **置信度总是1.0的原因**：confidence 是通过 sigmoid(logits[3]) 计算的，但训练时 `target_solved` 是二值的（0或1），导致模型学会了"总是输出高置信度"
- **成功率2.2%的原因**：模型过拟合到训练集的启发式规则，泛化能力极差

### 1.4 推理时的 Soft-gate 机制掩盖了问题

**问题**：`paramAgent_architer9.py` 中的推理逻辑：
```python
if router_conf < router_conf_threshold:
    # 低置信度时混合 fallback 和 router 输出
    blended_scores = [
        ((1.0 - alpha) * fallback_scores[idx]) + (alpha * fused_scores[idx])
        for idx in range(len(candidates))
    ]
    return candidates[best_idx], [w0, w1, w2], router_conf, True
```

**问题**：
- 因为 router_conf 总是接近 1.0，所以这个 soft-gate 从未真正触发
- 即使 router 输出错误的 mix 权重，系统也会"信任"它，导致性能下降

---

## 二、架构创新方案：Multi-Stage Adaptive Router (MSAR)

### 2.1 核心思想

**不是修补现有 router，而是重新设计一个"分阶段自适应路由器"**：
1. **Stage 1: Candidate Quality Assessment** - 评估每个候选记忆的质量
2. **Stage 2: Context-Aware Mixing** - 基于当前状态和候选质量，动态决策 mix 权重
3. **Stage 3: Confidence Calibration** - 输出校准后的置信度，而不是过拟合的1.0

### 2.2 新特征工程：从"静态特征"到"候选-query交互特征"

**新增特征类别**：

#### A. 候选记忆质量特征（Per-Candidate）
```python
# 对每个候选记忆，提取：
- prompt_sim_score: 与当前问题的相似度
- reflection_sim_score: 与当前reflection的相似度（如果有）
- negative_penalty: 负样本惩罚
- candidate_solved_flag: 该候选记忆是否来自成功轨迹
- candidate_reflection_depth: 该候选记忆的reflection深度
- candidate_code_quality: 代码质量指标（长度、复杂度等）
```

#### B. 候选集分布特征（Aggregate）
```python
# 统计所有候选记忆的分布：
- prompt_sim_std: prompt相似度的标准差（区分"都很像"vs"只有1个像"）
- prompt_sim_top3_gap: top1和top3的相似度差距
- reflection_candidate_ratio: 有reflection的候选占比
- negative_candidate_ratio: 负样本候选占比
- candidate_diversity: 候选记忆的多样性（embedding聚类）
```

#### C. 当前状态特征（Context）
```python
# 保留部分有用的静态特征，但增加动态特征：
- current_attempt_count: 当前尝试次数
- current_failure_pattern: 当前失败模式（syntax/assert/timeout）
- reflection_available: 是否有reflection query
- memory_bank_size: 记忆库大小
```

**新特征维度**：约 40-50 维（比原来的23维更丰富）

### 2.3 新模型架构：Transformer-based Router

**为什么用 Transformer？**
- 候选记忆之间存在"竞争关系"（选了A就不选B）
- Transformer 的 self-attention 可以建模候选之间的相对重要性
- 可以处理变长的候选集（通过 padding/masking）

**架构设计**：
```python
class TransformerRouter(nn.Module):
    def __init__(self, feature_dim=50, num_candidates=20, d_model=256, nhead=4, num_layers=2):
        super().__init__()

        # 1. Candidate Encoder: 将每个候选的特征编码
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 2. Context Encoder: 编码当前状态
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 3. Transformer: 建模候选之间的交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Mix Predictor: 输出3个记忆类型的权重
        self.mix_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # [w_prompt, w_reflection, w_negative]
        )

        # 5. Confidence Predictor: 输出校准后的置信度
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, candidate_features, context_features, candidate_mask=None):
        # candidate_features: [batch, num_candidates, feature_dim]
        # context_features: [batch, feature_dim]

        # Encode candidates
        cand_encoded = self.candidate_encoder(candidate_features)  # [batch, num_cand, d_model]

        # Encode context and expand
        ctx_encoded = self.context_encoder(context_features).unsqueeze(1)  # [batch, 1, d_model]

        # Concatenate context + candidates
        seq = torch.cat([ctx_encoded, cand_encoded], dim=1)  # [batch, 1+num_cand, d_model]

        # Apply transformer
        if candidate_mask is not None:
            # Prepend True for context token
            mask = torch.cat([torch.ones(candidate_mask.size(0), 1, device=candidate_mask.device),
                             candidate_mask], dim=1)
        else:
            mask = None

        transformed = self.transformer(seq, src_key_padding_mask=mask)  # [batch, 1+num_cand, d_model]

        # Use context token (first token) for prediction
        context_repr = transformed[:, 0, :]  # [batch, d_model]

        # Predict mix weights
        mix_logits = self.mix_head(context_repr)  # [batch, 3]
        mix_weights = F.softmax(mix_logits, dim=-1)

        # Predict confidence
        conf_logit = self.conf_head(context_repr)  # [batch, 1]
        confidence = torch.sigmoid(conf_logit).squeeze(-1)  # [batch]

        return mix_weights, confidence
```

### 2.4 新训练策略：从"启发式标签"到"强化学习"

**问题**：我们无法直接获得"正确的 mix 权重"标签。

**解决方案**：使用 **Reinforcement Learning from Execution Feedback (RLEF)**

#### 训练流程：
1. **收集轨迹数据**：
   - 运行现有系统，记录每个问题的：
     - 候选记忆集合
     - 当前状态特征
     - 使用不同 mix 权重后的结果（solved/failed）

2. **构建奖励函数**：
   ```python
   def compute_reward(mix_weights, execution_result):
       if execution_result['solved']:
           # 成功：奖励高效的mix（鼓励稀疏权重）
           entropy = -sum(w * log(w) for w in mix_weights if w > 0)
           return 1.0 - 0.1 * entropy
       else:
           # 失败：惩罚
           return -0.5
   ```

3. **Policy Gradient 训练**：
   ```python
   # 使用 REINFORCE 算法
   for batch in dataloader:
       # Forward pass
       mix_weights, confidence = model(candidate_features, context_features)

       # Sample action (mix weights)
       action = mix_weights  # 或者加入探索噪声

       # 计算奖励（从离线数据中查询，或在线执行）
       reward = compute_reward(action, execution_result)

       # Policy gradient loss
       log_prob = torch.log(mix_weights + 1e-8).sum(dim=-1)
       policy_loss = -log_prob * reward

       # Confidence calibration loss (使用实际成功率作为标签)
       conf_loss = F.binary_cross_entropy(confidence, actual_success_rate)

       total_loss = policy_loss + 0.5 * conf_loss
       total_loss.backward()
       optimizer.step()
   ```

### 2.5 置信度校准：Temperature Scaling

**问题**：即使用新模型，confidence 仍可能过拟合。

**解决方案**：在验证集上进行 **Temperature Scaling**：
```python
class CalibratedRouter(nn.Module):
    def __init__(self, base_router):
        super().__init__()
        self.base_router = base_router
        self.temperature = nn.Parameter(torch.ones(1))  # 可学习的温度参数

    def forward(self, *args, **kwargs):
        mix_weights, conf_logit = self.base_router(*args, **kwargs)

        # Apply temperature scaling to confidence
        calibrated_conf = torch.sigmoid(conf_logit / self.temperature)

        return mix_weights, calibrated_conf

# 在验证集上优化 temperature
def calibrate_temperature(model, val_loader):
    # 固定 base_router 参数，只优化 temperature
    for param in model.base_router.parameters():
        param.requires_grad = False

    optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)

    def eval():
        loss = 0.0
        for batch in val_loader:
            _, conf = model(batch['features'], batch['context'])
            actual_success = batch['solved']
            loss += F.binary_cross_entropy(conf, actual_success)
        return loss

    optimizer.step(eval)
```

---

## 三、实现路径

### Phase 1: 新特征工程（1-2小时）
1. 修改 `feature_schema.py`，实现新的特征提取函数
2. 修改 `dataset_builder.py`，生成包含候选级特征的训练数据
3. 验证新特征的信息量（通过简单的决策树模型）

### Phase 2: Transformer Router 实现（2-3小时）
1. 创建 `memory_router/transformer_model.py`
2. 实现 `TransformerRouter` 类
3. 实现新的训练脚本 `train_transformer_router.py`

### Phase 3: 强化学习训练（3-4小时）
1. 收集离线轨迹数据（运行现有系统，记录所有中间状态）
2. 实现 RLEF 训练循环
3. 在 HumanEval/MBPP 上训练新 router

### Phase 4: 置信度校准（1小时）
1. 实现 Temperature Scaling
2. 在验证集上校准

### Phase 5: 集成与验证（2小时）
1. 修改 `paramAgent_architer9.py`，集成新 router
2. 在 24-worker 上运行全量 MBPP 验证
3. 对比 architer2-7 的 74.81% 基线

---

## 四、预期效果

### 4.1 Router 性能提升
- **置信度分布**：从"总是1.0"变为"0.3-0.9的合理分布"
- **成功率**：从2.2%提升到 >60%（当 conf > 0.7 时）
- **Mix 权重多样性**：不再是固定的 [0.33, 0.33, 0.33]，而是根据状态动态调整

### 4.2 整体性能提升
- **MBPP Pass@1**：从 74.81% 提升到 **78-80%**（保守估计）
- **关键场景改进**：
  - 有高质量 reflection 时，w_reflection 自动提升
  - 候选记忆质量差时，router 输出低置信度，触发 fallback
  - 负样本多时，w_negative 自动增强，避免错误记忆

### 4.3 可解释性提升
- 可以分析"哪些特征对 router 决策影响最大"（通过 attention weights）
- 可以可视化"router 在哪些问题上有信心，哪些没有"

---

## 五、立即行动计划

**现在开始实现 Phase 1-2，预计 3-4 小时完成核心代码，然后启动 24-worker 验证。**
