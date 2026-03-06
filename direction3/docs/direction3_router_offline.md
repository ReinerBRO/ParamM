# Direction3 Router Offline Pipeline

本模块提供 Direction3 路由器的离线数据构建与训练流程，目标输出：
- `router_mix`：3 维记忆权重（非负且归一化）
  1) prompt-sim 正样本记忆
  2) reflection-sim 正样本记忆
  3) negative-trajectories 抑制信号
- `router_conf`：是否 solved 的置信度

仅依赖 Python 标准库 + `torch`。

## 1) 生成离线监督数据

执行：

```bash
scripts/run_direction3_router_data.sh \
  <humaneval_phase1.jsonl> <humaneval_phase2.jsonl> <humaneval_mem.jsonl_or_pkl> \
  <mbpp_phase1.jsonl> <mbpp_phase2.jsonl> <mbpp_mem.jsonl_or_pkl> \
  <output_dir>
```

等价模块命令：

```bash
python -m memory_router.dataset_builder \
  --humaneval_phase1 ... --humaneval_phase2 ... --humaneval_mem ... \
  --mbpp_phase1 ... --mbpp_phase2 ... --mbpp_mem ... \
  --output_dir ...
```

输出文件：
- `output_dir/train_humaneval.jsonl`
- `output_dir/train_mbpp.jsonl`
- `output_dir/cross_eval_splits.json`
- `output_dir/dataset_stats.json`

样本结构为 `state -> target_mix -> outcome_solved`。缺失字段会自动回退为安全默认值。

## 2) 训练路由器

执行：

```bash
scripts/run_direction3_router_train.sh \
  <train_jsonl> <val_jsonl> <output_ckpt_or_dir> [epochs] [batch_size] [lr] [seed]
```

等价模块命令：

```bash
python -m memory_router.train_router \
  --train_jsonl ... --val_jsonl ... --output_ckpt ... \
  --epochs 20 --batch_size 32 --lr 1e-3 --seed 42
```

说明：
- 若 `--output_ckpt` 传目录，训练脚本会保存为 `<目录>/router.ckpt`
- 若传 `.ckpt` 文件路径，则按该路径保存

产物：
- `router.ckpt`
  - `state_dict`
  - `feature_order`
  - `norm_mean`, `norm_std`
  - 训练配置与最佳验证 loss
- `train_metrics.json`（与 ckpt 同目录）

## 3) 单条 state 推理

从 JSON 文件读取：

```bash
python -m memory_router.infer_router \
  --ckpt <router.ckpt> \
  --state_json <state.json>
```

从命令行字符串读取：

```bash
python -m memory_router.infer_router \
  --ckpt <router.ckpt> \
  --state_json_str '{"prompt":"...","reflections":[],"test_feedback":[]}'
```

返回字段：
- `router_mix`（归一化 3 维权重）
- `router_conf`（sigmoid 置信度）
- `features` 与 `feature_order`
