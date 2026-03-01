# ParamAgent 复现所需模型与数据集指南（arXiv:2602.23320）

更新时间：2026-03-01

## 1. 论文复现必需数据集

根据论文正文与附录（Setup / Appendix B.1 / Appendix B.2），复现 ParamAgent 主要需要：

| 用途 | 数据集 | 说明 |
| --- | --- | --- |
| 评测 | HumanEval | 代码任务评测 |
| 评测 | MBPP | 代码任务评测 |
| 评测 | LiveCodeBench | 代码任务评测（online benchmark 子集） |
| 评测 | MATH | 数学推理评测 |
| 评测 | HotpotQA | 多跳问答评测 |
| 评测 | 2WikiMultiHopQA | 多跳问答评测 |
| 训练 | APPS (`codeparrot/apps`) | 参数化模块微调数据 |
| 训练 | Synthetic data | 论文中额外构造数据，需要项目脚本生成，不是公开直接下载集 |

## 2. 三台机器当前数据状态

根路径：
- HPC2：`/hpc2hdd/home/jzhu997`
- HPC3：`/data/user/jzhu997`
- HPC4-test：`/data/user/user06`

状态（已核验）：
- `HumanEval`：已存在（`ParamAgent/benchmarks/humaneval_full.jsonl`）
- `MBPP`：已存在（`ParamAgent/benchmarks/mbpp-py.jsonl`）
- `MATH`：已存在（`cache/data/math`）
- `HotpotQA`：已存在（`cache/data/hotpotqa`）
- `2WikiMultiHopQA`：已存在（`cache/data/2wiki`）
- `LiveCodeBench`：此前缺失，现已补齐（`cache/data/livecodebench`）
- `APPS`：此前缺失，现已补齐（`cache/data/apps`）

## 3. 缺失数据集下载（在 HPC2 执行）

HPC3/HPC4-test 离线，统一在 HPC2 下载后分发。下载前必须取消代理并用清华/HF 镜像通道：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
```

下载命令：

```bash
huggingface-cli download --repo-type dataset livecodebench/code_generation_lite \
  --local-dir ~/cache/data/livecodebench

huggingface-cli download --repo-type dataset codeparrot/apps \
  --local-dir ~/cache/data/apps
```

## 4. 分发到 HPC3 / HPC4-test

在 HPC2 执行：

```bash
rsync -avh --delete ~/cache/data/livecodebench/ \
  jzhu997@hpc3login.hpc.hkust-gz.edu.cn:/data/user/jzhu997/cache/data/livecodebench/

rsync -avh --delete ~/cache/data/apps/ \
  jzhu997@hpc3login.hpc.hkust-gz.edu.cn:/data/user/jzhu997/cache/data/apps/

rsync -avh --delete ~/cache/data/livecodebench/ \
  user06@hpc4-test.hpc.hkust-gz.edu.cn:/data/user/user06/cache/data/livecodebench/

rsync -avh --delete ~/cache/data/apps/ \
  user06@hpc4-test.hpc.hkust-gz.edu.cn:/data/user/user06/cache/data/apps/
```

## 5. 完整性验证

三台机器都应满足以下检查（示例）：

```bash
test -f ~/cache/data/livecodebench/test.jsonl && echo "livecodebench ok"
test -f ~/cache/data/apps/test.jsonl && echo "apps ok"
```

当前核验结果：
- 三台机器均存在 `livecodebench/test.jsonl`
- 三台机器均存在 `apps/test.jsonl`

## 6. 运行时环境建议（离线服务器）

在 HPC3 / HPC4-test 的 sbatch 中建议固定：

```bash
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

这样可确保只用本地模型与数据集，不触发外网下载。
