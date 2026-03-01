# HPC2 / HPC3 任务提交规范（SLURM / sbatch）

## 1) 两台机器关键差异

| 项目 | HPC2 | HPC3 |
|---|---|---|
| 登录别名 | `hpc2` | `hpc3` |
| Home 路径 | `/hpc2hdd/home/jzhu997` | `/data/user/jzhu997` |
| `sbatch` 路径 | 常见为 `/opt/slurm/bin/sbatch`（若 PATH 无命令） | `/usr/bin/sbatch` |
| 网络 | 可联网（相对宽松） | 离线/受限网络（外网通常不可达） |
| 典型环境路径 | `/hpc2hdd/home/jzhu997/miniforge3/envs/<env>` | `/data/user/jzhu997/envs/<env>` 或 `/data/user/jzhu997/miniforge3/envs/<env>` |

> 建议：在 HPC2 上先完成依赖下载和环境构建，再迁移到 HPC3 使用。

---

## 2) 可用分区（实测）

### HPC2（`/opt/slurm/bin/sinfo`）

常用 CPU 分区：
- `i64m512u`
- `i64m512ue`
- `long_cpu`
- `emergency_cpu`

常用 GPU 分区：
- `i64m1tga800u`
- `i64m1tga800ue`
- `i64m1tga40u`
- `i64m1tga40ue`
- `long_gpu`
- `emergency_gpu`
- `emergency_gpua40`

其他可见分区（按需使用）：
- `a128m512u`、`a128m512ue`
- `i96m3tu`、`i96m3tue`
- `debug`
- 租赁/专用队列（如 `czhangcn_rent`、`slurm01_qos` 等）

### HPC3（`sinfo`）

- `acd_u`（默认 GPU 分区，带抢占）
- `acd_ue`（GPU 分区，不可抢占）
- `emergency_acd`（紧急 GPU 分区）

> 重要：HPC3 当前策略通常要求必须带 `--gres=gpu:N`，纯 CPU 作业可能被拒绝。

---

## 3) 脚本模板

### 3.1 HPC2 CPU 模板

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -p i64m512ue
#SBATCH -n 16
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /hpc2hdd/home/jzhu997/pythonprojects/<project>

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/job_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

source /hpc2hdd/home/jzhu997/miniforge3/envs/<env_name>/bin/activate
export PYTHONUNBUFFERED=1

python main.py
```

### 3.2 HPC2 GPU 模板

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /hpc2hdd/home/jzhu997/pythonprojects/<project>

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/job_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
source /hpc2hdd/home/jzhu997/miniforge3/envs/<env_name>/bin/activate
export PYTHONUNBUFFERED=1

nvidia-smi || true
python main.py
```

### 3.3 HPC3 GPU 模板（推荐）

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -p acd_ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/pythonprojects/<project>

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/job_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true

# 优先自定义 envs，其次 miniforge3/envs
source /data/user/jzhu997/envs/<env_name>/bin/activate
# 或：source /data/user/jzhu997/miniforge3/envs/<env_name>/bin/activate

export PYTHONUNBUFFERED=1

# HPC3 离线常用设置
export HF_HOME=/data/user/jzhu997/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
python main.py
```

---

## 4) 关键规范

### SBATCH 头部参数

| 参数 | 说明 | 示例 |
|---|---|---|
| `-J` | 任务名称 | `#SBATCH -J train_v1` |
| `-p` | 分区 | `#SBATCH -p acd_ue` |
| `-n` | CPU 核心数 | `#SBATCH -n 8` |
| `--gres=gpu:N` | GPU 数量 | `#SBATCH --gres=gpu:2` |
| `-D` | 工作目录（绝对路径） | `#SBATCH -D /data/user/jzhu997/pythonprojects/xxx` |
| `-o` / `-e` | 标准输出/错误 | 推荐设为 `/dev/null`，用 `exec` 重定向到自定义日志 |

### Conda 激活

SLURM 环境下不要依赖 `conda activate`，优先使用绝对路径：

```bash
# HPC2
source /hpc2hdd/home/jzhu997/miniforge3/envs/<env_name>/bin/activate

# HPC3
source /data/user/jzhu997/envs/<env_name>/bin/activate
# 或 /data/user/jzhu997/miniforge3/envs/<env_name>/bin/activate
```

### 工作目录

`-D` 就是作业运行时 `$PWD`，`./logs` 等相对路径都以此为基准。

---

## 5) 常用命令

```bash
# 提交
sbatch job.sh

# 查询队列
squeue -u jzhu997

# 任务详情
scontrol show job <job_id>

# 取消任务
scancel <job_id>

# 历史
sacct -u jzhu997 --format=JobID,JobName,State,Elapsed,Start,End
```

### HPC2 命令找不到时

```bash
# 临时补 PATH
export PATH=/opt/slurm/bin:$PATH

# 或直接用绝对路径
/opt/slurm/bin/sbatch job.sh
/opt/slurm/bin/sinfo
```
