# hpc4-test 通用环境安装与计算节点任务提交指引（NPU/CPU）

本指引是“跨项目可复用”模板，不绑定任何具体仓库名称。  
目标场景：在登录节点搭环境，在计算节点执行 CPU/NPU 任务。

## 1. 环境创建与使用（登录节点）

### 1.1 基本原则
- 环境只在登录节点创建和更新。
- 训练、评测、推理任务都在计算节点执行。
- 路径、环境名、日志目录使用固定约定，便于自动化与多 AI 协作。

### 1.2 一次性初始化
下载环境前，先执行以下网络与源配置：

```bash
# 1) 关闭代理，避免下载走错误出口
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

# 2) Conda 优先使用清华源
source <MINICONDA_HOME>/etc/profile.d/conda.sh
conda config --remove-key channels || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

建议：能用 `conda install` 的包优先用 conda，`pip` 作为补充。

然后再做环境初始化：

```bash
# 1) Ascend 运行时环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) Conda
source <MINICONDA_HOME>/etc/profile.d/conda.sh
conda create -n <ENV_NAME> python=3.10 -y
conda activate <ENV_NAME>

# 3) 基础构建工具
pip install -U pip setuptools wheel
```

建议写入 `~/.bashrc`：
```bash
echo '[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
echo '[ -f <MINICONDA_HOME>/etc/profile.d/conda.sh ] && source <MINICONDA_HOME>/etc/profile.d/conda.sh' >> ~/.bashrc
```

### 1.3 NPU 依赖安装（通用模板）
```bash
conda activate <ENV_NAME>

# 版本要与集群 CANN 兼容（示例）
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.8.0

# 安装项目依赖（按需过滤 CUDA-only 包）
grep -vE '^(nvidia-|triton|bitsandbytes)' requirements.txt > requirements_npu.txt
pip install -r requirements_npu.txt
```

### 1.4 可用性检查
登录节点（通常无 NPU）：
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("npu available on login:", hasattr(torch, "npu") and torch.npu.is_available())
PY
```

计算节点（应可见 NPU）：
```bash
npu-smi info -l
python - <<'PY'
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
PY
```

## 2. 计算节点使用指南（CPU 与 NPU）

### 2.1 先确认资源视图
```bash
sinfo -a -o '%P %a %l %D %c %m %G'
sinfo -N -o '%N %P %t %c %m %G'
scontrol show config | egrep 'GresTypes|SelectType'
```

### 2.1.1 Dev Node 固定信息（本环境）
- Dev Node 资源：`4 张 Ascend 910`，每张卡 `2 个 chip`，合计 `8 个 chip`。
- 调度建议：默认一个任务独占一个 Dev Node。

模板 A（`miku`）：
```bash
DEV_NODE="miku"
ssh "$DEV_NODE" "hostname"

ssh "$DEV_NODE" 'bash -s' <<'EOS'
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH}"
npu-smi info -l | awk -F: '
  /NPU ID/ {gsub(/ /,"",$2); n=$2}
  /Chip Count/ {gsub(/ /,"",$2); c=$2+0; for(i=0;i<c;i++) print n, i}
' | while read -r npu chip; do
  out="$(npu-smi info -t usages -i "$npu" -c "$chip" 2>/dev/null || true)"
  hbm="$(awk -F: "/HBM Usage Rate\\(%\\)/{gsub(/ /,\"\",\\$2);print \\$2;exit}" <<< "$out")"
  util="$(awk -F: "/NPU Utilization\\(%\\)/{gsub(/ /,\"\",\\$2);print \\$2;exit}" <<< "$out")"
  echo "npu${npu}_chip${chip} hbm=${hbm:-NA}% util=${util:-NA}%"
done
EOS
```

模板 B（`yui`）：
```bash
DEV_NODE="yui"
ssh "$DEV_NODE" "hostname"

ssh "$DEV_NODE" 'bash -s' <<'EOS'
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH}"
npu-smi info -l | awk -F: '
  /NPU ID/ {gsub(/ /,"",$2); n=$2}
  /Chip Count/ {gsub(/ /,"",$2); c=$2+0; for(i=0;i<c;i++) print n, i}
' | while read -r npu chip; do
  out="$(npu-smi info -t usages -i "$npu" -c "$chip" 2>/dev/null || true)"
  hbm="$(awk -F: "/HBM Usage Rate\\(%\\)/{gsub(/ /,\"\",\\$2);print \\$2;exit}" <<< "$out")"
  util="$(awk -F: "/NPU Utilization\\(%\\)/{gsub(/ /,\"\",\\$2);print \\$2;exit}" <<< "$out")"
  echo "npu${npu}_chip${chip} hbm=${hbm:-NA}% util=${util:-NA}%"
done
EOS
```

### 2.2 CPU 任务（Slurm 模板）
`slurm/job_cpu.sbatch`：
```bash
#!/usr/bin/env bash
#SBATCH -J cpu_job
#SBATCH -p hpc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH -e slurm/logs/%x_%j.err

set -euo pipefail
cd <PROJECT_ROOT>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <MINICONDA_HOME>/etc/profile.d/conda.sh
conda activate <ENV_NAME>

python <CPU_ENTRY>.py
```

提交与查看：
```bash
sbatch slurm/job_cpu.sbatch
squeue -u $USER
```

说明：
- 当前规则：Slurm 仅用于 CPU 任务申请。

### 2.3 NPU 任务（Dev Node 直连）
NPU 任务统一走 Dev Node 直连，不使用 Slurm：
```bash
ssh "$DEV_NODE"
cd <PROJECT_ROOT>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <MINICONDA_HOME>/etc/profile.d/conda.sh
conda activate <ENV_NAME>

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
nohup bash <RUN_SCRIPT>.sh > logs/job_$(date +%F_%H%M%S).log 2>&1 &
```

## 3. NPU 适配注意事项

### 3.1 依赖与框架
- 不安装 CUDA 专用依赖：`nvidia-*`、`triton`、`bitsandbytes`。
- `torch`、`torch_npu`、`CANN` 版本必须匹配。
- 若代码用了 CUDA 特定算子（例如 flash-attn），需替换为 NPU 可用实现（如 `sdpa`）。

### 3.2 常用环境变量
- `ASCEND_RT_VISIBLE_DEVICES`：进程可见 NPU 列表。
- `ASCEND_DEVICE_ID`：单进程场景指定设备。
- `HCCL_CONNECT_TIMEOUT=1800`：多卡通信超时保护。
- `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`：降低显存碎片风险。

### 3.3 观测与排错
- 拓扑：`npu-smi info -l`
- 单卡用量：`npu-smi info -t usages -i <npu_id> -c <chip_id>`
- 队列：`squeue -u $USER`
- 作业详情：`scontrol show job <jobid>`
- 取消作业：`scancel <jobid>`

若 `npu-smi` 报 `libdrvdsmi_host.so` 缺失，先设置：
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH}
```

## 4. 显存占用探针能力（HBM）

### 4.1 快速查看（即时）
```bash
npu-smi info -l
npu-smi info -t usages -i <npu_id> -c <chip_id>
```

建议重点记录：
- `HBM Capacity(MB)`
- `HBM Usage Rate(%)`
- `NPU Utilization(%)`
- `Aicore Usage Rate(%)`

### 4.2 持续采样（建议）
若项目没有现成脚本，可直接用循环采样：
```bash
while true; do
  date '+%F %T'
  npu-smi info -t usages -i <npu_id> -c <chip_id>
  sleep 10
done | tee logs/npu_probe_$(date +%F_%H%M%S).log
```

若项目已自带探针脚本（例如 `npu.sh`、`run_npu_mem_probe.sh`），优先复用仓库脚本。

### 4.3 与 W&B 同步（可选）
- 建议只上报“每卡 HBM 使用率”这类核心指标，避免日志噪音。
- 指标命名建议统一为：`npu/<card>_hbm_usage_pct`。
- 具体开关变量名以项目实现为准（不同项目前缀可能不同）。

## 5. 推荐执行顺序（给其他 AI 复用）
1. 在登录节点创建并验证环境。  
2. 在计算节点做 1-5 分钟小样本 smoke 测试。  
3. 验证 NPU 可见性、显存探针和日志路径。  
4. 提交正式任务：CPU 走 Slurm，NPU 走 Dev Node 直连。  
5. 监控指标与日志，异常时先降并发或缩 batch。  
