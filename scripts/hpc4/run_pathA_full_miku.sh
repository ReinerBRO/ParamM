#!/usr/bin/env bash
set -euo pipefail

DEV_NODE="miku"
ENV_NAME="${1:-paramm}"
PROJECT_ROOT="/data/user/user06/ParamAgent"
RUN_NAME="${2:-paramAgent_humaneval_llama8b_miku}"

ssh "$DEV_NODE" "bash -lc '
set -euo pipefail
cd \"$PROJECT_ROOT\"

source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate \"$ENV_NAME\"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

set -a
source ./.env
set +a

export OPENAI_BASE_URL=\"\${OPENAI_BASE_URL:-https://api.zhizengzeng.com/v1}\"
if [ -z \"\${OPENAI_API_KEY:-}\" ]; then
  export OPENAI_API_KEY=\"\${ZZZ_API_KEY_2:-}\"
fi

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export UV_THREADPOOL_SIZE=32

python scripts/gen_visible_tests.py
mkdir -p ./results/humaneval/paramAgent

nohup python main_param.py \
  --run_name \"$RUN_NAME\" \
  --root_dir ./results/humaneval/paramAgent/ \
  --dataset_path benchmarks/humaneval_full.jsonl \
  --strategy dot \
  --language py \
  --model llama3_1_8b \
  --pass_at_k 1 \
  --max_iters 5 \
  --inner_iter 5 \
  --use_mistakes \
  --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
  --verbose \
  > ./results/humaneval/paramAgent/${RUN_NAME}_nohup.log 2>&1 &

echo \"Started full Path-A run on ${DEV_NODE}. Log: ./results/humaneval/paramAgent/${RUN_NAME}_nohup.log\"
'"
