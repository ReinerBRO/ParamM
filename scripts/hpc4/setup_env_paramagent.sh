#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-paramm}"
PROJECT_ROOT="/data/user/user06/ParamAgent"
CONDA_HOME="/data/user/user06/miniconda3"

cd "$PROJECT_ROOT"

# Follow HPC4 guide: disable proxies for package install.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

source "$CONDA_HOME/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | rg -x "$ENV_NAME" >/dev/null 2>&1; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# Path A hard requirements for runtime imports.
python - <<'PY'
import importlib
mods = [
    "openai",
    "jsonlines",
    "numpy",
    "tqdm",
    "termcolor",
    "tenacity",
]
for m in mods:
    importlib.import_module(m)
print("env check OK")
PY

echo "Environment ready: $ENV_NAME"
