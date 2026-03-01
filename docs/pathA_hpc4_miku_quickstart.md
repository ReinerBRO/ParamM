# Path A (HumanEval) on HPC4 `miku`

## 1) Build environment (login node)

```bash
cd /data/user/user06/ParamAgent
bash scripts/hpc4/setup_env_paramagent.sh paramm
```

## 2) Full run (164 samples) on `miku`

```bash
cd /data/user/user06/ParamAgent
bash scripts/hpc4/run_pathA_full_miku.sh paramm paramAgent_humaneval_llama8b_miku
```

## 3) Result summary

```bash
python scripts/hpc4/check_pathA_results.py \
  /data/user/user06/ParamAgent/results/humaneval/paramAgent/paramAgent_humaneval_llama8b_miku
```

## Notes

- Relay endpoint defaults to `https://api.zhizengzeng.com/v1`.
- If `OPENAI_API_KEY` is missing, scripts fall back to `ZZZ_API_KEY_2`.
- `yui` is intentionally not used in any script.
