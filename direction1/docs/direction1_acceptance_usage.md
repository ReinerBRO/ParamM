# Direction1 Acceptance Usage

This document describes the minimal A/B acceptance workflow for Direction1 (ToDo8).

## Prerequisites

- Two completed run directories, each containing `first_stage_log.jsonl` and optional `second_stage_log.jsonl`.
- One run produced with `phase2_search_mode=linear`.
- One run produced with `phase2_search_mode=search`.

## One-Command Acceptance

Run from any location:

```bash
bash /data/user/user06/ParamAgent/direction1/scripts/run_direction1_acceptance.sh <linear_run_dir> <search_run_dir> <dataset_name>
```

- `dataset_name` must be `humaneval` or `mbpp`.

Example:

```bash
bash /data/user/user06/ParamAgent/direction1/scripts/run_direction1_acceptance.sh \
  /path/to/run_linear \
  /path/to/run_search \
  humaneval
```

## Generated Outputs

All outputs are written under:

`/data/user/user06/ParamAgent/direction1/results/ab_reports/`

- `<linear_run_name>.json`: A/B metrics report for linear run.
- `<search_run_name>.json`: A/B metrics report for search run.
- `acceptance_<dataset>.json`: Final acceptance decision and reasons.

## Acceptance Logic

The acceptance script checks Direction1 thresholds:

- Search Phase2 absolute gain over linear is at least 1.5 points.
- Token growth is at most 25%.
- Cost growth is at most 25%.
- Search required fields coverage (`search_nodes_expanded`, `verifier_cost`, `best_score_trace`) is complete.
