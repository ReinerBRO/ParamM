# Direction3 Router Online Integration

## CLI Switches
- `--router_enable`: enable router-guided retrieval in DoT stage2.
- `--router_conf_threshold` (default `0.6`): minimum confidence required to trust router output.
- `--router_ckpt_path` (default empty): path to router checkpoint (for `memory_router.infer_router.infer`).

## Runtime Behavior
- Router logic is only attempted in stage2 retrieval when all are true:
  - `--router_enable` is set,
  - checkpoint path exists,
  - router infer module import succeeds.
- Router state includes at least:
  - problem `prompt`,
  - current reflection round,
  - feedback statistics (`attempt_count`, `failure_count`, recent `test_feedback`),
  - retrieval candidate count.
- Candidate ranking uses B-lite fusion:
  - `score = w0 * prompt_sim + w1 * reflection_sim - w2 * negative_penalty`
  - where `w0/w1/w2` come from router `router_mix`.

## Fallback Rules
- Fallback to the original fixed strategy if:
  - router is disabled/unavailable,
  - router inference throws an exception,
  - `router_conf < router_conf_threshold`.
- Original fixed strategy remains:
  - reflection-sim top1 when reflection embeddings are available,
  - otherwise prompt-sim top1.

## Stage2 Log Fields
Each item in stage2 log output contains:
- `router_mix`: 3 fusion weights used for selection (or default `[0.0, 0.0, 0.0]`).
- `router_conf`: router confidence for latest selection decision (or `0.0`).
- `fallback_flag`: `true` when fixed strategy was used; `false` when router-guided fusion was used.
