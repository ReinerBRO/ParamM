# Direction3 Ensemble Innovation - Breakthrough Architecture

## Executive Summary

**Innovation**: Multi-Strategy Ensemble Phase2 (bypasses broken router)
**Status**: Running 24-worker full validation
**Run**: `dir3_ensemble_mbpp_20260306_000311`
**Expected Impact**: +5-10 problems (target >75.31%, ~298+ solved)

## Critical Bottleneck Analysis

### Current State (architer7-8)
- **Accuracy**: 73.55% (292/397) - stuck at 72-75% range
- **Phase1**: 261/397 (65.74%)
- **Phase2**: Only +31 problems (+7.81%)
- **Router Performance**: BROKEN
  - Always returns confidence=1.0
  - Only 3/137 solves (2.2% success rate)
  - Fallback solves 260/263 (98.9% of all solves)

### Root Causes Identified
1. **Router is overtrained/broken**: 68.5% val accuracy, always outputs conf=1.0
2. **Phase2 adds almost nothing**: Only 3 new solves from 136 failed problems
3. **Memory bank retrieval is weak**: Even fallback only helps marginally
4. **No diversity in phase2**: Single-path reflexion with weak memory augmentation

## Breakthrough Innovation: Multi-Strategy Ensemble

### Core Idea
Instead of relying on broken router, generate solutions from **3 parallel strategies** and pick best by public test feedback:

1. **Strategy A: Pure Reflexion** (no memory)
   - Baseline strong approach
   - No retrieval overhead
   - Good for problems where memory is misleading

2. **Strategy B: Prompt-Similarity Retrieval**
   - Use closest successful examples by embedding similarity
   - Current fallback approach
   - Good for similar problem patterns

3. **Strategy C: Feedback-Guided Retrieval** (NEW)
   - Find examples that fixed similar test failures
   - Extract error patterns (assert, type, index, timeout, value)
   - Score: 60% prompt similarity + 40% feedback pattern match
   - Good for debugging-style problems

### Ensemble Selection
- Generate from all 3 strategies
- Test each on public tests
- Pick best by `_select_candidate_index_by_feedback`:
  1. Priority: any passing candidate
  2. Fallback: fewer failed asserts

### Key Advantages
1. **No ML overhead**: Bypasses broken router completely
2. **Diversity**: 3 different retrieval strategies
3. **Strong signal**: Public test feedback is reliable
4. **Graceful degradation**: If one strategy fails, others compensate

## Implementation Details

### Modified Files
- `/data/user/user06/ParamAgent/direction3/paramAgent.py` (deployed)
  - Replaced `select_stage2_traj` (138 lines) with `select_stage2_traj_ensemble` (68 lines)
  - Added 3-way ensemble generation logic (100 lines)
  - Removed all router inference code
  - Added feedback-guided retrieval strategy

### Key Code Changes

```python
def select_stage2_traj_ensemble(
    retrieval_pool, curr_prompt_emb, reflection_text, 
    feedback_history, strategy="prompt_sim"
):
    """
    Multi-strategy retrieval (no broken router).
    Strategies: "none", "prompt_sim", "feedback_guided"
    """
    if strategy == "feedback_guided":
        # Extract error patterns from recent feedback
        recent_feedback = " ".join(feedback_history[-3:]).lower()
        error_keywords = extract_error_patterns(recent_feedback)
        
        # Score by feedback pattern + prompt similarity
        for traj in retrieval_pool:
            feedback_score = count_matching_patterns(traj, error_keywords)
            prompt_score = dot_similarity(traj, curr_prompt_emb)
            combined = 0.6 * prompt_score + 0.4 * feedback_score
        
        return best_trajectory
```

### Ensemble Generation Logic

```python
# Generate from 3 strategies
for strat_name, strat_traj in [("none", None), ("prompt_sim", traj1), ("feedback_guided", traj2)]:
    strategy_impl = gen.func_impl(strategy_prompt, model, "simple", temperature=1.0)
    strat_passing, strat_feedback = exe.execute(strategy_impl, tests_i)
    ensemble_candidates.append(strategy_impl)
    ensemble_feedbacks.append(strat_feedback)

# Pick best by public test feedback
best_idx = _select_candidate_index_by_feedback(
    [is_passing for is_passing in ensemble_feedbacks],
    ensemble_feedbacks
)
cur_func_impl = ensemble_candidates[best_idx]
```

## Validation Plan

### Current Run
- **24 workers** processing 397 MBPP problems
- **Phase1**: Standard DOT with pitfalls (expected ~261 solved)
- **Phase2**: Ensemble innovation (target +35-40 new solves)
- **Total target**: >298 solved (>75.31%)

### Evidence Collection
1. **Ensemble logs**: `[ENSEMBLE] Strategy=X, passing=Y` in worker logs
2. **Strategy distribution**: Count which strategy wins most often
3. **Improvement over baseline**: Compare to architer7-8 (292 solved)
4. **Anti-replay proof**: Timestamp, PID, unique run directory

### Monitoring
```bash
# Monitor progress
RUN_DIR=/data/user/user06/ParamAgent/direction3/results/mbpp_router_runs/dir3_ensemble_mbpp_20260306_000311
tail -f $RUN_DIR/pipeline.log

# Check phase1 completion
find $RUN_DIR/workers -name "first_stage_log.jsonl" | wc -l  # Should be 24

# Check phase2 completion
find $RUN_DIR/workers -name "second_stage_log.jsonl" | wc -l  # Should be 24

# Final results
cat $RUN_DIR/results.txt
```

## Expected Outcomes

### Conservative Estimate
- **Phase1**: 261/397 (same as baseline)
- **Phase2**: +35 new solves (vs +31 baseline)
- **Total**: 296/397 (74.56%)
- **Improvement**: +4 problems over architer7-8

### Optimistic Estimate
- **Phase1**: 265/397 (ensemble helps even in phase1)
- **Phase2**: +40 new solves
- **Total**: 305/397 (76.83%)
- **Improvement**: +13 problems, **BREAKS 75.31% BARRIER**

### Why This Will Work
1. **Diversity**: 3 strategies cover different problem types
2. **No router overhead**: Broken router removed completely
3. **Strong selection**: Public test feedback is reliable signal
4. **Feedback-guided is new**: Never tried before, should help debugging problems
5. **Graceful degradation**: If memory misleads, pure reflexion wins

## Alternative Innovations (Not Implemented)

### Innovation 2: Iterative Memory Distillation
- Phase2 round 1: Use global memory
- Phase2 round 2: Use phase1 successes as fresh memory
- Phase2 round 3: Use phase2 round 1-2 successes
- **Why not chosen**: More complex, requires multiple passes

### Innovation 3: Confidence-Calibrated Router Bypass
- Disable router, use heuristic: >5 test failures → pure reflexion, else memory
- **Why not chosen**: Less sophisticated than ensemble

## Files Modified

```
direction3/
├── paramAgent.py (DEPLOYED - ensemble version)
├── paramAgent_architer9.py (backup of previous version)
├── paramAgent_ensemble.py (development version)
├── scripts/hpc4/
│   └── run_mbpp_ensemble_innovation_miku.sh (NEW)
└── ENSEMBLE_INNOVATION_SUMMARY.md (this file)
```

## Timeline

- **2026-03-05 23:59**: Innovation designed and implemented
- **2026-03-06 00:03**: Full 24-worker run launched
- **2026-03-06 ~01:30**: Phase1 expected completion (90 min)
- **2026-03-06 ~03:00**: Phase2 expected completion (90 min)
- **2026-03-06 ~03:00**: Final results available

## Success Criteria

1. **Primary**: >298 solved (>75.31%) - BREAKS BARRIER
2. **Secondary**: >296 solved (>74.56%) - beats architer4 best
3. **Minimum**: >292 solved (>73.55%) - matches architer7-8

## Contact & Monitoring

- **Run directory**: `/data/user/user06/ParamAgent/direction3/results/mbpp_router_runs/dir3_ensemble_mbpp_20260306_000311`
- **Pipeline log**: `/tmp/ensemble_launch3.log`
- **Pipeline PID**: `$(cat /tmp/ensemble_pipeline3.pid)`
- **Worker count**: `ps aux | grep main_param.py | grep -v grep | wc -l` (should be 24)

---

**Innovation Status**: ✅ DEPLOYED & RUNNING
**Expected Completion**: ~3 hours from launch
**Confidence**: HIGH (addresses root cause directly)
