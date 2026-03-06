# ParamAgent Chinese Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a Chinese technical report for tomorrow's advisor update strictly grounded in repository files with per-claim evidence paths.

**Architecture:** Read specified docs and code, extract structured facts, compute/confirm results from provided result files, then draft sections A-I with evidence lines after key claims.

**Tech Stack:** Bash, rg, python (for parsing if needed), plain text drafting.

---

### Task 1: Inventory required sources

**Files:**
- Read: `docs/**`
- Read: `direction1/paramAgent.py`
- Read: `direction1/search_engine/*`
- Read: `direction1/scripts/eval_direction1_ab.py`
- Read: `direction1/results/direction1_serial_full24_search_fix13_rerun_20260303_233730_orchestration_summary.json`
- Read: `direction1/results/mbpp/paramAgent/direction1_serial_full24_search_fix13_rerun_20260303_233730_mbpp/results.txt`
- Read: `direction1/results/humaneval/paramAgent/direction1_serial_full24_search_fix13_rerun_20260303_233730_humaneval/results.txt`
- Read: `docs/实验结果数据.md`

**Step 1: List files and confirm presence**
Run: `rg --files docs direction1 | rg 'paper|summary|research|paramAgent.py|search_engine|eval_direction1_ab.py|orchestration_summary.json|mbpp/results.txt|humaneval/results.txt|实验结果数据.md'`
Expected: paths listed above.

### Task 2: Extract paper/project intent from docs

**Files:**
- Read: `docs/**`

**Step 1: Open paper/summary/research docs**
Run: `rg -n "paper|summary|research" docs -S`
Expected: file paths with line numbers.

**Step 2: Capture problem, method, baseline info**
Note key passages with line numbers for section B.

### Task 3: Understand code flow and key modules

**Files:**
- Read: `direction1/paramAgent.py`
- Read: `direction1/search_engine/*`
- Read: `direction1/scripts/eval_direction1_ab.py`

**Step 1: Trace main entry and flow**
Use `rg -n "class|def|main|run" direction1/paramAgent.py` and open relevant sections.

**Step 2: Identify key modules and Phase1/Phase2 behavior**
Open relevant files in `direction1/search_engine/*` and `eval_direction1_ab.py`.

### Task 4: Parse experimental results

**Files:**
- Read: `direction1/results/.../orchestration_summary.json`
- Read: `direction1/results/mbpp/.../results.txt`
- Read: `direction1/results/humaneval/.../results.txt`
- Read: `docs/实验结果数据.md`

**Step 1: Extract HE/MBPP numbers and ParamAgent-plus comparisons**
Use `python - <<'PY'` to parse json/text if needed.

**Step 2: Cross-check with docs/实验结果数据.md**
Record any discrepancies with line numbers.

### Task 5: Draft report sections A-I with evidence paths

**Files:**
- Read: all above sources

**Step 1: Build A (8-12 conclusions) with evidence paths**
Each conclusion ends with `证据路径: file:line/field`.

**Step 2: Build sections B-F**
Ensure all key claims are anchored to evidence paths.

**Step 3: Build sections G-H**
Craft 3-minute and 8-minute scripts and 5 Q&A with evidence references for technical claims.

**Step 4: Build section I**
Provide 1-page PPT skeleton (title + 4 sections).

### Task 6: Final consistency check

**Step 1: Verify every key claim has evidence path**
Use a quick scan before delivery.

**Step 2: Ensure no invented content**
Cross-check any derived statements are clearly marked as inference.

