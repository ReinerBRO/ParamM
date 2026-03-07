# HPC4 Output Ownership Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the AI-facing HPC4 guide so any compute-node launch that writes shared project outputs preserves deletable ownership for local `user06`.

**Architecture:** Keep the change documentation-only. Add one mandatory ownership section that defines the invariant, then replace the generic NPU launch example with an owner-safe template that works on nodes such as `eren` where login lands as `root`.

**Tech Stack:** Markdown, shell snippets, existing HPC4 operational conventions

---

### Task 1: Document The Ownership Invariant

**Files:**
- Modify: `/data/user/user06/ParamAgent/docs/HPC4_TEST_环境与任务提交通用指引.md`

**Step 1: Add a mandatory shared-output ownership subsection**

Describe these rules explicitly:
- discover `OWNER_UID` and `OWNER_GID` from `/data/user/user06`
- create result namespace directories with `install -d -o/-g`
- if current UID differs from owner UID, launch via `setpriv`
- never "run first, chown later"
- verify ownership immediately after launch

**Step 2: Preserve existing node inventory edits**

Keep the already-added `mutsumi` and `eren` discovery templates intact. Only add ownership guidance around them.

### Task 2: Replace The Generic NPU Launch Example

**Files:**
- Modify: `/data/user/user06/ParamAgent/docs/HPC4_TEST_环境与任务提交通用指引.md`

**Step 1: Rewrite the generic NPU launch block**

Replace the current bare `ssh -> cd -> conda -> nohup` example with an owner-safe version that:
- computes `OWNER_UID` and `OWNER_GID`
- defines `ensure_owner_dir`
- defines `run_as_owner`
- notes that `eren` normally logs in as `root`
- verifies `RUN_ROOT` ownership after launch

**Step 2: Add failure semantics**

State that AI must abort and report an error if:
- `setpriv` is unavailable while current UID differs from owner UID
- required parent directories cannot be prepared as owner-owned

### Task 3: Verify The Guide Text

**Files:**
- Modify: `/data/user/user06/ParamAgent/docs/HPC4_TEST_环境与任务提交通用指引.md`

**Step 1: Run targeted text checks**

Run:
```bash
rg -n "OWNER_UID|OWNER_GID|setpriv|run first, chown later|eren.*root|install -d" /data/user/user06/ParamAgent/docs/HPC4_TEST_环境与任务提交通用指引.md
```

Expected:
- every required ownership rule appears in the guide

**Step 2: Inspect the rendered section boundaries**

Run:
```bash
nl -ba /data/user/user06/ParamAgent/docs/HPC4_TEST_环境与任务提交通用指引.md | sed -n '90,260p'
```

Expected:
- the new ownership subsection sits near compute-node guidance
- the new NPU launch example is complete and internally consistent
