# HPC4 Output Ownership Design

**Problem**

Some compute nodes, especially `eren`, are reached as `root`. If an AI launches long-running jobs directly under that login, shared output directories under `/data/user/user06/...` become `root`-owned. The local `user06` account can then see the results but cannot reliably delete them.

**Goal**

Any file or directory produced by compute-node jobs in shared project paths must remain deletable by local `user06`. The guide must encode this as a hard requirement for AI agents, not as an optional cleanup step.

**Design**

1. Treat shared-directory ownership as an invariant.
   Files and directories written under `/data/user/user06/...` must match the UID/GID of `/data/user/user06`, discovered dynamically with `stat`.

2. Require owner-safe directory creation before launch.
   AI agents must create the result namespace and `RUN_ROOT` with `install -d -o "$OWNER_UID" -g "$OWNER_GID"` before any worker or orchestrator starts writing.

3. Require privilege drop when remote login identity differs from the shared owner.
   If the current remote UID is not the owner UID, the AI must launch all file-writing processes via `setpriv --reuid/--regid`. On `eren`, this is the normal path because `ssh eren` lands as `root`.

4. Forbid "run first, chown later".
   Detached workers continue creating new files after launcher exit, so post-hoc `chown -R` is not a safe invariant. The guide should state that launch must abort if owner-safe execution cannot be guaranteed.

5. Add a post-launch ownership verification step.
   The guide should require `stat` and shallow `find` checks on `RUN_ROOT` immediately after startup to confirm that the output tree is writable and removable by `user06`.

**Scope**

This change updates only the AI-facing compute-node guide in `/data/user/user06/ParamAgent/docs/HPC4_TEST_环境与任务提交通用指引.md`. It does not yet refactor repository launcher scripts.
