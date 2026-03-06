# ParamAgent 源码流程说明：Phase1 / Phase2（HumanEval / MBPP）

> 位置：`/data/user/user06/ParamAgent/paramAgent.py`
> 目的：只解释你这次实验实际走到的 HumanEval / MBPP 路径（不展开 LiveCodeBench / LeetCode 分支）

## Phase1（第一阶段）在做什么

1. **逐题进入主循环**
   对数据集每道题依次处理。

2. **读取题目字段（HE/MBPP）**
   使用 `prompt / entry_point / test` 作为本题输入信息。

3. **进入外层尝试轮次**
   外层条件：`while cur_pass < pass_at_k and not is_solved`。

4. **准备中间测试集合（本次实际路径）**
   - **HumanEval**：使用 `visible_tests[task_id]['given_tests']`
   - **MBPP**：使用 `gen.internal_tests(prompt, model, 1)` 生成中间测试

5. **初始实现尝试（inner_iter）**
   在 `inner_iter` 预算内，结合 pitfall/mistake insight 生成 `strategy="simple"` 的实现。

6. **两级验证**
   - 先用中间测试执行 `exe.execute(...)`
   - 再用隐藏真实测试执行 `exe.evaluate(...)`
   通过则直接记为 solved，并写入正样本轨迹。

7. **未通过则进入反思迭代（max_iters）**
   - 生成多样化 reflection
   - 每轮最多尝试 2 条 reflection
   - 用 `strategy="reflexion"` 生成新实现并再次执行测试
   - 按测试反馈给候选打分

8. **更新 memory bank 与当前解**
   - 成功轨迹写入 `positive_trajectories`
   - 失败轨迹写入 `negative_trajectories`
   - 若继续迭代，从候选中按分数采样一个作为下一轮当前解（线性推进）

9. **每题结束后持久化**
   回写日志字段，并落盘 `log_path / mem_bank.pkl / failed_probs.pkl`。

---

## Phase2（第二阶段，memory-augmented）在做什么

1. **加载第一阶段结果并做快照**
   读取 `memory_bank` 和第一阶段日志，写 `first_stage_json`。

2. **筛选 Phase2 输入题目**
   只保留第一阶段失败且还没打 `stage2=True` 的题。

3. **可选加载全局 memory**
   若传入 `global_mem_bank_path`，加载全局正样本轨迹。

4. **逐题进入外层尝试轮次**
   仍是 `while cur_pass < pass_at_k and not is_solved`。

5. **准备中间测试集合（本次实际路径）**
   - **HumanEval**：使用 `visible_tests[task_id]['given_tests']`
   - **MBPP**：使用 `gen.internal_tests(prompt, model, 1)`

6. **第一次检索增强 + 初始修复**
   - 检索池：`global_positive_trajectories + memory_bank["positive_trajectories"]`
   - 用 prompt embedding 做最近邻检索（实现里 `k=1`）
   - 构造 `augmented_prompt` 后生成初始修复实现

7. **两级验证**
   - 先中间测试 `exe.execute(...)`
   - 再隐藏测试 `exe.evaluate(...)`
   通过则写正样本并结束该题。

8. **未通过则反思修复（线性）**
   - 进入 `max_iters` 反思循环
   - 每轮最多尝试 2 条 reflection
   - 优先做 reflection 相似检索（`k=1`），否则回退 prompt 相似检索（`k=1`）
   - 生成新实现、执行测试、按反馈打分

9. **写 memory + 选择下一轮当前解**
   - 通过写 `positive_trajectories`，失败写 `negative_trajectories`
   - 若继续迭代，从候选中按分数采样一个作为下一轮当前解（线性推进）

10. **每题结束后回写 stage2 日志并落盘**
    更新 `stage2/is_solved/cost/tokens/...`，写 `second_stage_json`，同步更新 `mem_bank.pkl`。

---

## 一句话结论（原版 ParamAgent，按 HE/MBPP 实际路径）

Phase1 和 Phase2 都是单题线性迭代修复；Phase2 相比 Phase1 的核心增量是“跨样本 memory 检索增强”，不是 beam/tree 搜索。
