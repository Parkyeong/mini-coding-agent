# Mini Coding Agent — Architecture (v4)

> v3 的实施步骤已完成并经多轮重构演进至此。本文档描述**当前**的架构、数据流、关键设计决策和已知局限。代码部分英文，讲解中文。

---

## 1. 全局观

整个 system 是一个 **plan → execute → verify** 的 orchestrator，跑 MBPP / SWE-bench 这类 benchmark：

- **唯一的 `Agent` 类**（不再有子类），role 是构造参数
- 三个角色：`planner` / `coder` / `verifier`
  - planner 和 coder 是 LLM 驱动的 Agent 实例
  - verifier 是**纯 pytest 函数**，不调 LLM
- LLM 走 OpenRouter（OpenAI 兼容）
- 每次实验自包含在 `Execution/<exp_name>/` 目录

```
用户 prompt
    │
    ▼
┌─────────┐  plan_steps  ┌─────────┐  pass/fail  ┌──────────┐
│ planner │─────────────▶│  coder  │◀───────────▶│ verifier │
└─────────┘              └─────────┘             └──────────┘
     ▲                       │  ▲                     │
     │                       │  │ fix_suggestion      │
     │                       │  └─────────────────────┘
     │                       │       (per-step retry)
     │                       ▼
     │                   pytest 跑 → returncode
     │
     └─ failure_context (across-plan replan)
```

**两层反馈循环**（这是核心，下面 §3 详细描述）：

1. **同 step 内 retry**：verifier 输出的 `reason` + `fix_suggestion`（pytest stderr 末 1500 字）拼回 coder 的下一轮 prompt
2. **跨 plan replan**：当前 plan 跑挂后，`failure_context` 喂给 planner 重新规划

---

## 2. 模块图

```
agent.py            唯一的 Agent 类
                    ReAct 循环 + 自动往 event_log 推送 tool_call/tool_result/llm_call/text

llm.py              OpenRouter (OpenAI 兼容) chat()
                    懒加载 client；拦 JSON 解析错误转成 _parse_error 让 LLM 自纠

tools/
  __init__.py       TOOL_DEFINITIONS + execute_tool(name, args, env, memory) dispatcher
  fs.py             read_file / write_file / list_dir / search_in_files / replace_in_file
                    纯函数，只依赖 env，不写 memory
  shell.py          run_command（委托给 env.run_command）
  memory_tool.py    save_memory（唯一吃 memory 的工具）

environment.py      class Environment：有状态 sandbox
                    workspace + safe_path + read/write/list/walk/run
                    将来要换 docker 只动这一个文件

memory.py           EventLog + WorkingMemory + MemoryManager
                    （任务级事件流 + 任务级语义状态 + 跨任务持久化 facts）

metrics.py          per-LLM-call token / latency 累计；summary() 一行字
test_runner.py      pytest subprocess runner（verifier 用）

planner.py          PROMPT + build_input + parse_plan + create_plan(agent, ...)
coder.py            PROMPT + build_input + run_coder(agent, step, memory)
verifier.py         verify(memory) → {passed, reason, fix_suggestion, test_block}
                    纯函数，不 import agent，不调 LLM

main.py             run_task() 编排 plan→exec→verify 循环
                    build_agents(env, memory, metrics) → (planner, coder)
                    interactive CLI

runners/mbpp_task.py
                    单文件 MBPP runner（setup + run + report）
                    实验目录 scoping，三个子命令：setup / run / all

config.py           所有可调实验参数
                    MODEL / MAX_STEPS / MAX_REPLANS / 各种 memory 阈值

Execution/
  <exp_name>/
    single_case_details/
      mbpp_XXXX/
        prompt.md, solution.py, test_solution.py, memory.json
    mbpp_global_facts.json
    mbpp_exp_final_results.json
```

---

## 3. 数据流（重点：反馈循环）

### 3.1 顶层：plan → exec → verify → retry / replan

`main.run_task()` 的伪代码：

```python
for replan in range(MAX_REPLANS + 1):
    plan_steps = create_plan(planner, user_task, memory_context, failure_context)

    for step_idx, step_desc in enumerate(plan_steps):
        for attempt in range(MAX_RETRIES_PER_STEP):

            coder_result = run_coder(coder, current_step, memory)
            verify_result = verify(memory)             # 纯 pytest

            if verify_result["passed"]:
                break
            else:
                # 把 verifier 的反馈拼到 next attempt 的 prompt 前面
                current_step = (
                    f"{step_desc}\n\n"
                    f"Previous attempt failed:\n"
                    f"Reason: {verify_result['reason']}\n"
                    f"Fix suggestion: {verify_result['fix_suggestion']}"
                )

        if not step_passed:
            failure_context = _build_failure_context(...)   # 给 planner 看的
            break  # 出本次 plan 循环，触发 replan
```

### 3.2 反馈层 1：coder 看 verifier 结果

每次 coder 跑完，verifier 立刻跑 pytest 拿 returncode：

```
coder.run(step) → 写代码 → verify(memory)
                              │
                              ├─ pytest -q test_solution.py
                              │
                              ▼
                          {passed: bool,
                           reason: "tests failed (returncode=1)",
                           fix_suggestion: <pytest stderr 末 1500 字>}
                              │
                       挂掉时 ─┴─→ 拼到下一轮 coder prompt 里
```

`fix_suggestion` 装着真实的 pytest 失败堆栈（哪个 assert 挂了、期望值/实际值），coder 下一轮 LLM 调用直接看到。

### 3.3 反馈层 2：planner 看整轮失败

某个 step retry 用尽后，本次 plan 终止，但 task 还没结束——会触发 replan。`failure_context` 包含：

- 上次 plan 的完整步骤
- 在第几步挂的
- 最后 3 次 attempt 的 reason + fix_suggestion

这个 context 喂回 planner 的 prompt（[planner.py](planner.py) 的 `build_input` 第二段就是 `Previous Attempt Failed`），让它换思路出新 plan。

### 3.4 隐式反馈：working memory snapshot

跨 step 之间 `coder.reset_message()` 会清空对话，但 `WorkingMemory` 的 event_log 持久化到任务结束。每次 coder 启动前，`coder_role.build_input(step, memory)` 会把 working memory 的 snapshot 拼到 prompt 前面：

```
[WorkingMemory] task_id = mbpp_0011_0001
Current Plan:
 - 1. Read prompt
 - 2. Implement function
 - 3. Verify with pytest
Recent observations from earlier steps:
 - [read_file] solution.py: """MBPP task — implement..."""
 - [list_dir] .: [FILE]prompt.md ...
Files changed so far: ['solution.py']

current step: <当前 step 描述>
```

让 coder 知道"我上一步读过哪些文件、改过哪些代码"，避免重复劳动。

### 3.5 Agent loop（agent.py 内部）

每个 LLM 驱动的角色都跑这个循环：

```python
def run(input_text):
    messages.append({"role": "user", "content": input_text})

    for step in range(max_steps):
        response = llm.chat(messages, system_prompt, tools)

        # 自动落事件
        event_log.append("llm_call", {role, tokens, latency})
        if response.text:
            event_log.append("text", {content})

        if not response.tool_calls:
            return {text, completed: True}     # 干完了

        for call in response.tool_calls:
            event_log.append("tool_call", {name, args})
            result = execute_tool(call.name, call.args, env=self.env, memory=self.memory)
            event_log.append("tool_result", {name, args, result})

        messages.extend(tool_result_messages)

    return {text: "max step reached", completed: False}
```

工具不写 memory，agent loop 写。tool 函数纯净。

---

## 4. Memory 模型

三层结构，**生命周期不同**：

```
                  生命周期           谁写              谁读
                ─────────────────────────────────────────────────────
EventLog        per-task           agent loop         memory 派生属性
                （任务结束清空）   （tool_call/result   working.observations
                                   /llm_call/text）   working.files_changed

WorkingMemory   per-task           save_memory tool    Coder snapshot
                （任务结束 promote  （candidate_facts）  Verifier 读 files_changed
                 或丢弃）          set_plan(...)

Long-term       persistent          end_task() 时       Planner 注入 prompt
(MemoryManager) （写盘）            promote 通过的      （get_context_for_planner）
                                    candidate facts
```

### 4.1 EventLog

`memory.EventLog` 就是个 append-only 列表，每条 `{kind, payload, ts}`。

事件 kind：
- `llm_call`: 每次 LLM 往返
- `text`: LLM 输出的最终文本
- `tool_call`: agent 决定调一个工具
- `tool_result`: 工具返回

`WorkingMemory.files_changed` 和 `observations()` 都是从 event_log **派生**的属性，不再是字段。这样 tool 函数完全不需要 import memory——agent loop 拿到结果后自动 push 事件，谁需要谁订阅。

### 4.2 Candidate fact 的生命周期

```
LLM 调 save_memory(fact, category)
        │
        ▼
WorkingMemory.candidate_facts (per-task list)
        │
        │ task 跑完
        ▼
end_task(passed=True)?
        │
        ├─ 是 → 遍历 candidate_facts → _promote_fact()
        │       ├─ fact 已存在 → confidence += 0.2 (cap 1.0), reinforce_count +=1
        │       └─ fact 不存在 → 插入 (confidence=0, reinforce_count=0)
        │
        └─ 否 → candidate_facts 全部丢弃（保守策略，见 L2）

每次 end_task 后：
        _evict_facts_if_needed(current_task_idx)
            按 (in_grace_period, score, age) 排序，超过 MAX_MEMORY_FACTS 时淘汰
```

Score = confidence × reinforce_count。Grace period 保护新 fact 不被立即淘汰。

### 4.3 双轨模式（runner 用）

`MemoryManager(memory_file=..., global_facts_file=...)`：

- `memory_file`：per-instance（task_history + project_context）
- `global_facts_file`：跨 instance 共享（facts 累积，让 reinforce 真的能跑起来）

runner 把 `global_facts_file` 指向 `Execution/<exp>/mbpp_global_facts.json`，每个实验有独立的 facts 累积。

---

## 5. Runner（实验目录布局）

每次实验自包含在一个文件夹：

```
Execution/
  <exp_name>/                         ← 实验名（CLI 参数 --exp）
    single_case_details/              ← 所有 case workspace 在这里
      mbpp_0011/
        prompt.md                     ← 题面 + tests
        solution.py                   ← agent 填这个
        test_solution.py              ← 由 test_list 生成
        memory.json                   ← per-instance task_history
      mbpp_0012/
        ...
    mbpp_global_facts.json            ← 这次实验的累积 facts
    mbpp_exp_final_results.json       ← 结构化 report
```

### 5.1 子命令

| 命令 | 干什么 |
|---|---|
| `setup --exp NAME [--subset sanitized\|full] [--split test\|train\|...] [--limit N]` | 从 HuggingFace 下载 MBPP，物化 N 个 case 到 `single_case_details/` |
| `run --exp NAME [--limit N]` | 跑 agent 过所有 (或前 N 个) 物化的 case，写报告 |
| `all --exp NAME ...` | 上面两个连起来 |

### 5.2 报告格式

`mbpp_exp_final_results.json`：
```json
{
  "experiment": "baseline",
  "model": "openai/gpt-4o-mini",
  "started_at": "2026-04-27T...",
  "finished_at": "2026-04-27T...",
  "totals": {"total": 10, "passed": 7, "failed": 2, "crashed": 1, "other": 0},
  "results": [
    {"instance": "mbpp_0011", "status": "passed", "attempts": 3,
     "files_changed": ["solution.py"], "result_text": "..."},
    ...
  ]
}
```

### 5.3 终端输出（极简化后）

```
========== mbpp_0011 ==========
Plan: 4 steps
  step 1/4: PASS
  step 2/4: PASS
  step 3/4: FAIL (tests failed (returncode=1)) — retry
  step 3/4: PASS (retry 1)
  step 4/4: PASS
PASSED (5 attempts, 0 replans)
metrics: 8 calls, 2340/680 in/out tokens, 5.1s
```

不再打印 plan 全文 / tool 调用 / coder LLM 中间步骤——这些都在 event_log 里，要查直接看 `memory.json`。

---

## 6. 关键设计决策

### 6.1 为什么用 unified Agent，不是子类

旧版有 `BaseAgent` + `Coder(BaseAgent)` + `Planner(BaseAgent)` + `Verifier(BaseAgent)`。每个子类只是为了塞个不同的 prompt 或加个 parser，loop 逻辑都一样。

现在：**一个 `Agent` 类，role 是构造参数**。

```python
planner = Agent(role="planner", system_prompt=PLANNER_PROMPT, max_steps=1)
coder   = Agent(role="coder",   system_prompt=CODER_PROMPT, tools=Tools.get_tools(),
                env=env, memory=memory)
```

好处：
- 三个角色吃同一份 event_log / metrics / 错误处理
- 以后想给 planner / verifier 加工具就改个参数
- 概念上只有一种 "agent"

### 6.2 为什么 event_log 是一等公民

旧版 tool 通过模块全局 `_memory_manager` 偷偷写 working memory（observations / files_changed）。**两个模块隐式耦合**，tool 不可单独测试。

现在：tool 纯函数式，agent loop 在每次 tool_call/result 后**显式** push 到 event_log，working memory 的 `observations()` / `files_changed` 都从 event_log 派生。

好处：
- `tools/` 整个目录不 import memory
- 同一份事件流可以喂 metrics / debug log / working memory 不同消费者
- `Execution/<exp>/single_case_details/mbpp_XXXX/memory.json` 里完整保留事件序列，事后能复盘整个 task

### 6.3 为什么 environment 抽出来

旧版 tool 直接 `subprocess.run(cwd=config.WORKSPACE)`，test_runner 也直接 subprocess。两条独立的"沙箱执行"路径。

现在：`Environment` 是个有状态对象，封装 `workspace + sandbox kind`，所有物理操作都走它。

好处：
- 将来上 docker / 远程沙箱，只动 `environment.py` 一个文件
- tool 函数 signature 干净：`def read_file(env, file_path)` 显式依赖

### 6.4 为什么 verifier 不调 LLM

旧版 verifier 跑完 pytest 还要把结果喂给 LLM 做 PASSED/FAILED 判断。对 MBPP 这种 benchmark：

- pytest returncode == 0 就是过，没什么好"判"的
- 多一次 LLM 调用 = 多 token / 多延迟 / 多噪声 / 多误判风险

现在：`verifier.verify(memory)` 是纯函数，看 `result.passed()`，挂时把 stderr 末 1500 字塞 `fix_suggestion` 给 coder。

### 6.5 为什么 experiment-scoped 目录

让每次实验**自包含**：

- 一个文件夹 = 一次实验的所有产物（cases / facts / report）
- 删一个文件夹整个实验就没了，可复现
- 不同实验的 facts 独立累积，方便做 "with memory vs without memory" 这类对比

### 6.6 为什么把 setup 和 run 分开（即使有 `all` 子命令）

可以**复用同一份物化的 instance** 跑多个不同的实验：

```bash
# 物化一次
python -m runners.mbpp_task setup --exp dataset_v1 --limit 50

# 跑实验 A
python -m runners.mbpp_task run --exp dataset_v1     # 用默认配置

# 修改 prompt / model / 任何 config，开新实验
cp -r Execution/dataset_v1 Execution/exp_a_new_prompt
python -m runners.mbpp_task run --exp exp_a_new_prompt
```

或者反复跑 setup 同步新 split / 新 subset，run 不变。

---

## 7. 已知局限（保留自 v3，仍然适用）

> 这些不是 bug，是**有意接受**的工程取舍。未来扩展时再回头处理。

### L1. Memory：没有减分机制（no demotion）

fact 一旦进 long-term，confidence 只能 +0.2，永远不会下降。错的 fact 只能等 evict 策略被动清除。**风险**：错 fact 被反复 reinforce 到 1.0，比真实 fact 排前面。

**为什么接受**：主动减分需要"凭什么减"的归因能力，复杂度大。当前依赖被动淘汰 + 失败时丢弃 candidate 作为补偿。

### L2. Memory：task 失败时 candidate facts 全部丢弃

不区分这条 fact 跟失败有没有关系。**风险**：因为算错边界值挂的任务，把同次学到的好 fact（"项目用 pytest"）一起丢了。

**为什么接受**：错杀代价低（重学一次），错放代价高（错 fact 进 long-term 没法纠正，见 L1）。

### L3. Memory：没有相关性检索

Planner / coder 取 facts 只按 confidence 排序 top-N，不做关键词或 embedding 筛选。**风险**：随 facts 累积，注入 context 的信噪比下降。

### L4. Memory：fact 质量完全依赖 LLM 自觉

`save_memory` 工具就是个无脑写入接口。LLM 可能存任务细节级别的 fact（`"user_id=42 是 admin"`），也可能完全不调。

**为什么接受**：当前用 prompt 工程教 LLM "只存可泛化经验"，是软约束。

### L5. TestRunner：marker 探测只支持 Python

只识别 `pytest.ini` / `pyproject.toml` / `tests/`。**为什么接受**：MBPP + SWE-bench 都是 Python，YAGNI。

### L6. TestRunner：不扫 workspace 根目录的 `test_*.py`

故意不扫，避免一个 stray 测试文件误触发 pytest。是项目目录组织规约的一部分。

### L7. SubprocessRunner 是唯一后端

没有 docker / k8s / 远程 runner 抽象。**为什么接受**：SWE-bench 的 docker 用法是把 `docker run ...` 嵌进 `test_command` 字符串，subprocess 一样跑。真要远程隔离时再抽 `BaseRunner` 接口。

### L8. Verifier 只跑 pytest

挪出 LLM judge 之后，verifier 完全相信 pytest returncode。**风险**：测试本身有 bug 时无法察觉（但 MBPP / SWE-bench 的测试是数据集自带，不是 agent 写的，所以这风险很小）。

---

## 8. 后续可扩展点（不在当前范围）

| 方向 | 在哪加 | 收益 |
|---|---|---|
| **Prompt caching** | `llm.py` chat 参数 | 重复 system prompt 节省巨量 token，跑 batch 是 30%+ 成本下降 |
| **持久化 transcript** | `MemoryManager.end_task` | 现在 event_log 任务结束就丢，落盘后能事后排查任意 case |
| **Async 批量** | runners 层 | MBPP 257 题串行很慢，改并行至少 5x |
| **Docker sandbox** | `environment.py` | SWE-bench 必需，每个 instance 自带 deps |
| **Single-loop role** | 多写一个 `system_prompt` + 一个 `build_agents` 变体 | 跟 plan→verify 多角色对比，做消融 |

这些都是**扩展**，不是 rewrite——当前结构已经足够干净，加任何能力都不需要动核心。
