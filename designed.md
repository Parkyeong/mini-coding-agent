# Mini Coding Agent — Architecture

> 本文档描述当前的架构、数据流、关键设计决策和已知局限。代码部分英文，讲解中文。

---

## 1. 全局观

整个 system 是一个 **plan → execute → verify** 的 orchestrator，跑 MBPP / SWE-bench 这类 benchmark：

- **唯一的 `LLMNode` 类**（位于 `llm_node.py`，不再有子类），role 是构造参数
- 概念区分：**LLMNode 是单元（node），engine + 多个 LLMNode + verifier + memory 才是 agent（系统）**
- 三个角色：`planner` / `coder` / `verifier`
  - planner 和 coder 是 LLM 驱动的 LLMNode 实例
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

## 1.5 一次 task 走一遍（按时间顺序的叙事）

把"一个 prompt 进来"到"task 结束"的完整路径串起来。各层的角色分工在叙事里就清楚了。

**1. runner 入口**（[runners/mbpp_task.py](runners/mbpp_task.py)）
- 读 `prompt.md` 拿到 user_prompt（不是 terminal 输入）
- 新建 `Environment(workspace)`、`MemoryManager`（加载 long_term_memory.json + global_facts.json）
- 调用 `engine.run_task(prompt, planner, coder, memory, metrics)`

**2. engine 启动 task**
- `memory.begin_task(task_id)` 创建 WorkingMemory（空白）
- `memory.get_context_for_planner()` 从 long-term 取 facts + project_context + 最近 task_history，拼成字符串

**3. planner 出 plan**
- engine 调 `planner.create_plan(user_task, memory_context)`
- planner LLMNode 跑**一次** LLM（`max_steps=1`，无 tools）
- 解析输出成 `plan_steps`（list[str]）
- `working.set_plan(plan_steps)`

**4. engine 逐 step 派给 coder**
每个 step 最多 retry `MAX_RETRIES_PER_STEP` 次：

```
┌─ coder 跑（LLMNode.run，带 tools，max_steps=8）──────────────┐
│  · 收到 step + working memory snapshot 拼好的 prompt        │
│  · LLM 决定调 tool（read_file / write_file /                │
│    run_command / save_memory / ...）                        │
│  · tools/ 委托给 env 干物理活（read 真去 open，run 真去      │
│    subprocess）                                              │
│  · 每次 LLM 调用 / tool_call / tool_result 都自动            │
│    push 到 event_log                                         │
│  · LLM 不再调 tool 时返回                                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ verifier 验证（verify(memory, env)）─────────────────────── ┐
│  · test_runner 决定 test_command（memory hint /             │
│    marker / none 三级 fallback）                            │
│  · test_runner 委托 env.run_command 跑 pytest               │
│  · 包成 verdict dict {passed, reason, fix_suggestion}        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ engine 看 verdict 做决定 ─────────────────────────────────┐
│  · passed → break，进下一 step                              │
│  · failed 但还有 retry 次数 → 把 fix_suggestion 拼进下一轮  │
│    coder 的 prompt（这是 coder 唯一"看到"verdict 的方式）   │
│  · retry 用尽 → 触发 replan，回到第 3 步重新出 plan          │
│  · replan 用尽 → task 整体失败                              │
└────────────────────────────────────────────────────────────┘
```

**5. engine 结束 task**（`memory.end_task(passed=...)`）
- `_save_working_memory()` —— working_memory.json 落盘（含完整 event_log + plan + candidate_facts）
- 在 task_history 追加一条记录（不含 event_log，避免重复）
- 如果 `passed=True`：candidate_facts → `_promote_fact()` → 升级到 long-term facts → 同步写到 global file
- 如果 `passed=False`：candidate_facts 全部丢弃
- `_save()` 把 long_term_memory.json + global_facts.json 写盘
- print 结果摘要

### 关键分工：决定 vs 干活

| 层 | 角色 | 干嘛 |
|----|------|------|
| **engine** | 流程总管 = "agent" | 决定 retry / replan / 结束的时机；把 verifier 反馈塞回 coder 下一轮 |
| **planner / coder** | LLMNode 实例 | "想" —— 出计划、改代码 |
| **tools / test_runner** | 翻译 / 包装层 | 接 LLM 的话（schema），包 env 的结果（TestRunResult）；不直接做 subprocess |
| **env (Environment)** | 底层执行器 | 真的 `subprocess.run` / `open()` / `os.walk()` —— 全代码库唯一一处 subprocess 调用 |
| **memory** | 横切层 | 任务级 event_log（LLMNode 循环自动写）+ 持久化 task_history / facts |

读起来一句话："**engine 调度，planner / coder 想，tools / test_runner 翻译，env 干活，memory 旁观记录**"。

---

## 2. 模块图

```
llm_node.py         唯一的 LLMNode 类（最小 LLM 循环单元）
                    ReAct 循环 + 自动往 event_log 推送 tool_call/tool_result/llm_call/text

llm.py              OpenRouter (OpenAI 兼容) chat()，requests 直接打 HTTP，无 SDK 依赖
                    拦 JSON 解析错误转成 _parse_error 让 LLM 自纠

tools/
  __init__.py       TOOL_DEFINITIONS + execute_tool(name, args, env, memory) dispatcher
  fs.py             read_file / write_file / list_dir / search_in_files / replace_in_file
                    纯函数，只依赖 env，不写 memory
  shell.py          run_command（委托给 env.run_command）
  memory_tool.py    save_memory（唯一吃 memory 的工具）

environment.py      class Environment：workspace 范围限制 + 物理操作原语
                    workspace + safe_path + read/write/list/walk/run_command
                    backend_name 标识后端（"subprocess"），DockerEnvironment 子类可覆盖
                    tools/ 和 test_runner 都通过它做物理操作 —— 唯一 subprocess 调用点
                    （注：当前不是真正的 sandbox，只是路径作用域 + cwd 限制）

memory.py           EventLog + WorkingMemory + MemoryManager
                    任务级事件流 + 任务级语义状态 + 持久化（三文件布局）

metrics.py          per-LLM-call token / latency 累计；summary() 一行字
test_runner.py      test 领域逻辑（test_command 探测 + TestRunResult + 输出截断）
                    底层 subprocess 调用委托给 env.run_command —— docker 化只动 environment.py

planner.py          PROMPT + build_input + parse_plan + create_plan(agent, ...)
coder.py            PROMPT + build_input + run_coder(agent, step, memory)
verifier.py         verify(memory, env) → {passed, reason, fix_suggestion, test_block}
                    纯函数，不 import agent，不调 LLM；通过 env.workspace 找 workspace

engine.py           agent system —— 把多个 LLMNode + verifier + memory 串起来
                    run_task() 编排 plan→exec→verify 循环
                    build_llm_nodes(env, memory, metrics) → (planner, coder)
                    本身是库，没有 main()，不可独立运行

runners/mbpp_task.py
                    单文件 MBPP runner（setup + run + report）—— 唯一 CLI 入口
                    实验目录 scoping，三个子命令：setup / run / all

config.py           所有可调实验参数
                    MODEL / MAX_STEPS / MAX_REPLANS / 各种 memory 阈值

Execution/
  <exp_name>/
    single_case_details/
      mbpp_XXXX/
        prompt.md, solution.py, test_solution.py
        working_memory.json      ← 任务级 event_log + plan + candidate_facts
        long_term_memory.json    ← project_context + task_history + 本 case facts
    mbpp_global_facts.json       ← 跨 case 共享 facts pool（confidence 累积）
    mbpp_exp_final_results.json  ← 结构化报告
```

---

## 3. 数据流（重点：反馈循环）

### 3.1 顶层：plan → exec → verify → retry / replan

`engine.run_task()` 的伪代码：

```python
for replan in range(MAX_REPLANS + 1):
    plan_steps = create_plan(planner, user_task, memory_context, failure_context)

    for step_idx, step_desc in enumerate(plan_steps):
        for attempt in range(MAX_RETRIES_PER_STEP):

            coder_result = run_coder(coder, current_step, memory)
            verify_result = verify(memory, env=coder.env)   # 纯 pytest，env 显式传入

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

### 3.2 反馈层 1：engine 把 verifier 反馈拼回 coder 下一轮 prompt

注意：**coder 不直接看 verifier 输出**。verdict 回到 engine，engine 决定要不要 retry，要 retry 就把 `fix_suggestion` 拼进下一轮 step 描述里。coder 下一轮才"间接"看到失败信息。

```
coder.run(step) → 写代码（通过 tools → env 改 workspace 文件）
        │
        ▼ engine 调
verify(memory, env=coder.env) → test_runner → env.run_command(pytest ...)
        │
        ▼ 返回给 engine
{passed: bool,
 reason: "tests failed (returncode=1)",
 fix_suggestion: <pytest stderr 末 1500 字>}
        │
        ▼ engine 决定
passed=True  → break，进下一 step
passed=False → 把 reason + fix_suggestion 拼成 current_step 给 coder 重试
                ├ "{step_desc}\n\nPrevious attempt failed:\n
                ├  Reason: ...\n
                └  Fix suggestion: ..."
```

`fix_suggestion` 装着真实的 pytest 失败堆栈（哪个 assert 挂了、期望值 / 实际值）。**coder 在下一轮 LLM 调用看到的是 engine 拼好的 prompt**，里面带着上轮的失败摘要。整个反馈链 coder 永远不主动调 verifier，永远不直接读 verdict 结构 —— 反馈完全由 engine 中介。

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

### 3.5 LLMNode loop（llm_node.py 内部）

每个 LLM 驱动的角色（planner / coder）都跑这个循环。LLMNode 是最小可复用单元 —— 不带 role 知识，只跑"喂 prompt → LLM → 调 tool → 循环"。

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

逻辑层面**两层**（Working / Long-term），落盘**三个文件**：

```
                  生命周期            谁写                              谁读
                ──────────────────────────────────────────────────────────────────────
EventLog        per-task            agent loop 自动 push              派生属性的实现
（WM 内部）     （任务进行中）      （llm_call/text/                  working.observations()
                                    tool_call/tool_result 全自动）   working.files_changed

WorkingMemory   per-task            • engine.run_task                 • coder 启动每步前
（runtime 对象） 任务结束写盘         set_plan(plan_steps)               snapshot_for_coder
                后丢弃              • coder LLM 决定调 save_memory    • end_task 读 candidate_facts
                                    tool → add_candidate_fact            决定 promote

Long-term       persistent          end_task() 时                     Planner 注入 prompt
(MemoryManager) （写盘）            promote 通过的                    （get_context_for_planner）
                                    candidate facts + task_history
```

**关键点**：
- `event_log` 是 agent loop 自动推的（产生事件就 push，不需要 LLM 决策）
- `candidate_facts` 是 **coder LLM 主动决策**写的：LLM 看 coder.py 的 prompt 要求，决定要不要调 save_memory tool。tool 本身只是 1 行包装，调用 `working.add_candidate_fact(fact, category)`
- `plan` 是 engine 在 planner 出完计划后主动写的，不是 LLM 直接写

**三个落盘文件**（per benchmark instance）：

| 文件 | 内容 | 何时写 |
|------|------|--------|
| `working_memory.json` | 任务级 event_log + plan + candidate_facts + files_changed | `end_task()` 时（覆盖式） |
| `long_term_memory.json` | project_context + task_history（不含 event_log）+ 本 case 学到的 facts | `end_task()` 时 |
| `<exp>/mbpp_global_facts.json` | 跨 case 共享 facts pool（confidence 累积） | 每个 case `end_task` 时累加 |

event_log 只活在 `working_memory.json` 里，不在 `task_history` 里重复出现，避免数据冗余。

### 4.1 EventLog

`memory.EventLog` 就是个 append-only 列表，每条 `{kind, payload, ts}`。

事件 kind：
- `llm_call`: 每次 LLM 往返
- `text`: LLM 输出的最终文本
- `tool_call`: agent 决定调一个工具
- `tool_result`: 工具返回

`WorkingMemory.files_changed` 和 `observations()` 都是从 event_log **派生**的属性，不再是字段。这样 tool 函数完全不需要 import memory——agent loop 拿到结果后自动 push 事件，谁需要谁订阅。

### 4.2 Candidate fact 的生命周期

注意因果方向：**task 通过决定 fact 能不能升级**，不是 fact 决定 task 能不能通过。fact 只是 coder LLM 主动留的"经验记录"，跟 task 通过与否没有因果关系。

```
（task 进行中）
coder LLM 决定要存经验 → 调 save_memory(fact, category) tool
        │
        ▼ tool 1 行包装
WorkingMemory.candidate_facts (per-task list，加进去)

（task 跑完）
        │
        ▼ engine 调 memory.end_task(passed, ...)
判断 passed？
        ├─ passed=True → 遍历 candidate_facts → _promote_fact()
        │       ├─ fact 已存在 → confidence += 0.2 (cap 1.0), reinforce_count +=1
        │       └─ fact 不存在 → 插入 (confidence=0, reinforce_count=0)
        │
        └─ passed=False → candidate_facts 全部丢弃（保守策略，见 L2）

（每次 end_task 后）
        _evict_facts_if_needed(current_task_idx)
            按 (in_grace_period, score, age) 排序，超过 MAX_MEMORY_FACTS 时淘汰
```

每个 task 内 coder 调 `save_memory` 几次都行（**至少一次**是 prompt 里要求的，但实测 LLM 经常一口气调 3-5 次）。candidate_facts 只是 task 进行中的暂存清单，决定升级/丢弃的关键是 task 整体过没过。

Score = confidence × reinforce_count。Grace period 保护新 fact 不被立即淘汰。

**"fact 已存在" 怎么判断？** 字符串归一化后等值比较 —— 不是语义级 dedup。归一化 = `lower()` + 空白折叠（连续空白都压成单个空格）。所以 `"Use pytest"` / `"use   pytest"` / `"USE PYTEST  "` 视为同一条；但 `"this project uses pytest for testing"` 和 `"Functions are typically tested using pytest"` 字面不同 → 视为两条独立 fact。

实现：

```python
def _normalize_fact(fact: str) -> str:
    return " ".join(fact.strip().lower().split())
```

`add_candidate_fact` 和 `_promote_fact` 都用同一套归一化。这是已知局限（见 L1 / L4）—— 字面相似但不完全相同的 fact 会被当成不同条目存进 long-term。当前依赖 `MAX_MEMORY_FACTS` 上限 + LRU-style eviction 控制噪音；要语义级 dedup 需要 embedding，YAGNI。

### 4.3 三文件布局（runner 用）

`MemoryManager(long_term_file=..., working_memory_file=..., global_facts_file=...)`：

- `long_term_file`（必传）：per-instance long-term —— project_context + task_history + 本 case facts
- `working_memory_file`（可选）：per-instance working memory dump —— 任务结束时把 WM 完整状态落盘
- `global_facts_file`（可选）：跨 instance 共享 facts pool

runner 在 `run_one()` 里给三个文件路径都拼好：

```python
memory = MemoryManager(
    long_term_file=os.path.join(workspace, "long_term_memory.json"),
    working_memory_file=os.path.join(workspace, "working_memory.json"),
    global_facts_file=facts_file,    # Execution/<exp>/mbpp_global_facts.json
)
```

**facts 的双向写入**（dual-file mode）：
- 每个 case 学到的 facts → 同时写 per-case 的 `long_term_memory.json`（按 source_task_id 过滤本 case 自己的）和 `mbpp_global_facts.json`（跨 case 累积）
- planner 启动时拿到的 facts = global file 的全部（让 reinforce 真的能跨 case 发生）

每个实验目录独立 → 不同实验的 facts 互不污染，方便做 "with memory vs without memory" 对照实验。

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
        long_term_memory.json         ← project_context + task_history + 本 case facts
        working_memory.json           ← end_task 时落盘的 WM 快照（event_log + plan + cand. facts）
      mbpp_0012/
        ...
    mbpp_global_facts.json            ← 这次实验的累积 facts（跨 case 共享）
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

### 6.1 为什么用 unified LLMNode，不是子类

旧版有 `BaseAgent` + `Coder(BaseAgent)` + `Planner(BaseAgent)` + `Verifier(BaseAgent)`。每个子类只是为了塞个不同的 prompt 或加个 parser，loop 逻辑都一样。

现在：**一个 `LLMNode` 类，role 是构造参数**。

```python
planner = LLMNode(role="planner", system_prompt=PLANNER_PROMPT, max_steps=1)
coder   = LLMNode(role="coder",   system_prompt=CODER_PROMPT, tools=Tools.get_tools(),
                  env=env, memory=memory)
```

好处：
- 两个角色（planner / coder）吃同一份 event_log / metrics / 错误处理
- 以后想给某个角色加工具就改个参数
- 概念上只有一种"LLM 节点"，role 配置由参数决定 —— prefer composition over inheritance

### 6.1.1 为什么叫 LLMNode 不叫 Agent

历史上 ReAct 风格的 LLM 循环习惯叫 "agent"。但当代框架（LangGraph 等）已经把 "agent" 留给"完整的目标驱动系统"，而把这种 LLM 循环单元叫 **node**。本项目里：

- **LLMNode** = 单元（一次"prompt → LLM → tool 循环"的封装）
- **agent** = 系统（engine.run_task 编排出来的 plan → execute → verify → retry → 学习的整体）—— 是 engine.py 这一层的概念

避免一个词指两种东西，命名跟着新的约定走。

### 6.2 为什么 event_log 是一等公民

旧版 tool 通过模块全局 `_memory_manager` 偷偷写 working memory（observations / files_changed）。**两个模块隐式耦合**，tool 不可单独测试。

现在：tool 纯函数式，agent loop 在每次 tool_call/result 后**显式** push 到 event_log，working memory 的 `observations()` / `files_changed` 都从 event_log 派生。

好处：
- `tools/` 整个目录不 import memory
- 同一份事件流可以喂 metrics / debug log / working memory 不同消费者
- `Execution/<exp>/single_case_details/mbpp_XXXX/memory.json` 里完整保留事件序列，事后能复盘整个 task

### 6.3 为什么 environment 抽出来 + test_runner 走 env

`Environment` 是个有状态对象，封装 `workspace` + 所有物理操作（read/write/list/walk/run_command）。**tools/ 和 test_runner 都通过 env.run_command 跑命令**，全代码库只有一处 `subprocess.run` 调用（在 `environment.py`）。

好处：
- 将来上 docker：实现 `DockerEnvironment(Environment)`，覆盖 `run_command` / `read_file` 等方法。tools / test_runner / verifier 一行不动
- 路径安全（safe_path）只写一遍，所有 fs tool 自动继承
- 加全局行为（log / metrics / 资源限制）只改 env 一处，全代码库生效
- tool 函数 signature 干净：`def read_file(env, file_path)` 显式依赖

### 6.4 为什么 verifier 不调 LLM

旧版 verifier 跑完 pytest 还要把结果喂给 LLM 做 PASSED/FAILED 判断。对 MBPP 这种 benchmark：

- pytest returncode == 0 就是过，没什么好"判"的
- 多一次 LLM 调用 = 多 token / 多延迟 / 多噪声 / 多误判风险

现在：`verifier.verify(memory, env)` 是纯函数，看 `result.passed()`，挂时把 stderr 末 1500 字塞 `fix_suggestion` 给 coder。`env` 显式传入避免读全局 `config.WORKSPACE`。

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

## 7. 已知局限

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

### L4.1 Memory：fact 去重只看字符串归一化（TODO 后续修）

`add_candidate_fact` 和 `_promote_fact` 判断"fact 已存在"是 `_normalize_fact()` 后字符串相等比较 —— 只做 lowercase + 空白折叠。

后果：`"this project uses pytest for testing"` 和 `"Functions are typically tested using pytest"` 字面不同 → 当成两条独立 fact 存进 long-term。实测一个 case 跑下来可能有 5 条 fact，其中 2-3 条语义重复。

**修法待定**：候选方向 (a) 编辑距离 / Jaccard 相似度阈值（轻量），(b) embedding 相似度（准确但每条 fact 多一次 API 调用）。先放着，等 facts 池长到几十条噪音真的影响 planner 时再修。

### L5. TestRunner：marker 探测只支持 Python

只识别 `pytest.ini` / `pyproject.toml` / `tests/`。**为什么接受**：MBPP + SWE-bench 都是 Python，YAGNI。

### L6. TestRunner：不扫 workspace 根目录的 `test_*.py`

故意不扫，避免一个 stray 测试文件误触发 pytest。是项目目录组织规约的一部分。

### L7. Environment 当前只有 subprocess 后端

没有 docker / k8s / 远程 backend。**为什么接受**：SWE-bench 的 docker 用法是把 `docker run ...` 嵌进 `test_command` 字符串，env.run_command 用 subprocess 一样跑。真要每个操作都进容器（包括 coder 的 read_file）时再抽 `DockerEnvironment(Environment)` 子类，覆盖几个方法即可。

### L8. Verifier 只跑 pytest

挪出 LLM judge 之后，verifier 完全相信 pytest returncode。**风险**：测试本身有 bug 时无法察觉（但 MBPP / SWE-bench 的测试是数据集自带，不是 agent 写的，所以这风险很小）。

---

## 8. 后续可扩展点（不在当前范围）

| 方向 | 在哪加 | 收益 |
|---|---|---|
| **Prompt caching** | `llm.py` chat 参数 | 重复 system prompt 节省巨量 token，跑 batch 是 30%+ 成本下降 |
| **Async 批量** | runners 层 | MBPP 257 题串行很慢，改并行至少 5x |
| **Docker sandbox** | `environment.py` + `test_runner.py` | SWE-bench 必需；新增 `DockerRunner` backend，env 加 docker exec 路径 |
| **Single-loop role** | 多写一个 `system_prompt` + 一个 `build_llm_nodes` 变体 | 跟 plan→verify 多角色对比，做消融 |
| **Runner 抽象层** | 提取 `runners/_base.py`（common run_one/cmd_run/_exp_paths） | 加 SWE-bench / HumanEval 时不需要复制 mbpp_task.py 的样板代码 |
| **真 sandbox** | `environment.py` 加 ulimit / unshare / docker | 当前只是路径作用域，LLM 可任意 `subprocess` 跑 shell；非可信场景需要真隔离 |


这些都是**扩展**，不是 rewrite——当前结构已经足够干净，加任何能力都不需要动核心。
