# Mini Coding Agent — Architecture

> 本文档描述当前的架构、数据流、关键设计决策和已知局限。代码部分英文，讲解中文。

---

## 1. 全局观

整个 system 是一个 **plan → execute → verify** 的 orchestrator，跑 MBPP / SWE-bench 这类 benchmark：

- **唯一的 `LLMNode` 类**（位于 `llm_node.py`，不再有子类），role 是构造参数
- 概念区分：**LLMNode 是单元（node），engine + 多个 LLMNode + verifier + memory 才是 agent（系统）**
- 五个角色：`planner` / `coder` / `verifier` / `summarizer` / `dedup`
  - planner / coder / summarizer / dedup 都是 LLM 驱动的 LLMNode 实例
    - planner / coder 用主 agent 模型(`gpt-4o-mini`)
    - summarizer 用 `SUMMARIZER_MODEL`(默认 `gpt-4o-mini`)，每个 passed case 末尾跑一次,从全 task trace 提炼 1-2 条 project-level 经验
    - dedup 用 `DEDUP_MODEL`(默认 `gpt-4.1-mini`),merge facts 时判语义等价
  - verifier 是**纯 pytest 函数**，不调 LLM
- LLM 走 OpenRouter（OpenAI 兼容）
- 每次实验自包含在 `Execution/<exp_name>/` 目录

```
用户 prompt
    │
    ▼
┌─────────┐ plan_steps ┌────────┐ pass/fail ┌──────────┐ on pass ┌────────────┐
│ planner │───────────▶│ coder  │──────────▶│ verifier │────────▶│ summarizer │
└─────────┘            └────────┘           └──────────┘         └────────────┘
     ▲                    │  ▲                   │                      │
     │                    │  │ fix_suggestion    │                      │ 1-2 facts
     │                    │  └───────────────────┘                      │
     │                    │       (per-step retry)                       ▼
     │                    ▼                                       candidate_facts
     │                pytest 跑 → returncode                         (per-task)
     │                                                                   │
     └─ failure_context (across-plan replan)                              ▼
                                                                  long_term_memory
                                                                          │
                                              每完成 batch_size 个 case   │
                                                                          ▼
                                                                  ┌─────────┐
                                                                  │  dedup  │ ← 流式判等价
                                                                  └────┬────┘
                                                                       ▼
                                                                 global_facts pool
                                                                 (cap MAX_MEMORY_FACTS)
                                                                       │
                                                                       │ seed for
                                                                       ▼ next batch
                                                                  planner.context
```

**三层反馈循环**：

1. **同 step 内 retry**：verifier 输出的 `reason` + `fix_suggestion`（pytest stderr 末 1500 字）拼回 coder 的下一轮 prompt（§3.2）
2. **跨 plan replan**：当前 plan 跑挂后，`failure_context` 喂给 planner 重新规划（§3.3）
3. **跨 case 经验积累**：passed case → summarizer 提炼 → batch 边界 dedup merge → global pool → 下一 case 的 planner seed（§4）

---

## 1.5 一次 task 走一遍（按时间顺序的叙事）

把"一个 prompt 进来"到"task 结束"的完整路径串起来。各层的角色分工在叙事里就清楚了。

**1. runner 入口**（[runners/mbpp_task.py](runners/mbpp_task.py)）
- 读 `prompt.md` 拿到 user_prompt（不是 terminal 输入）
- 新建 `Environment(workspace, protected_files=["test_solution.py"])`、`MemoryManager(seed_facts=<当前 global pool 快照>)`
- `build_llm_nodes()` 构造三个 LLMNode：planner / coder / summarizer
- 调用 `engine.run_task(prompt, planner, coder, summarizer, memory, metrics)`

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
│  · LLM 决定调 tool(read_file / write_file /                 │
│    run_command / replace_in_file / list_dir /              │
│    search_in_files) —— 不再有 save_memory                   │
│  · tools/ 委托给 env 干物理活(read 真去 open,run 真去        │
│    subprocess)。env 拒写 protected_files(test_solution.py)  │
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

**5. engine 调 summarizer**(只在 `overall_passed=True` 后)
- 构造 case trace markdown(`summarizer.build_case_trace`):prompt + plan + 动作时间线 + 最终 solution.py
- `summarizer.summarize(node, memory, env)` 跑一次 LLM,要求输出 1-2 条 JSON 格式的 project-level fact
- 解析 JSON,逐条调 `working.add_candidate_fact(fact, category)` 写进 candidate_facts
- 解析失败/输出空数组都 fail-soft(不影响 task 通过状态)

**6. engine 结束 task**（`memory.end_task(passed=...)`）
- `_save_working_memory()` —— working_memory.json 落盘（含完整 event_log + plan + candidate_facts）
- 在 task_history 追加一条记录（不含 event_log，避免重复）
- 如果 `passed=True`：candidate_facts → `_promote_fact()` → 升级到 case 自己的 long_term_memory.json
- 如果 `passed=False`：candidate_facts 全部丢弃(summarizer 此时也根本没跑过)
- `_save()` 把 long_term_memory.json 写盘(并发模式不写 global file —— runner 在 batch 边界统一 merge)
- print 结果摘要

### 关键分工：决定 vs 干活

| 层 | 角色 | 干嘛 |
|----|------|------|
| **engine** | 流程总管 = "agent" | 决定 retry / replan / summarize / 结束的时机;把 verifier 反馈塞回 coder 下一轮 |
| **planner / coder** | LLMNode 实例 | "想" —— 出计划、改代码 |
| **summarizer / dedup** | LLMNode 实例(辅助 role) | summarizer 在 case 末尾提炼经验;dedup 在 merge 时判语义等价 |
| **tools / test_runner** | 翻译 / 包装层 | 接 LLM 的话(schema)、包 env 的结果(TestRunResult);不直接做 subprocess |
| **env (Environment)** | 底层执行器 | 真的 `subprocess.run` / `open()` / `os.walk()` —— 全代码库唯一一处 subprocess 调用;`protected_files` 黑名单挡掉对 grading 文件的写 |
| **memory** | 横切层 | 任务级 event_log(LLMNode 循环自动写)+ 持久化 task_history / facts;暴露 add_facts_to_pool / cap_pool 给 runner 做跨 case merge |

读起来一句话："**engine 调度,planner / coder 想,verifier 验,summarizer 总结,dedup 判等价,tools / test_runner 翻译,env 干活,memory 旁观记录**"。

---

## 2. 模块图

### 项目分层(文件按职责归类)

```
Roles (cognitive — 都有 PROMPT,被 LLMNode 包装):
  planner.py        用 LLM 出 plan
  coder.py          用 LLM 写代码 / 调 tools
  verifier.py       (没 LLM,纯 pytest 函数)
  summarizer.py     用 LLM 提炼 1-2 条 project-level 经验(passed case 末尾)
  dedup.py          用 LLM 判 fact 语义等价(merge 时)

Infrastructure (data + orchestration):
  memory.py         EventLog + WorkingMemory + MemoryManager + facts pool helpers
  engine.py         编排 (run_task + build_llm_nodes)
  environment.py    workspace 抽象 + protected_files 黑名单
  llm_node.py       LLM 循环单元
  llm.py            OpenRouter HTTP 客户端
  metrics.py        per-LLM-call token / latency 累计

Tools (LLM 看到的工具,纯翻译层):
  tools/__init__.py  TOOL_DEFINITIONS + execute_tool dispatcher
  tools/fs.py        read_file / write_file / list_dir / search_in_files / replace_in_file
  tools/shell.py     run_command(委托给 env.run_command)
  tools/memory_tool.py  save_memory —— 已不再 import,dead code(保留文件备用)
                        coder 不再写 facts,summarizer 在 case 末尾统一提炼

Runners (CLI 入口):
  runners/mbpp_task.py        主入口(setup / run / all)
  runners/mbpp_html.py        生成 dataset.html(被 mbpp_task run 自动调用,
                              也可单独 `python -m runners.mbpp_html` 重渲染)

Config:
  config.py         所有可调实验参数(MODEL / MAX_STEPS / MAX_REPLANS / DEDUP_MODEL /
                    SUMMARIZER_MODEL / MAX_MEMORY_FACTS 等)
```

**这套分层的核心约束**:
- Roles 都是 cognitive units(LLM call 加薄壳),不直接做 IO
- Infrastructure 不调 LLM,只管数据 / 编排 / 物理操作
- Tools 是 Roles 看到的"翻译层",把 LLM 的工具调用接到 Infrastructure
- 加新 role(比如想加一个 critic 评分):只新增一个 .py(像 summarizer.py 一样),不动 Infrastructure

### 实验产物布局

```
Execution/
  <exp_name>/
    single_case_details/
      mbpp_XXXX/
        prompt.md, solution.py, test_solution.py(read-only)
        working_memory.json      ← 任务级 event_log + plan + candidate_facts
        long_term_memory.json    ← project_context + task_history + 本 case facts
    mbpp_global_facts.json       ← 跨 case 共享 facts pool(LLM dedup 后,cap 40)
    mbpp_exp_final_results.json  ← 结构化报告
    dataset.html                 ← 渲染后的可视化报告(自动生成)
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

facts 现在由独立的 **summarizer 角色** 在 case 结束时统一提炼,不再由 coder 在执行过程中散写。原因见 §6.9。

**因果方向**: task 通过决定 fact 能不能升级,不是 fact 决定 task 能不能通过。summarizer 也只在 `overall_passed=True` 后才被触发,失败 case 完全不进入 fact 生成路径。

```
(task 进行中)
coder 解题, 不再调 save_memory tool —— coder.PROMPT 也不再有"存经验"指令

(verifier 报 overall_passed=True)
        │
        ▼ engine 调 summarizer.summarize(node, memory, env)
summarizer.build_case_trace(memory, env)
   → markdown trace: prompt + plan + 动作时间线 + 最终 solution.py
        │
        ▼ 1 次 LLM 调用 (gpt-4o-mini, max_steps=1, no tools)
LLM 输出 JSON: [{"fact": "...", "category": "..."}, ...]
        │
        ▼ summarizer.summarize() 解析、清理、cap 至 3 条
WorkingMemory.candidate_facts (1-2 条 project-level)

(engine 调 memory.end_task(passed=True, ...))
        │
        ▼ 遍历 candidate_facts → _promote_fact()
        ├─ fact 已存在(_normalize_fact 字面相等) → confidence += 0.2, reinforce_count += 1
        └─ fact 不存在 → 插入 long_term (confidence=0, reinforce_count=0)
        │
        ▼ 写盘 long_term_memory.json
        (并发模式下,global_facts_file=None,这一步不碰共享 pool)

(runner 在 batch 边界)
        每 batch_size 个 case 完成,主线程调 add_facts_to_pool()
        → dedup_node(gpt-4.1-mini)逐条流式判等价
        → cap_pool(MAX_MEMORY_FACTS=40),按 reinforce_count desc 淘汰

(每次 end_task 后)
        _evict_facts_if_needed(current_task_idx) —— per-case 淘汰
            因为单 case 自己 facts 池只有 1-2 条,从未触发(死代码)
            真正的 cap 在 runner batch merge 时做(见 cap_pool)
```

**两层 dedup 注意区分**:
- **case 内部** `_promote_fact`:用 `_normalize_fact`(lowercase + 折叠空白) 字面比对 —— summarizer 单 case 输出 1-2 条,基本不重复,这里抓不到啥
- **跨 case batch merge**:用 `dedup_node`(LLM 判语义) —— 真正的 dedup 在这里发生

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

并发跑 case 时多个 worker 同时往 `mbpp_global_facts.json` 写会触发 race condition（互相覆盖丢数据）。所以 runner 现在采用 **case-local + post-run merge** 模式：

```python
# run_one() — 跑的过程中,每个 case 是 single-file mode
memory = MemoryManager(
    long_term_file=os.path.join(workspace, "long_term_memory.json"),
    working_memory_file=os.path.join(workspace, "working_memory.json"),
    global_facts_file=None,    # 关键:不写共享文件,纯 per-case
)
```

**facts 的两阶段写入：**

1. **跑期(并发安全)**：每个 case 学到的 facts 全部写在自己的 `long_term_memory.json` 里，**完全不碰 `mbpp_global_facts.json`**。worker 之间无共享文件 → 0 锁竞争
2. **跑完后(主线程，单线程)**：runner 构造一个 dedup LLMNode（用 [`config.DEDUP_MODEL`](config.py)，默认 `openai/gpt-4.1-mini`），调 `memory.merge_facts_into_global(local_files, target, dedup_node=...)` 扫描所有 case 的 `long_term_memory.json`，**每条新 fact 让 LLM 判断和池里已有 fact 是否语义等价**：等价就累加 `reinforce_count` / `confidence` / 合并 `reinforced_by`；不等价就当 novel 加进池。最后写一次 `mbpp_global_facts.json`。详见 [`dedup.py`](dedup.py) 的 PROMPT 设计。

**为什么不用字符串 normalize 做 dedup**：之前用过 lowercase + 折叠空白的归一化，实测在 baseline 822 条 raw fact 里只抓到 42 条重复（5%）。绝大多数重复是"同义改写"——`"uses pytest -q"` vs `"tests run quietly via pytest -q"` 这种字面差异大但语义相同的，字符串 normalize 完全无能为力。所以现在直接用 LLM 判，gpt-4.1-mini 一次 ~$0.0002，跑全 baseline 也就 ~$0.30。

**取舍：** 跑期间 case 之间互相看不到对方的 fact —— 即使顺序跑也是这样（之前的 dual-file mode 里跨 case reinforce 在实测数据里 reinforce_count 始终 0，本来就基本不发生）。换来的是并发安全 + 跑期更轻 IO。

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

`mbpp_task` 是主入口，3 个子命令通过 `python -m runners.mbpp_task <子命令>` 调用：

| 命令 | 干什么 |
|---|---|
| `setup --exp NAME [--subset sanitized\|full] [--split test\|train\|...] [--limit N]` | 从 HuggingFace 下载 MBPP，物化 N 个 case 到 `single_case_details/` |
| `run --exp NAME [--limit N] [--workers N] [--skip-existing] [--config c0_baseline\|c1_judge\|c2_planspec\|c3_codespec]` | 跑 agent 过所有(或前 N 个)物化的 case，写报告 + 生成 dataset.html |
| `all --exp NAME ...` | setup + run 连起来(继承两边所有参数) |

另有独立工具：

| 命令 | 干什么 |
|---|---|
| `python -m runners.mbpp_html --exp NAME` | 单独重渲染 dataset.html(`mbpp_task run` 跑完会自动调一次，改了模板想重渲染就手动跑这个) |

**`--workers N`**（默认 4）：并发 worker 数。`run_one()` I/O bound（HTTP + subprocess），用 `ThreadPoolExecutor` 启 N 条线程并行。
- `1` = 顺序跑（debug 用）
- `4` = 默认（~3-3.5x 加速，257 题约 2.5 小时）
- `8` = 激进（~5x 加速，约 1.5 小时；要看 OpenRouter rate limit 余量）

**`--skip-existing`**：跳过已经有 `working_memory.json` 的 case。用于断点续跑 —— 长跑挂了就这样恢复：

```bash
# 第一次跑挂在第 187 题
python -m runners.mbpp_task all --exp baseline --split test --limit 0 --workers 4

# 重跑,跳过 1-187,从 188 开始
python -m runners.mbpp_task run --exp baseline --skip-existing --workers 4
```

### 5.2 报告格式

`mbpp_exp_final_results.json`：
```json
{
  "experiment": "baseline",
  "model": "openai/gpt-4o-mini",
  "started_at": "2026-04-27T...",
  "finished_at": "2026-04-27T...",
  "workers": 4,
  "skipped_existing": [],
  "totals": {"total": 10, "passed": 7, "failed": 2, "crashed": 1, "other": 0},
  "results": [
    {"instance": "mbpp_0011", "status": "passed", "attempts": 3,
     "files_changed": ["solution.py"], "result_text": "..."},
    ...
  ]
}
```

`results` 数组在写盘前按 `instance` 名字排序，所以并发完成顺序不影响输出 → 报告 deterministic，方便不同实验之间 diff。

### 5.3 终端输出（极简化后）

**顺序模式（`--workers 1`）**：每个 case 一个块，看得到细节：

```
========== [3/10] mbpp_0011 ==========
Plan: 4 steps
  step 1/4: PASS
  step 2/4: PASS
  step 3/4: FAIL (tests failed (returncode=1)) — retry
  step 3/4: PASS (retry 1)
  step 4/4: PASS
PASSED (5 attempts, 0 replans)
metrics: 8 calls, 2340/680 in/out tokens, 5.1s
```

**并发模式（`--workers >= 2`）**：多 case 交错，每个 case 完成时打一行结果：

```
[exp=baseline] Running 257 MBPP instances (workers=4) ...
  [  1/257] mbpp_0017: passed
  [  2/257] mbpp_0011: passed
  [  3/257] mbpp_0019: passed
  [  4/257] mbpp_0014: failed
  ...
```

并发模式下 plan / step 细节不打印（多 case 交错会乱），细节都在 `working_memory.json` 的 event_log 里。

不再打印 plan 全文 / tool 调用 / coder LLM 中间步骤——这些都在 event_log 里，要查直接看 `memory.json`。

### 5.4 常用工作流

下面是日常跑批量实验的命令。前提是 `OPENROUTER_API_KEY` 已经 export(没 export 会在 [llm.py](llm.py#L27) 直接报错)。

```bash
# smoke test: 下载 + 跑前 10 个 case
python -m runners.mbpp_task all --exp smoke --limit 10

# 全量 sanitized(427 题)
python -m runners.mbpp_task all --exp full_baseline --limit 0 --split test

# 续跑:第一次跑挂了，跳过已完成的 case 接着跑
python -m runners.mbpp_task run --exp full_baseline --skip-existing --workers 4

# 单独重渲染 HTML(`run` 跑完会自动调一次，模板改了想刷新时再手动跑)
python -m runners.mbpp_html --exp full_baseline
```

#### 常用 flag 速查

| flag | 默认 | 作用 |
|---|---|---|
| `--workers` | 4 | 并行 agent 数 |
| `--batch-size` | 4 | 每 N 个 case 完成后合并一次 facts pool(必须 ≥ workers) |
| `--skip-existing` | off | 续跑：跳过已有 `working_memory.json` 的 case |
| `--limit` | 10 (setup) / 0 (run) | 0 = all |
| `--config` | `c0_baseline` | `c0_baseline` / `c1_judge` / `c2_planspec` / `c3_codespec` |
| `--split` | `train` | MBPP split |

> 做 ablation 对比实验(同一份 case 列表跑多个 config)是偶发场景，已不在主 CLI 里——单独写脚本组合 `setup` / `run` 调用即可。

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

### 6.7 为什么把 `test_solution.py` 设成 read-only

MBPP 的官方 test_list 在 setup 时被写进 `test_solution.py`，agent 拿到的 toolbox 包含 `write_file` / `replace_in_file`，**理论上能改这个文件**。早期实测发现：约 11% 的 case，agent 在 debug 过程中把 test 改成"自己 solution 能过的"版本（"hello world" → "world hello" 这类），verifier 看到 returncode=0 直接判 pass —— 但 solution 实际跑官方 test 不一定过。这把 reported pass@1 往上拉了 ~2-3 个百分点的虚高。

修法：**`Environment` 加 `protected_files` 黑名单**，写文件操作（`Environment.write_file`，所有 fs 工具的 choke point）检测 basename 命中就 raise `PermissionError`。runner 在 `run_one()` 构造 env 时传 `protected_files=["test_solution.py"]`，agent 调 `write_file('test_solution.py', ...)` / `replace_in_file('test_solution.py', ...)` 都会被拦下，工具返回 "Refused: ... do not retry — modify solution.py instead." 给 LLM。

好处：
- **单点防御**：所有 fs 工具走 `Environment.write_file` 一处，加一道 check 就全覆盖
- **零误判**：lock 后 verifier 报的 pass@1 ≡ clean re-grade pass@1，跟 AFlow / 论文里报的 pass@1 直接可比
- **agent 行为更干净**：coder 的 system prompt 加了一段 "test_solution.py is locked"，LLM 一开始就知道，不去白浪费 token 尝试

**已知小破口**（见 L9）：`run_command` 走 shell 仍然能改文件（`echo > test_solution.py`），但 4o-mini 实测里几乎不会主动这么干 —— 它只用专门的 fs 工具。

### 6.8 为什么并发选 ThreadPoolExecutor 而不是 multiprocessing / asyncio

`run_one()` 是经典 I/O bound：每题 ~2 分钟里，95%+ 的时间在等 OpenRouter HTTP 响应或 pytest subprocess。CPU 几乎不动。

| 方案 | 适合不适合 |
|---|---|
| **threads** ✓ | I/O 等待时 Python 释放 GIL，多线程能真并行；每个 case 一个 MemoryManager / Environment 实例，无 GIL 内争抢 |
| processes | I/O bound 场景下进程的隔离收益用不上，反而每进程额外开销（启动、序列化）拖慢 |
| asyncio | 需要把 `requests` 全换成 `httpx`，`subprocess` 全换成 `asyncio.subprocess`，整个调用链都得 async-ify，性价比低 |

实测 4 worker ~3-3.5x 加速、8 worker ~5x 加速，符合 I/O bound 场景的预期（不是线性扩展，因为 OpenRouter 偶尔慢响应让 worker 互相等）。

### 6.9 为什么把"写 fact"从 coder 拆给独立的 summarizer

历史上 coder 有一个 `save_memory` tool,在执行过程中自己调用记录"经验"。这套有两个根本问题:

**问题 1: 视角错配。** coder 是"解题 + 调工具"的角色,它的视角是细节级 ——"我用 `re.search()` 解了这个 regex 题"、"我导入了 `math.pi`"、"我用 set 追踪重复字符"。但 **global facts 池的消费者是 planner**,planner 是"做计划"的角色,它要的是项目级共识 ——"项目用 pytest -q"、"function 签名必须严格匹配"、"test_solution.py 是 read-only"。**写者和读者关心的粒度不在一个 level**。

baseline(217/257 passed)的 220 条 facts pool 里,implementation/functionality 类(细节级)占了 153 条(70%),全是 coder 视角噪音,planner 拿来做计划用不上。

**问题 2: 触发频率失控。** coder 在每个 case 平均调 5-10 次 save_memory,257 题 case 就刷了 ~822 条 raw fact。即使后续做 dedup,池子也很臃肿。更糟的是 coder 在测试通过后还会继续调 save_memory(过度执行),浪费 token。

**修法**:删掉 coder 的 save_memory tool,新增独立的 [`summarizer.py`](summarizer.py) role。summarizer 在 `overall_passed=True` 之后被 engine 调一次,看到 case 的全 trace(prompt + plan + 动作时间线 + 最终代码),用全系统视角(不只 planner 也不只 coder)输出 1-2 条 project-level 经验。**作者就是读者**(都是高层视角),消除视角错配;频率从 ~5/case 降到 1-2/case。

实测一道 case(mbpp_0011) summarizer 输出:
> `[testing] Tests must be run with 'pytest -q test_solution.py' to confirm all assertions pass after implementation.`

干净、project-level、planner 直接能用。

### 6.10 为什么 dedup / summarizer 是 role,不是 memory.py 的一部分

工程上 dedup 跟"global facts pool"紧密耦合(只在 merge 时调),很容易想塞进 memory.py。但**架构分层上它是 role,不是 infrastructure**:

```
Roles (每个都是 PROMPT + helper + 被 LLMNode 包装):
  planner.py / coder.py / verifier.py / summarizer.py / dedup.py
  
Infrastructure (data + orchestration,不调 LLM):
  memory.py / engine.py / environment.py / llm_node.py / llm.py
```

把 dedup 塞进 memory.py 会让 memory 同时承担"数据持久化"和"LLM cognition"两件事。判断标准:**"改 dedup PROMPT 跟 memory 存储格式有关吗?" 没有 → 它们是两层东西**。

当前的取舍:
- 让 memory.py 暴露 `add_facts_to_pool(pool, facts, dedup_node)` 这个接口
- dedup_node 是从外面注入的(由 runner 构造)
- memory 不知道 dedup_node 内部用什么 PROMPT、什么模型,只调它的接口
- 这就是依赖注入(DI)。换 PROMPT、换模型、换实现都不用动 memory.py

summarizer 同理:也是 role 文件,被 engine 调,memory 只接收它的输出(candidate_facts)。

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

### L4. Memory：fact 质量完全依赖 LLM 自觉(已部分缓解)

之前 `save_memory` 是 coder 的工具,LLM 可能存任务细节(`"user_id=42 是 admin"`)。

**当前状态**:save_memory 已经从 coder 移除(§6.9),改成由独立的 summarizer 在 case 末尾从 trace 中提炼。summarizer PROMPT 强约束"必须 project-level、可泛化",输出失控的概率比 coder mid-execution 写小很多。但仍是软约束,本质还是 LLM 自觉。

### L4.1 Memory：跨 case dedup 用 LLM 判,case 内 dedup 仍是字符串归一化

**跨 case (batch merge 时)**:`memory.add_facts_to_pool` 调 `dedup_node`(gpt-4.1-mini)做语义判等价,能抓到 "uses pytest -q" vs "tests run quietly via pytest -q" 这种 paraphrase。这是主要的去重战场。

**case 内 (`_promote_fact` 把 candidate_facts 提升到 long_term 时)**:仍用 `_normalize_fact()`(lowercase + 空白折叠)。

**为什么 case 内不上 LLM**:summarizer 单 case 只输出 1-2 条,内部基本不重复,字符串归一化够用,不值得多花 LLM 调用。真正的 dedup 留给跨 case 的 batch merge。

### L5. TestRunner：marker 探测只支持 Python

只识别 `pytest.ini` / `pyproject.toml` / `tests/`。**为什么接受**：MBPP + SWE-bench 都是 Python，YAGNI。

### L6. TestRunner：不扫 workspace 根目录的 `test_*.py`

故意不扫，避免一个 stray 测试文件误触发 pytest。是项目目录组织规约的一部分。

### L7. Environment 当前只有 subprocess 后端

没有 docker / k8s / 远程 backend。**为什么接受**：SWE-bench 的 docker 用法是把 `docker run ...` 嵌进 `test_command` 字符串，env.run_command 用 subprocess 一样跑。真要每个操作都进容器（包括 coder 的 read_file）时再抽 `DockerEnvironment(Environment)` 子类，覆盖几个方法即可。

### L8. Verifier 只跑 pytest

挪出 LLM judge 之后，verifier 完全相信 pytest returncode。**风险**：测试本身有 bug 时无法察觉（但 MBPP / SWE-bench 的测试是数据集自带，不是 agent 写的，所以这风险很小）。

### L9. `protected_files` 只在工具层拦截，shell 能绕

`Environment.write_file` 拦下了 `write_file` / `replace_in_file` 工具，但 `run_command` 走 shell 还是能改任何文件（比如 `echo "..." > test_solution.py` / `rm test_solution.py`）。

**为什么接受**：
- 4o-mini 实测里从不主动 shell 重定向去改文件，prompt 工程已经教会它"想改文件用 fs 工具"
- 真要补的话，在 verifier grading 前用 prompt.md 重新生成 canonical 版 `test_solution.py` 覆盖一次（双层防御）。先放着，等观测到真的有 case 被 shell 绕过去再加

### L10. 跨 case fact reinforce 在并发模式下不发生

case-local + post-run merge 模式下，跑期间 case A 学到的 fact 不会进入 case B 的 planner 上下文（merge 在所有 case 跑完后才进行）。**实测影响很小**：之前 dual-file mode 的数据里，`reinforce_count` 也几乎全是 0 —— 各个 case 学到的 fact 字面措辞各异，本来 reinforce 就基本不发生。

**修法待定**：如果以后想要"边跑边学"，可以让 merge 在每 N 个 case 完成后增量跑一次，并把 merged 结果广播给后续的 worker。当前 N=∞（只在结束时跑一次），对 pass@1 数字无影响。

---

## 8. 后续可扩展点（不在当前范围）

| 方向 | 在哪加 | 收益 |
|---|---|---|
| **Prompt caching** | `llm.py` chat 参数 | 重复 system prompt 节省巨量 token，跑 batch 是 30%+ 成本下降 |
| ~~Async 批量~~ ✓ 已实现 | `runners/mbpp_task.py` `cmd_run` `ThreadPoolExecutor` | MBPP 257 题 4 worker ~3.5x 加速 |
| **Docker sandbox** | `environment.py` + `test_runner.py` | SWE-bench 必需；新增 `DockerRunner` backend，env 加 docker exec 路径 |
| **Single-loop role** | 多写一个 `system_prompt` + 一个 `build_llm_nodes` 变体 | 跟 plan→verify 多角色对比，做消融 |
| **Runner 抽象层** | 提取 `runners/_base.py`（common run_one/cmd_run/_exp_paths） | 加 SWE-bench / HumanEval 时不需要复制 mbpp_task.py 的样板代码 |
| **真 sandbox** | `environment.py` 加 ulimit / unshare / docker | 当前只是路径作用域，LLM 可任意 `subprocess` 跑 shell；非可信场景需要真隔离 |
| **shell 层 file lock** | `Environment.run_command` 拦截 / verifier grade 前 restore canonical | 闭合 L9 的破口，让 `protected_files` 防御覆盖 shell 路径 |
| **增量 merge** | `cmd_run` 每 N 个 case 完成后跑一次 `merge_facts_into_global` 并广播 | 让并发模式下也能跨 case fact reinforce（L10 修法） |


这些都是**扩展**，不是 rewrite——当前结构已经足够干净，加任何能力都不需要动核心。
