# Mini Coding Agent — 改造设计文档 v3 (designed.md)

> 本文档是经过多轮讨论之后的最终设计，所有细节已与你确认。
>
> **优先级**：
> - **P0**：Verifier 独立测试环境 + Memory 管理优化（B.2 两层结构）— 核心改动
> - **P1**：能跑 MBPP 数据集 — 本周
> - **P2**：能跑 SWE-bench 数据集 — 周末前
>
> 代码部分全英文，讲解部分中文。Review 完成后可以直接按"实施清单"开工。

---

## 🔧 实操指南（开工前必读）

### Python 版本

确认环境：**Python 3.14.3**。文档里所有 `list[str]` / `tuple[A, B | None]` / `Optional[X]` 等现代类型语法**直接可用**，不需要降级到 `from typing import List, Tuple`。

### 改动顺序（严格按"实施清单"Step 0 → 20）

**不要跳步**。每完成一个 Step 做一次 git commit，改坏了能回滚：

```bash
cd "/home/obob/Research Project/mini coding agent"
git init                       # 如果还没 init
git add -A && git commit -m "snapshot before designed.md v3 refactor"
```

之后每个 Step 完成后：
```bash
git add -A && git commit -m "step N: <短描述>"
```

### 推荐落地节奏

| 顺序 | 动作 | 为什么这个顺序 |
|---|---|---|
| 1 | **Step 0（前置）** — 把 `tool.py` 顶部 `from config import WORKSPACE` 改成 `import config`，所有 `WORKSPACE` 改成 `config.WORKSPACE`。详见 P0.0 章节 | 这是基础设施改动；先做完，后续任何代码都站在正确的地基上。如果先写了 P0 的代码再回头修，会很烦 |
| 2 | Step 1 — 在 `config.py` 末尾追加 verifier sandbox + memory B.2 的常量 | 后面所有模块都依赖这些常量 |
| 3 | Step 2 — 新增 `test_runner.py`（只有 SubprocessRunner，没有 docker 后端） | 简化版直接写一遍，不会返工 |
| 4 | Step 3 — 重写 `memory.py`，**先 `cp memory.py memory.py.bak`** 备份原文件 | 新版有 bug 时能对照旧版 debug |
| 5 | Step 4 — 改 `tool.py` 散点（save_memory / 加 _record_file_change / _record_observation） | 后面的 verifier/coder/main 都依赖 tool 已经能正确写 working memory |
| 6 | Step 5 → 7 — verifier / coder / main 顺序写 | 数据流方向，先底再顶 |
| 7 | **Step 8 — P0 自验证（关键）**：跑一个手工任务，检查 3 件事 | 见下面"自验证检查清单"。任何一项不对，不要往 P1 走 |
| 8 | Step 9 → 14 — P1 (MBPP) | 单独成块，跟 P0 解耦 |
| 9 | Step 15 → 20 — P2 (SWE-bench) | 周末做 |

### 容易踩坑的点（核对清单）

#### 坑 1：Step 0 改完后，grep 自检
```bash
grep -n "WORKSPACE" tool.py
grep -n "WORKSPACE" memory.py
grep -n "WORKSPACE" test_runner.py
```
确认所有引用都是 `config.WORKSPACE`，**没有任何裸的 `WORKSPACE`**。一个遗漏就会让 P1/P2 切 workspace 失效。

为什么必须这样：`from config import WORKSPACE` 是**值绑定**，import 那一刻就把 `WORKSPACE` 的值 copy 到当前模块命名空间。之后 `config.WORKSPACE` 被改写时，已经 import 过的模块**看不到新值**。`runners/run_mbpp.py` 会在跑每条样本前改 `config.WORKSPACE`，所以 `tool.py` 必须用 `import config` + `config.WORKSPACE` 的"现查"写法。

`COMMAND_TIMEOUT` / `PROVIDER` / `MODEL` 这种**不会运行时变化的常量**保持 `from config import` 即可，**只有 `WORKSPACE` 和 `PROJECT_NAME`** 必须动态读。

#### 坑 2：`main.py` 的 `run_task` 必须用 `try / finally` 包起来

P0.2.6 的代码示例已经写了 `try / finally`。这不是装饰，是**必需**：如果中途 LLM 报错或 KeyboardInterrupt，没有 `finally` 的话 `memory.end_task()` 不会被调用，working memory 会**泄漏到下次任务**，下次 task 看到的 plan 和 observations 是上一个失败任务残留的。

#### 坑 3：`tool.py` 的 `_record_observation` 不要给 `run_command` 加 hook

文档 P0.2.4 (c) 已经说了，但很容易写顺手就加上。`run_command` 输出可能巨大且很多噪音（编译输出、依赖安装日志），会塞爆 working memory。**只对 `read_file` / `list_dir` 加 observation hook**。

#### 坑 4：`save_memory` 改写后，旧行为完全消失

旧版 `save_memory` 是直接写 long-term。**新版完全不写 long-term**，只写 working candidate，由 `end_task` 在 task 通过时 promote。如果你在 Step 8 自验证时看到 task 失败但 `memory.json` 的 `facts` 还是新增了，说明 promote 逻辑有 bug，回头查 `tool.py:save_memory` 和 `memory.py:end_task`。

#### 坑 5：`runners/` 目录下放空 `__init__.py`，**不要叫 `datasets/`**

HuggingFace 的 `datasets` 库 `pip install datasets` 装的就是 `datasets` 这个包名，跟本地目录冲突——你执行 `from datasets import load_dataset` 时 Python 会优先找到我们自己的目录，找不到 `load_dataset` 报错。所以数据集相关脚本目录命名为 `runners/`。没有 `__init__.py` 在 Python 3 里也能 import（namespace package），但建议放一个空文件，避免一些 IDE 和 linter 报警。

#### 坑 6：`MemoryManager.__init__` 接受 `memory_file` 参数

新版 `memory.py` 的 `MemoryManager` 必须支持 `memory_file=...` 参数（不是只用 default 的 `WORKSHOP/PROJECT_NAME/memory.json`），因为 P1 的 `run_mbpp.py` 要传 per-instance 的 memory.json 路径。文档 P0.2.3 已经写了，**不要漏**。

#### 坑 7：`Coder` / `Verifier` 构造时必须注入 `memory`

P0.2.6 里 `main()` 改动的部分：
```python
coder    = Coder(metrics_tracker=metrics, memory=memory)
verifier = Verifier(metrics_tracker=metrics, memory=memory)
```
两个 `memory=memory` **不能漏**。漏了的话 Coder 的 `run()` 不会注入 working snapshot，Verifier 不会读 `test_command`，等于改了一半。

### Step 8 自验证检查清单（P0 完成的硬标准）

在 `Execution/mini coding agent/` 现有 workspace 下，跑一个简单任务（比如 "在 README.md 末尾加一行"），然后检查：

| 检查项 | 通过标准 | 不通过怎么办 |
|---|---|---|
| ✅ verifier 输出里有 `[TestRunner]` block | 终端 print 能看到 `[TestRunner] backend=subprocess command=... returncode=...` 这一行 | 检查 `verifier.py` 的 `_maybe_run_tests`、`config.VERIFIER_RUN_TESTS=True` |
| ✅ `memory.json` 的 `task_history` 最后一条 `files_changed` 非空 | 你确实改了文件，task_history 那条 record 的 `files_changed` 数组里能看到该文件 | 检查 `tool.py:_record_file_change` 是否真的被调到、`memory.py:end_task` 是否真的写了 `wm.files_changed` |
| ✅ `facts` 只在 task 通过后才出现 | 跑一个**故意失败**的任务（比如让它改一个不存在的文件），结束后 `memory.json` 的 `facts` 数组**不应该有新增** | 检查 `memory.py:end_task` 里 `if passed and wm:` 这个分支判断 |

3 项全过 → P0 完成，可以进 P1。**任何一项不过，先回头查 bug，不要往下走**。

### 卡住的时候怎么办

随时回来问。给我提供：
1. 你在哪一个 Step
2. 完整的报错 traceback（不要省略）
3. 你改后的相关代码片段（贴到对话里，或者告诉我文件路径让我自己读）

不要只说"跑不通"——信息越具体我越能帮上。

---

---

## 目录

- [0. 总览](#0-总览)
- [P0.1 Verifier + Sandbox 测试环境](#p01-verifier--sandbox-测试环境)
- [P0.2 Memory B.2 — Working + Long-term 两层](#p02-memory-b2--working--long-term-两层)
- [P0.3 files_changed 收集](#p03-files_changed-收集)
- [P1 跑 MBPP 数据集](#p1-跑-mbpp-数据集)
- [P2 跑 SWE-bench 数据集](#p2-跑-swe-bench-数据集)
- [实施清单（分阶段）](#实施清单分阶段)

---

## 0. 总览

### 0.1 当前架构痛点回顾

| 模块 | 痛点 |
|---|---|
| Verifier | 只看 coder 自报文本，没真跑代码，对"functional correctness"基本是猜的 |
| Memory | append-only、无去重合并、无相关性检索；context 注入是"全部 facts + 最近 5 条"，越塞越大；幻觉 fact 直接污染长期记忆 |
| files_changed | `task_history` 永远填 `[]`，假装有但根本没收集 |

### 0.2 改造后的数据流

```
User
 │
 ▼
run_task(task_id) ─ memory.begin_task(task_id) ─ working_memory created
 │
 ├── Planner ──── reads memory.get_context_for_planner() (long-term)
 │
 ├── Coder ────── reads working_memory.snapshot_for_coder()
 │                writes via tool.py (write_file/replace_in_file)
 │                ↳ tool.py records files_changed into _file_tracker
 │                ↳ tool.py records candidate facts into working_memory
 │
 ├── Verifier ─── reads project_context.test_command from memory
 │                runs SubprocessRunner (test_command may be a `docker run ...` string for SWE-bench)
 │                also receives files_changed for scope check
 │                returns verdict + raw test result
 │
 └── memory.end_task(task_id, passed):
        ├── if passed: promote candidate facts → long-term
        ├── else:      discard candidate facts (no pollution)
        └── always:    write task_history record
```

### 0.3 改动文件总览

| 文件 | 动作 | Step 0 | P0 | P1 | P2 |
|---|---|---|---|---|---|
| `tool.py` | **Step 0 前置改造** + P0 散点改 | ✓ | ✓ | | |
| `config.py` | 追加常量 | | ✓ | ✓ | ✓ |
| `test_runner.py` | **新增**（只有 SubprocessRunner） | | ✓ | | |
| `verifier.py` | 重写 | | ✓ | | |
| `memory.py` | 重写 | | ✓ | | |
| `main.py` | 改 | | ✓ | | |
| `coder.py` | 小改 | | ✓ | | |
| `runners/__init__.py` | **新增** 空文件 | | | ✓ | |
| `runners/setup_mbpp.py` | **新增** | | | ✓ | |
| `runners/run_mbpp.py` | **新增** | | | ✓ | |
| `runners/setup_swebench.py` | **新增** | | | | ✓ |
| `runners/run_swebench.py` | **新增** | | | | ✓ |

---

# P0.0 Step 0 前置改造 — `tool.py` 切换到 `import config`

## P0.0.1 为什么这一步必须最先做

`tool.py` 当前顶部是：
```python
from config import WORKSPACE, COMMAND_TIMEOUT, PROVIDER
```

这是 Python 的**值绑定**：import 那一刻就把 `config.WORKSPACE` 当时的值（字符串）copy 到 `tool` 模块自己的命名空间。之后任何代码改 `config.WORKSPACE` 都**不会传播**到 `tool.py`，`tool.py` 内部看到的永远是启动时的旧值。

P1/P2 阶段 `runners/run_mbpp.py` 会在跑每条样本前改 `config.WORKSPACE` 切到对应的 instance 目录，如果 `tool.py` 没改成动态读取，**所有 `tool.py` 的文件操作都会写到错误的 workspace**——你会以为 LLM 笨，其实是 import 机制的坑。

`COMMAND_TIMEOUT` / `PROVIDER` 这种**不会运行时变化的常量**保持 `from config import` 没关系；只有 `WORKSPACE` 和 `PROJECT_NAME` 这种会被切换的才必须动态读。

## P0.0.2 具体改动

打开 `tool.py`，做以下 5 处改动：

```python
# Line 3 - replace this:
from config import WORKSPACE, COMMAND_TIMEOUT, PROVIDER

# with this:
import config
from config import COMMAND_TIMEOUT, PROVIDER
```

然后函数体里所有 `WORKSPACE` 引用改为 `config.WORKSPACE`：

```python
# safe_path() — line 11 and 13
abs_path = os.path.abspath(os.path.join(config.WORKSPACE, path))
workspace_abs = os.path.abspath(config.WORKSPACE)

# run_command() — line 77
result = subprocess.run(command, shell=True, cwd=config.WORKSPACE,
                        capture_output=True, text=True, timeout=COMMAND_TIMEOUT)

# search_in_files() — line 105
relative_path = os.path.relpath(file_path, WORKSPACE)
# becomes:
relative_path = os.path.relpath(file_path, config.WORKSPACE)
```

## P0.0.3 验证 Step 0 是否做完整

```bash
cd "/home/obob/Research Project/mini coding agent"
grep -n "WORKSPACE" tool.py
```

期望：每一行 `WORKSPACE` 都有 `config.` 前缀，**没有任何裸的 `WORKSPACE`**。如果还有裸引用，回去补。

> 注：`memory.py` 和 `test_runner.py` 也需要用 `import config` 的写法，但因为这两个文件是 P0 阶段重写/新建的，在 P0.2.3 / P0.1.3 的代码模板里已经写好，不需要在 Step 0 单独处理。

---

# P0.1 Verifier + Sandbox 测试环境

## P0.1.1 设计要点

1. Verifier 在让 LLM 判断之前，**先用 SubprocessRunner 跑一遍测试命令**，把真实的 returncode/stdout/stderr 拿到。
2. 测试命令来源优先级：
   1. `MemoryManager.project_context["test_command"]`（最可靠，setup 脚本初始化时写入）
   2. marker 自动探测（兜底，给随手跑的 random repo 用）
   3. 跳过执行 → 退化成纯 LLM 判断
3. **只有一个后端：SubprocessRunner**。verifier 不知道也不关心你跑的是 docker 还是 pytest——`test_command` 是什么字符串它就 subprocess 执行什么。
   - MBPP 的 `test_command = "pytest -q test_solution.py"`
   - SWE-bench 的 `test_command = 'docker run --rm -v ... <image> bash -c "pytest ..."'`
   - docker 镜像的 pull / cleanup 由 `runners/setup_swebench.py` 和 `runners/run_swebench.py` 在 instance 生命周期边界负责，不在 verifier 范围内
4. Timeout：v1 全局 `config.VERIFIER_TEST_TIMEOUT_DEFAULT = 60`；P2 在 setup 脚本里写 per-instance 的 `project_context["test_timeout"]` 覆盖默认。
5. Verifier 还要接收 `files_changed`，做 scope 检查（"用户没要求改 X，你为什么改 X？"）。

## P0.1.2 `config.py` 改动

在文件末尾追加：

```python
# === Verifier sandbox config (added in v3) ===
VERIFIER_RUN_TESTS = True              # master switch; False = legacy text-only verification
VERIFIER_TEST_TIMEOUT_DEFAULT = 60     # seconds, can be overridden by project_context["test_timeout"]
VERIFIER_OUTPUT_MAX_CHARS = 4000       # truncate stdout/stderr before sending to LLM
```

讲解：
- `VERIFIER_RUN_TESTS=False` 是消融实验开关，可以 A/B 对比"加沙盒 vs 不加沙盒"对 verifier 准确率的影响。
- 没有 `SANDBOX_BACKEND` / `DOCKER_*` 这种开关——因为只有一个后端。docker 嵌入到 `test_command` 字符串里。

## P0.1.3 新文件 `test_runner.py`

完整代码（注意：用 `import config`，不要 `from config import WORKSPACE`）：

```python
"""
Test runner for the verifier.

Single backend: SubprocessRunner. Runs whatever command is in test_command
(may be a plain `pytest -q` or a `docker run ...` string for SWE-bench).
"""
import os
import subprocess
from typing import Optional

import config
from config import VERIFIER_TEST_TIMEOUT_DEFAULT, VERIFIER_OUTPUT_MAX_CHARS


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------

class TestRunResult:
    def __init__(
        self,
        executed: bool,
        command: Optional[str],
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        timed_out: bool,
        backend: str,                   # "subprocess" | "skipped"
        detection_source: str,          # "memory" | "marker" | "none"
        error: Optional[str] = None,
    ):
        self.executed = executed
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.backend = backend
        self.detection_source = detection_source
        self.error = error

    def passed(self) -> bool:
        return self.executed and not self.timed_out and self.returncode == 0

    def to_prompt_block(self) -> str:
        if not self.executed:
            return f"[TestRunner] skipped (source={self.detection_source}): {self.error or 'no test command available'}"
        head = (
            f"[TestRunner] backend={self.backend} command={self.command} "
            f"source={self.detection_source} returncode={self.returncode} timed_out={self.timed_out}"
        )
        return f"{head}\n--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}"

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "backend": self.backend,
            "detection_source": self.detection_source,
            "error": self.error,
        }


def _truncate(text: str, limit: int = VERIFIER_OUTPUT_MAX_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [truncated {len(text) - limit} chars] ...\n" + text[-half:]


# ---------------------------------------------------------------------------
# Subprocess runner — only backend
# ---------------------------------------------------------------------------

class SubprocessRunner:
    backend_name = "subprocess"

    def run(self, command: str, workspace: str, timeout: int) -> TestRunResult:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=workspace,
                capture_output=True, text=True, timeout=timeout,
            )
            return TestRunResult(
                executed=True,
                command=command,
                returncode=proc.returncode,
                stdout=_truncate(proc.stdout),
                stderr=_truncate(proc.stderr),
                timed_out=False,
                backend=self.backend_name,
                detection_source="",  # filled by run_tests()
            )
        except subprocess.TimeoutExpired as e:
            return TestRunResult(
                executed=True, command=command, returncode=None,
                stdout=_truncate(e.stdout or ""), stderr=_truncate(e.stderr or ""),
                timed_out=True, backend=self.backend_name, detection_source="",
                error=f"timeout after {timeout}s",
            )
        except Exception as e:
            return TestRunResult(
                executed=False, command=command, returncode=None,
                stdout="", stderr="", timed_out=False,
                backend=self.backend_name, detection_source="",
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# Marker-based fallback detection (Python only, per project decision)
# ---------------------------------------------------------------------------

def detect_test_command_by_marker(workspace: str) -> Optional[str]:
    """Last-resort detection. Returns None if nothing matches.

    Note: we deliberately do NOT scan workspace root for stray test_*.py files.
    Per project decision, dataset outputs (MBPP / SWE-bench) live in
    dedicated subfolders, so a loose test_*.py at workspace root must not
    auto-trigger pytest.
    """
    if os.path.exists(os.path.join(workspace, "pytest.ini")):
        return "pytest -q"
    if os.path.exists(os.path.join(workspace, "pyproject.toml")):
        return "pytest -q"
    if os.path.isdir(os.path.join(workspace, "tests")):
        return "pytest -q"
    return None


# ---------------------------------------------------------------------------
# High-level entry point used by Verifier
# ---------------------------------------------------------------------------

def run_tests(
    workspace: str,
    memory_hint_command: Optional[str] = None,
    memory_hint_timeout: Optional[int] = None,
) -> TestRunResult:
    """
    Resolve the test command (memory -> marker -> none) and run it via
    SubprocessRunner. `workspace` must be passed in explicitly by the caller
    (verifier passes config.WORKSPACE) so we always see the current value.
    """
    timeout = memory_hint_timeout or VERIFIER_TEST_TIMEOUT_DEFAULT

    if memory_hint_command:
        command, source = memory_hint_command.strip(), "memory"
    else:
        detected = detect_test_command_by_marker(workspace)
        if detected:
            command, source = detected, "marker"
        else:
            return TestRunResult(
                executed=False, command=None, returncode=None,
                stdout="", stderr="", timed_out=False,
                backend="skipped", detection_source="none",
                error="no test_command in memory and no marker matched",
            )

    runner = SubprocessRunner()
    result = runner.run(command=command, workspace=workspace, timeout=timeout)
    result.detection_source = source
    return result
```

讲解：
- 只有一个 runner 类，没有抽象基类、没有 docker 后端。verifier 不需要知道 backend 类型。
- `run_tests` 的 `workspace` 是**必传参数**（不是 default 值）——这样 verifier 调用时传 `config.WORKSPACE`，每次都现查当前值，符合 Step 0 的动态读取约束。
- SWE-bench 的 docker 用法是把 `docker run ...` 写进 `test_command` 字符串，subprocess 跑这个字符串就行；镜像 pull/cleanup 由 `runners/setup_swebench.py` 和 `runners/run_swebench.py` 负责，不在这里做。
- `_truncate` 防止巨型 log 撑爆 prompt。

## P0.1.4 `verifier.py` 完整重写

```python
from agent import BaseAgent
import config
from config import VERIFIER_RUN_TESTS
from test_runner import run_tests, TestRunResult


VERIFIER_PROMPT = """
You are a code change verifier.

Your job: judge whether the step was completed correctly.

You will receive:
- the original user request
- the step that was being executed
- the coder's self-reported actions and results
- the list of files the coder actually modified during this step
- (optionally) a [TestRunner] block with the REAL output of running the project's tests

Trust the [TestRunner] block over the coder's self-report when they disagree.

Check these four things:
1. Functional correctness: did the tests actually pass? (returncode==0, no failures in stderr)
2. Intent alignment: does the change actually address what the user asked for?
3. Scope: are the modified files reasonable for the requested change?
   Flag any files modified that look unrelated to the request.
4. Side effects: anything in stderr suggesting unrelated breakage?

You MUST respond in EXACTLY this format (no other text):
STATUS: PASSED
REASON: <one line citing concrete evidence>
FIX_SUGGESTION: None

Or if failed:
STATUS: FAILED
REASON: <one line citing concrete evidence, e.g. "pytest reported 2 failures in test_foo.py">
FIX_SUGGESTION: <one line>
"""


class Verifier(BaseAgent):
    def __init__(self, metrics_tracker=None, memory=None):
        super().__init__(
            system_prompt=VERIFIER_PROMPT,
            tools=[],
            max_steps=1,
            metrics_tracker=metrics_tracker,
            agent_role="verifier",
        )
        self.memory = memory  # MemoryManager, used to read test_command/test_timeout

    def verify(
        self,
        user_prompt: str,
        step_description: str,
        coder_result: str,
        files_changed: list[str] | None = None,
    ) -> dict:
        files_changed = files_changed or []
        test_block, test_result = self._maybe_run_tests()

        verify_input = (
            f"Original User Request: {user_prompt}\n"
            f"Step Being Executed: {step_description}\n"
            f"Files Modified by Coder: {files_changed}\n"
            f"Coder's Actions and Results:\n{coder_result}\n\n"
            f"{test_block}\n"
        )

        self.reset_message()
        result = self.run(verify_input)
        verdict = self._parse_verdict(result["text"])
        verdict["test_run"] = test_result.to_dict() if test_result else None
        verdict["test_passed"] = test_result.passed() if test_result and test_result.executed else None
        return verdict

    def _maybe_run_tests(self) -> tuple[str, TestRunResult | None]:
        if not VERIFIER_RUN_TESTS:
            return "[TestRunner] disabled by config", None

        memory_hint_command = None
        memory_hint_timeout = None
        if self.memory is not None:
            ctx = self.memory.data.get("project_context", {})
            memory_hint_command = ctx.get("test_command") or None
            memory_hint_timeout = ctx.get("test_timeout") or None

        result = run_tests(
            workspace=config.WORKSPACE,
            memory_hint_command=memory_hint_command,
            memory_hint_timeout=memory_hint_timeout,
        )
        return result.to_prompt_block(), result

    def _parse_verdict(self, text: str) -> dict:
        lines = text.strip().splitlines()
        status = "FAILED"
        reason = ""
        fix_suggestion = ""
        for line in lines:
            line = line.strip()
            if line.upper().startswith("STATUS:"):
                status = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
            elif line.upper().startswith("FIX_SUGGESTION:"):
                fix_suggestion = line.split(":", 1)[1].strip()
        return {
            "passed": status == "PASSED",
            "reason": reason,
            "fix_suggestion": fix_suggestion,
        }
```

讲解：
- `verify()` 多接受 `files_changed` 参数（默认 `None` 兼容旧调用方）。
- 返回的 verdict 多 2 个字段：`test_run`（完整原始结果，给 metrics/debug 用）、`test_passed`（布尔，给 promote_facts 用作"是否真的过了测试"的硬证据）。
- prompt 显式让 LLM 看 `Files Modified by Coder`，做 scope 检查。

---

# P0.2 Memory B.2 — Working + Long-term 两层

## P0.2.1 设计要点

**Long-term Memory（持久化）**：
- 落盘 JSON
- 内容：`project_context` / `task_history` / `facts`
- 每条 fact 的字段：
  ```
  {
    "fact":             str,    # LLM 自己总结好的一句话
    "category":         str,    # 分类标签
    "confidence":       float,  # 0~1
    "reinforce_count":  int,    # 被 reinforce 的次数(独立追踪)
    "created_at_task":  int,    # 加入时的 task 序号(用于宽限期判断)
    ...其余 metadata: created_at, last_reinforced_at, reinforced_by
  }
  ```
- **confidence 计算规则**：
  - **新 fact 首次 promote**：`confidence = 0`，`reinforce_count = 0`
  - **同 fact 再次 promote**（在另一次任务里，coder 再次写下同一句话，且该任务通过）：`confidence += 0.2`（封顶 1.0），`reinforce_count += 1`
  - **关键**：`+0.2` 和 `count += 1` **只在 promote 时机触发**（即 `end_task(passed=True)` → 遍历 candidate → 命中现存 fact）。"被 LLM 调用 save_memory 写下" 这件事本身**不加分**，必须本次 task 真的通过才算
  - 起步 `0`（不是 0.5）的原因：没有减分机制时，起步只是一个偏移量，`0` 在语义上更清晰（"无证据" → "反复验证 5 次到顶"）
- **fact 排序分数**（用于淘汰）：
  ```
  score = confidence × reinforce_count
  ```
  - 高 confidence + 高 count → 分数高，靠前不会被淘汰
  - 低 confidence 又没被 reinforce → 分数低，优先淘汰
  - 满分 fact（封顶 1.0，count=5）`score = 5.0`
- **淘汰策略**（满 `MAX_MEMORY_FACTS` 时）：
  1. 候选淘汰池 = 年龄 ≥ `FACT_GRACE_PERIOD_TASKS`（默认 5 个 task）的 fact
  2. 在候选池里按 `score` 升序，扔分数最低的；同分按 `created_at_task` 升序（更老的先扔）
  3. 兜底：如果整个 long-term 都是新 fact（grace period 内），扔最老的新 fact 防止内存爆
- **Grace Period（宽限期）的作用**：保护新 fact 在加入后的 5 个 task 内不被淘汰，给它们"证明自己"的机会。一条 fact 加入后 5 个 task 内被 reinforce 过 → 长大成 score > 0，安全；没被 reinforce → 进入淘汰审判
- `add_fact` 去重做归一化匹配（lowercase + strip whitespace），不再字符串全等
- ⚠️ **没有减分机制**：错误 fact 一旦进 long-term，只能等被淘汰策略被动清除（详见 [Limitations L1](#l1-memory没有减分机制-no-demotion)）

**Working Memory（per-task，运行时只在内存）**：
- 不落盘，task 结束就丢
- 内容：当前 task 的 plan、step 之间的中间观察（read_file 摘要）、candidate facts（**未 promote**）
- 提供 `snapshot_for_coder()` 返回一段拼好的文本，注入 Coder 的 user_input
- candidate facts 由 **LLM 主动调用 `save_memory` 工具**写入。`save_memory` 不做任何提炼，**总结这件事是 LLM 自己的责任**：LLM 应该把"我看到了什么"压缩成一句可泛化的经验后再调用工具。详见 P0.2.4 (a)

**Promotion 规则**：
- task **passed**：working memory 里的所有 candidate facts 全部 promote 到 long-term
  - 新 fact → 以 0.5 起步加入
  - 已存在 fact → confidence `+= 0.2`（封顶 1.0）
- task **failed**：所有 candidate facts **一刀切丢弃**，不区分这条 fact 跟失败是否相关，**不污染长期记忆**
  - ⚠️ 已知代价：好 candidate 也会被错杀，下次任务要重新学一遍。这是有意的保守策略，详见 [Limitations L2](#l2-memorytask-失败时-candidate-facts-全部丢弃-一刀切)

**注入到下游 prompt 的方式**：
- planner / coder 拉取 facts 时，按 confidence 倒序取 top-N（`max_facts=10`）
- ⚠️ 当前**没有相关性检索**，纯按 confidence 排序，可能注入跟当前任务无关但 confidence 高的 fact。详见 [Limitations L3](#l3-memory没有相关性检索-no-relevance-retrieval)

## P0.2.2 `config.py` 改动

在文件末尾追加：

```python
# === Memory B.2 config (added in v2) ===
MAX_WORKING_OBSERVATIONS = 20      # max in-RAM observations per task
WORKING_OBSERVATION_MAX_CHARS = 500  # truncate each observation
FACT_INITIAL_CONFIDENCE = 0.0       # 起步无证据，只有被 reinforce 才涨
FACT_REINFORCE_DELTA = 0.2
FACT_MAX_CONFIDENCE = 1.0
FACT_GRACE_PERIOD_TASKS = 5         # 新 fact 至少存活 5 个 task 才能被淘汰
```

讲解 `FACT_GRACE_PERIOD_TASKS`：
- 配合 `_evict_if_needed` 使用：内存满需要淘汰时，**年龄不足 5 个 task 的 fact 完全免疫**
- 新 fact 在 grace period 内有 5 次机会被 reinforce（每个 task 一次），过期还是 0 分就接受淘汰
- 经验值，可调：被错杀就调大（7/10），新 fact 淤积就调小（3）

## P0.2.3 `memory.py` 完整重写

```python
"""
Two-layer memory: Working (per-task, in-RAM) + Long-term (persistent JSON).

- WorkingMemory: created per task, discarded after end_task; holds candidate
  facts and step-to-step observations the Coder produces.
- MemoryManager: persistent JSON store; only data the verifier confirms ends
  up here.
"""
import os
import json
from datetime import datetime
from typing import Optional

import config
from config import (
    WORKSHOP,
    MAX_MEMORY_TASKS, MAX_MEMORY_FACTS,
    MAX_WORKING_OBSERVATIONS, WORKING_OBSERVATION_MAX_CHARS,
    FACT_INITIAL_CONFIDENCE, FACT_REINFORCE_DELTA, FACT_MAX_CONFIDENCE,
    FACT_GRACE_PERIOD_TASKS,
)


# ---------------------------------------------------------------------------
# Working Memory (per-task)
# ---------------------------------------------------------------------------

class WorkingMemory:
    def __init__(self, task_id: str, user_prompt: str):
        self.task_id = task_id
        self.user_prompt = user_prompt
        self.plan: list[str] = []
        self.observations: list[dict] = []      # [{kind, content, ts}]
        self.candidate_facts: list[dict] = []   # [{fact, category}]
        self.files_changed: set[str] = set()

    # --- writers (called by main / coder / tool) ---

    def set_plan(self, plan: list[str]) -> None:
        self.plan = list(plan)

    def add_observation(self, kind: str, content: str) -> None:
        if len(self.observations) >= MAX_WORKING_OBSERVATIONS:
            self.observations.pop(0)
        self.observations.append({
            "kind": kind,
            "content": content[:WORKING_OBSERVATION_MAX_CHARS],
            "ts": datetime.now().isoformat(timespec="seconds"),
        })

    def add_candidate_fact(self, fact: str, category: str) -> None:
        norm = _normalize_fact(fact)
        if any(_normalize_fact(f["fact"]) == norm for f in self.candidate_facts):
            return
        self.candidate_facts.append({"fact": fact, "category": category})

    def add_file_changed(self, path: str) -> None:
        self.files_changed.add(path)

    # --- readers (used by Coder prompt) ---

    def snapshot_for_coder(self) -> str:
        if not self.observations and not self.plan:
            return ""
        parts = [f"[WorkingMemory] task_id={self.task_id}"]
        if self.plan:
            parts.append("Plan:")
            for i, step in enumerate(self.plan, 1):
                parts.append(f"  {i}. {step}")
        if self.observations:
            parts.append("Recent observations from earlier steps:")
            for obs in self.observations[-10:]:
                parts.append(f"  - [{obs['kind']}] {obs['content']}")
        if self.files_changed:
            parts.append(f"Files modified so far: {sorted(self.files_changed)}")
        return "\n".join(parts)


def _normalize_fact(fact: str) -> str:
    return " ".join(fact.strip().lower().split())


# ---------------------------------------------------------------------------
# Long-term Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    def __init__(self, memory_file: Optional[str] = None):
        if memory_file is None:
            memory_dir = os.path.join(WORKSHOP, config.PROJECT_NAME)
            os.makedirs(memory_dir, exist_ok=True)
            memory_file = os.path.join(memory_dir, "memory.json")
        self.memory_file = memory_file
        self.data = self._load()
        self._working: Optional[WorkingMemory] = None

    # --- load / save ---

    def _load(self) -> dict:
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_data()

    def _save(self) -> None:
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _default_data(self) -> dict:
        return {
            "project_context": {
                "project_name": config.PROJECT_NAME,
                "workspace": config.WORKSPACE,
                "language": "",
                "framework": "",
                "entry_file": "",
                "test_command": "",
                "test_timeout": None,
                "updated_at": "",
            },
            "task_history": [],
            "facts": [],
        }

    # --- project context ---

    def update_project_context(self, **kwargs) -> None:
        self.data["project_context"].update(kwargs)
        self.data["project_context"]["updated_at"] = self._now()
        self._save()

    # --- working memory lifecycle ---

    def begin_task(self, task_id: str, user_prompt: str) -> WorkingMemory:
        self._working = WorkingMemory(task_id=task_id, user_prompt=user_prompt)
        return self._working

    def get_working(self) -> Optional[WorkingMemory]:
        return self._working

    def end_task(
        self,
        task_id: str,
        passed: bool,
        plan: list[str],
        attempts: int,
        summary: str,
    ) -> None:
        wm = self._working
        files_changed = sorted(wm.files_changed) if wm else []

        # Current task index (1-based): the new task we're about to record.
        # task_history hasn't been appended yet, so this is len + 1.
        current_task_idx = len(self.data["task_history"]) + 1

        # Promote candidate facts only when task passed
        if passed and wm:
            for cf in wm.candidate_facts:
                self._promote_fact(cf["fact"], cf["category"], task_id, current_task_idx)

        # Always record task history
        record = {
            "task_id": task_id,
            "timestamp": self._now(),
            "user_prompt": wm.user_prompt if wm else "",
            "plan": plan,
            "status": "passed" if passed else "failed",
            "files_changed": files_changed,
            "attempts": attempts,
            "summary": summary,
        }
        self.data["task_history"].append(record)
        self._trim_task_history()

        # Eviction runs AFTER history append so current_task_idx is consistent
        # with what facts will see as "now"
        self._evict_facts_if_needed(current_task_idx)

        self._save()

        # Drop working memory
        self._working = None

    # --- facts management ---

    def _promote_fact(self, fact: str, category: str,
                      source_task_id: str, current_task_idx: int) -> None:
        norm = _normalize_fact(fact)
        for existing in self.data["facts"]:
            if _normalize_fact(existing["fact"]) == norm:
                # Reinforce existing fact
                existing["confidence"] = min(
                    FACT_MAX_CONFIDENCE,
                    existing.get("confidence", FACT_INITIAL_CONFIDENCE) + FACT_REINFORCE_DELTA,
                )
                existing["reinforce_count"] = existing.get("reinforce_count", 0) + 1
                existing["last_reinforced_at"] = self._now()
                existing.setdefault("reinforced_by", []).append(source_task_id)
                return

        # Brand-new fact: starts at confidence=0, count=0, age=0
        self.data["facts"].append({
            "fact": fact,
            "category": category,
            "confidence": FACT_INITIAL_CONFIDENCE,        # 0.0
            "reinforce_count": 0,
            "created_at_task": current_task_idx,          # for grace period
            "source_task_id": source_task_id,
            "created_at": self._now(),
            "last_reinforced_at": self._now(),
            "reinforced_by": [source_task_id],
        })

    def _evict_facts_if_needed(self, current_task_idx: int) -> None:
        """
        Evict facts when long-term exceeds MAX_MEMORY_FACTS.

        Policy (two-tier sort):
          1. Primary key: in_grace_period? (True sorts LAST = protected)
             Facts whose age < FACT_GRACE_PERIOD_TASKS are deprioritized
             from eviction unless we have no other choice.
          2. Secondary key: score = confidence × reinforce_count (lowest first).
             Within the same protection tier, the lowest score gets evicted.
          3. Tertiary key: created_at_task (oldest first).
             Same score → older fact had more chances, evict it first.

        This is NOT a hard exemption — grace period only LOWERS eviction
        priority. If memory pressure forces it (e.g. all facts are new),
        the lowest-score new fact still gets evicted instead of crashing.
        """
        def age(f):
            return current_task_idx - f.get("created_at_task", current_task_idx)

        def score(f):
            return f.get("confidence", 0) * f.get("reinforce_count", 0)

        def sort_key(f):
            in_grace = age(f) < FACT_GRACE_PERIOD_TASKS
            return (
                in_grace,                          # False (no protection) sorts first
                score(f),                          # then lowest score
                f.get("created_at_task", 0),       # then oldest
            )

        while len(self.data["facts"]) > MAX_MEMORY_FACTS:
            victim = min(self.data["facts"], key=sort_key)
            self.data["facts"].remove(victim)

    def _trim_task_history(self) -> None:
        if len(self.data["task_history"]) > MAX_MEMORY_TASKS:
            self.data["task_history"] = self.data["task_history"][-MAX_MEMORY_TASKS:]

    # --- context injection for Planner ---

    def get_context_for_planner(self, max_facts: int = 10) -> str:
        parts = []

        ctx = self.data.get("project_context", {})
        non_empty = {k: v for k, v in ctx.items()
                     if v and k not in ("updated_at",)}
        if non_empty:
            parts.append("Project info: " + ", ".join(f"{k}={v}" for k, v in non_empty.items()))

        facts = sorted(
            self.data.get("facts", []),
            key=lambda f: f.get("confidence", 0),
            reverse=True,
        )[:max_facts]
        if facts:
            parts.append("Known facts about this project:")
            for f in facts:
                parts.append(f"  - [{f['category']} conf={f.get('confidence', 0):.1f}] {f['fact']}")

        history = self.data.get("task_history", [])[-5:]
        if history:
            parts.append("Recent tasks:")
            for t in history:
                parts.append(f"  - [{t['status']}] {t['user_prompt']}")

        return "\n".join(parts) if parts else ""

    # --- helpers ---

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def generate_task_id(self) -> str:
        count = len(self.data.get("task_history", []))
        return f"{count + 1:04d}"
```

讲解关键决策：

1. **WorkingMemory 完全不落盘**。task 失败重试也不会污染。如果你想 debug 某次失败任务，加个 `--dump-working-memory` flag 把它打到日志即可，但 v1 不做。
2. **`_promote_fact` 是 promote 的唯一入口**。`tool.py:save_memory` 不再直接写 long-term，而是写到 working memory 的 candidate_facts。只有 verifier 通过后由 `end_task` 触发 promote。
3. **fact 去重用 `_normalize_fact`**（lowercase + collapse whitespace），比之前的字符串全等强很多。
4. **`_promote_fact` 不再调淘汰**，淘汰统一在 `end_task` 末尾调 `_evict_facts_if_needed`。这样 `current_task_idx` 在两个地方一致，年龄计算不会出错。
5. **`facts` 按 confidence 排序注入 prompt**，高 confidence 的优先，避免低质 fact 挤掉硬证据。注意：注入排序用 `confidence`，淘汰排序用 `score = confidence × reinforce_count`，两者用途不同。当前模型下两者单调对应（因为 `confidence = 0.2 × reinforce_count`），将来如果引入 LLM 自评 confidence 会解耦。
6. **`get_context_for_planner` 加 `max_facts=10` 限制**，不再无脑全量注入。
7. **新 fact 的 grace period 保护**：`_evict_facts_if_needed` 里 `age >= FACT_GRACE_PERIOD_TASKS` 的过滤是核心。新 fact 在加入后的 5 个 task 内绝对不会被淘汰，给它们机会被 reinforce。这是 cold-start 保护，避免"刚加进来还没机会证明自己就被踢出去"。

## P0.2.4 `tool.py` 改动

> **前置已完成**：在 Step 0（P0.0 章节）你已经把 `from config import WORKSPACE` 改成 `import config`，所有 `WORKSPACE` 改成 `config.WORKSPACE`。下面这些散点改动是在 Step 0 的基础上叠加的。

需要改 4 处：

### (a) `save_memory` 改为写 working memory 的 candidate

**定位**：`save_memory` 是一个**纯写入工具**，不做任何提炼/总结/分类智能。"把观察压缩成一句可泛化的经验"这件事**完全由 LLM 自己负责**——工具只接收 LLM 已经准备好的字符串，写到 working memory 的 candidate 区。

这跟旧版的关键区别：
| 维度 | 旧版（直接写 long-term） | 新版（写 working candidate） |
|---|---|---|
| 写入目标 | `MemoryManager.facts`（持久化） | `WorkingMemory.candidate_facts`（per-task 内存） |
| 即时可见性 | 立刻进 long-term，下次任务直接被读到 | 必须本次 task 通过 verifier 才会 promote |
| 失败保护 | 无，烂 fact 立刻污染长期记忆 | task 失败 → 全部丢弃 |

```python
def save_memory(fact: str, category: str) -> str:
    """
    Write-only tool. Does NOT summarize. The LLM must compress its observation
    into a single generalizable sentence BEFORE calling this. The fact lands in
    working memory's candidate area; it will be promoted to long-term only if
    the current task passes verification.
    """
    if _memory_manager is None:
        return "Memory manager not initialized."
    wm = _memory_manager.get_working()
    if wm is None:
        return "No active task; fact dropped."
    wm.add_candidate_fact(fact, category)
    return f"Recorded candidate fact [{category}]: {fact} (will be saved if task passes verification)"
```

**配套的 tool description**（注册到 LLM 工具列表的描述文案）建议改成：

```python
{
    "name": "save_memory",
    "description": (
        "Record a single, generalizable lesson you've learned during this task. "
        "IMPORTANT: do NOT dump raw file contents or task-specific details. "
        "Compress your observation into ONE sentence that will help future tasks "
        "in similar projects (e.g. 'this project uses pytest with -q' or "
        "'auth tokens are stored in env var AUTH_TOKEN'). "
        "The fact only becomes permanent if the current task passes verification."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "ONE sentence, already summarized and generalized by you."
            },
            "category": {
                "type": "string",
                "description": "Short tag like 'testing', 'config', 'architecture', 'convention'."
            }
        },
        "required": ["fact", "category"]
    }
}
```

> ⚠️ **注意**：fact 质量完全依赖 LLM 自觉，没有程序级的强制检查（长度/重复/泛化度都不校验）。详见 [Limitations L4](#l4-memoryfact-质量完全依赖-llm-自觉-soft-constraint)。如果发现 LLM 总是存任务细节级 fact，需要回头优化 system prompt 或加 promote 前过滤。

### (b) `write_file` / `replace_in_file` 成功后记录 files_changed

在每个写操作的 success 分支末尾追加：

```python
def _record_file_change(file_path: str) -> None:
    if _memory_manager is None:
        return
    wm = _memory_manager.get_working()
    if wm is None:
        return
    try:
        rel = os.path.relpath(safe_path(file_path), config.WORKSPACE)
    except Exception:
        rel = file_path
    wm.add_file_changed(rel)
```

`write_file` 在 `f.write(content)` 之后调用：
```python
_record_file_change(file_path)
return f"Content written successfully to {path}"
```

`replace_in_file` 在 `f.write(updated)` 之后同样调用：
```python
_record_file_change(file_path)
return f"'{old_text}' replaced with '{new_text}' in {path} successfully"
```

### (c) `read_file` / `list_dir` 成功后记录到 working observations（可选但建议）

让 working memory 在多步之间能复用之前的中间结果，避免 coder 重复读同一个文件：

```python
def _record_observation(kind: str, content: str) -> None:
    if _memory_manager is None:
        return
    wm = _memory_manager.get_working()
    if wm is None:
        return
    wm.add_observation(kind=kind, content=content)
```

`read_file` 末尾：
```python
result = content
_record_observation("read_file", f"{file_path}: {result[:200]}")
return result
```

`list_dir` 末尾：
```python
output = "\n".join(result) if result else "Directory is empty"
_record_observation("list_dir", f"{dir_path}: {output[:200]}")
return output
```

> 注意：不要给 `run_command` 加 observation hook，命令输出可能巨大且很多噪音。

## P0.2.5 `coder.py` 改动

让 Coder 在 `run()` 之前把 working memory snapshot 拼到第一条 user 消息：

```python
class Coder(BaseAgent):
    def __init__(self, metrics_tracker=None, memory=None):
        super().__init__(
            system_prompt=CODER_PROMPT,
            tools=Tools.get_tools(),
            metrics_tracker=metrics_tracker,
            agent_role="coder",
        )
        self.memory = memory

    def run(self, input_text: str) -> dict:
        if self.memory is not None:
            wm = self.memory.get_working()
            if wm is not None:
                snapshot = wm.snapshot_for_coder()
                if snapshot:
                    input_text = f"{snapshot}\n\n---\nCurrent step:\n{input_text}"
        return super().run(input_text)
```

## P0.2.6 `main.py` 改动

只改 `run_task` 和 `main` 两个函数：

```python
def run_task(user_prompt: str, planner: Planner, coder: Coder,
             verifier: Verifier, memory: MemoryManager,
             metrics: MetricsTracker = None) -> str:
    task_id = memory.generate_task_id()

    # Begin task -> create working memory
    working = memory.begin_task(task_id=task_id, user_prompt=user_prompt)

    memory_context = memory.get_context_for_planner()
    failure_context = None
    total_attempts = 0
    final_plan: list[str] = []
    overall_passed = False

    print(f"\n{'='*50}")
    print(f"Task[{task_id}]: {user_prompt}")
    print(f"{'='*50}")

    try:
        for replan in range(MAX_REPLANS + 1):
            if replan > 0:
                print(f"\n-- replan attempt {replan}/{MAX_REPLANS} --")

            print("\n[Phase: Planning]")
            plan_steps = planner.create_plan(
                user_task=user_prompt,
                memory_context=memory_context,
                failure_context=failure_context,
            )
            final_plan = plan_steps
            working.set_plan(plan_steps)

            for index, step in enumerate(plan_steps):
                print(f"  Step{index+1}: {step}")

            print("\n[Phase: Execution]")
            all_passed = True
            error_history = []

            for step_idx, step_desc in enumerate(plan_steps):
                print(f"\n --- Step {step_idx+1}/{len(plan_steps)}: {step_desc} ---")
                step_passed = False
                current_step = step_desc

                for attempt in range(MAX_RETRIES_PER_STEP):
                    total_attempts += 1
                    if attempt > 0:
                        print(f"Retry {attempt}/{MAX_RETRIES_PER_STEP-1}")

                    coder.reset_message()
                    coder_result = coder.run(current_step)
                    print(f"[Coder] {'completed' if coder_result['completed'] else 'max steps reached'}")

                    verify_result = verifier.verify(
                        user_prompt=user_prompt,
                        step_description=step_desc,
                        coder_result=coder_result["text"],
                        files_changed=sorted(working.files_changed),
                    )
                    print(f"[Verifier] {verify_result['reason']}")

                    if verify_result["passed"]:
                        print("PASSED")
                        step_passed = True
                        break
                    else:
                        print(f"FAILED: {verify_result['fix_suggestion']}")
                        error_history.append({
                            "step": step_desc,
                            "attempt": attempt + 1,
                            "reason": verify_result["reason"],
                            "fix_suggestion": verify_result["fix_suggestion"],
                        })
                        current_step = (
                            f"{step_desc}\n\n"
                            f"Previous attempt failed:\n"
                            f"Reason: {verify_result['reason']}\n"
                            f"Fix suggestion: {verify_result['fix_suggestion']}"
                        )

                if not step_passed:
                    all_passed = False
                    failure_context = _build_failure_context(plan_steps, step_idx, error_history)
                    break

            if all_passed:
                overall_passed = True
                summary = f"Completed {len(plan_steps)} steps successfully."
                _print_metrics(metrics)
                return f"Task completed successfully.\n{summary}"

        summary = f"Task failed after {MAX_REPLANS} replan attempts."
        _print_metrics(metrics)
        return summary

    finally:
        # End task: promote (if passed) and record history regardless
        memory.end_task(
            task_id=task_id,
            passed=overall_passed,
            plan=final_plan,
            attempts=total_attempts,
            summary=("Completed successfully." if overall_passed else "Failed."),
        )
```

`main()` 里构造 Coder/Verifier 时注入 memory：

```python
planner  = Planner(metrics_tracker=metrics)
coder    = Coder(metrics_tracker=metrics, memory=memory)
verifier = Verifier(metrics_tracker=metrics, memory=memory)
```

讲解关键点：
- `try / finally` 保证就算中途异常也会调 `end_task`，避免 working memory 泄漏到下次任务。
- promote 的判定就是 `overall_passed`，跟"是否走完所有 step 且 verifier 都说过"绑定。
- `verifier.verify(..., files_changed=sorted(working.files_changed))` 这一行就是 P0.3 的接口。

---

# P0.3 files_changed 收集

实际上已经在 P0.2.4 (b) 实现完了，这里集中说明设计。

**收集点**：`tool.py` 的 `write_file` / `replace_in_file` **执行成功之后**调 `_record_file_change(file_path)`。
**存储位置**：`WorkingMemory.files_changed: set[str]`，自动去重。
**生命周期**：随 working memory 一起在 task 开始时清空、结束时落到 `task_history` record。
**消费者**：
1. **Verifier** 实时拿到（每次 verify 调用）做 scope 检查
2. **task_history** 持久化记录，将来可以 grep "上次谁动了 auth.py"
3. **debug 输出**：task 结束时打印 "本次改动文件: X, Y, Z"，用户友好

**注意事项**：
- 路径统一转成相对 `WORKSPACE` 的相对路径（避免存绝对路径泄漏机器信息）
- `safe_path` 失败时 fallback 到原始字符串，不让 tool 因为 tracking 失败而崩

---

# P1 跑 MBPP 数据集

## P1.1 设计要点

- **数据集**：使用 `google-research-datasets/mbpp` 的 **`sanitized`** subset，**不用 `full`**
  - `sanitized` = 从 974 条原始众包数据里**人工校验出 427 条**，修复了任务描述歧义、test case bug、ground truth 不一致等问题
  - 是近年 LLM coding benchmark 论文（CodeT5 / StarCoder / SWE-bench 等）的**标配**，跑出来的分数有学术可比性
  - **split 选 `test`**（257 条），这是 README 明确指定的 benchmark 评测集；其他 split（train / validation / prompt）不在评测范围内
  - ⚠️ **关键字段名差异**：`sanitized` subset 的任务描述字段叫 **`prompt`**，不是 `full` 的 `text`。`task_id` / `code` / `test_list` 字段名两个 subset 一致
- **目录命名标准化**：每条 MBPP 样本一个独立目录，固定文件名 `solution.py` / `test_solution.py`
- **测试命令固定写死**：`pytest -q test_solution.py`，setup 时直接写进 memory
- **不需要 marker**（第一层 memory 命中就够）
- **不需要 docker**（subprocess 后端）

> **目录命名约定**：本节起所有数据集相关脚本（setup / run）放在 `runners/` 目录下，**不要叫 `datasets/`**。原因：HuggingFace 的 `datasets` 库 `pip install datasets` 装的就是 `datasets` 这个包名，跟本地目录冲突——你执行 `from datasets import load_dataset` 时 Python 会优先找到我们自己的目录，找不到 `load_dataset` 报错。`runners/` 目录下放一个空 `__init__.py` 即可。

## P1.2 目录结构

```
Execution/
└── mbpp_<task_id>/
    ├── solution.py        # initial: empty or stub
    ├── test_solution.py   # generated from MBPP sanitized `test_list`
    ├── memory.json        # per-instance memory (pre-fills test_command)
    └── prompt.md          # MBPP sanitized `prompt` field (task description)
```

每条样本一个独立 workspace，互不污染。`task_id` 来自 sanitized subset 的原始 task_id（11~510 的范围对应 test split）。

## P1.3 新文件 `runners/setup_mbpp.py`

```python
"""
Load the MBPP dataset and materialize each instance into a standardized
workspace under WORKSHOP, with a per-instance memory.json pre-populated
with the test_command.

Dataset: google-research-datasets/mbpp, subset='sanitized' (427 hand-verified
problems). We use the 'test' split (257 problems, task_id 11~510) for
benchmark evaluation. Field naming note: sanitized uses 'prompt' for the
task description (full uses 'text'); other fields (task_id, code, test_list)
are identical across both subsets.
"""
import os
import json
import argparse
from datasets import load_dataset

from config import WORKSHOP


SOLUTION_STUB = '''"""MBPP task — implement the function described in prompt.md."""
'''


def materialize_instance(task_id: int, text: str, code: str, test_list: list[str]) -> str:
    workspace = os.path.join(WORKSHOP, f"mbpp_{task_id:04d}")
    os.makedirs(workspace, exist_ok=True)

    # solution.py: empty stub (the agent will fill it in)
    with open(os.path.join(workspace, "solution.py"), "w", encoding="utf-8") as f:
        f.write(SOLUTION_STUB)

    # test_solution.py: import from solution and run MBPP asserts
    test_body = "from solution import *\n\n"
    for i, t in enumerate(test_list):
        test_body += f"def test_case_{i}():\n    {t}\n\n"
    with open(os.path.join(workspace, "test_solution.py"), "w", encoding="utf-8") as f:
        f.write(test_body)

    # prompt.md: human task description, used as user_prompt
    with open(os.path.join(workspace, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(f"# MBPP Task {task_id}\n\n{text}\n")

    # memory.json: pre-fill test_command so verifier hits the first fallback
    memory = {
        "project_context": {
            "project_name": f"mbpp_{task_id:04d}",
            "workspace": workspace,
            "language": "python",
            "framework": "pytest",
            "entry_file": "solution.py",
            "test_command": "pytest -q test_solution.py",
            "test_timeout": 30,
            "updated_at": "",
        },
        "task_history": [],
        "facts": [],
    }
    with open(os.path.join(workspace, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    return workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "prompt"])
    parser.add_argument("--limit", type=int, default=10, help="0 = all")
    args = parser.parse_args()

    print(f"Loading MBPP split={args.split} ...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=args.split)

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"Materializing {n} instances ...")

    for i in range(n):
        row = ds[i]
        ws = materialize_instance(
            task_id=row["task_id"],
            text=row["prompt"],          # sanitized split uses "prompt"
            code=row["code"],
            test_list=row["test_list"],
        )
        print(f"  [{i+1}/{n}] {ws}")

    print("Done.")


if __name__ == "__main__":
    main()
```

## P1.4 新文件 `runners/run_mbpp.py`

```python
"""
Run the agent over previously materialized MBPP instances.
Each instance has its own workspace + its own memory.json.
"""
import os
import json
import argparse
import glob

from config import WORKSHOP, ENABLE_METRICS
import config as config_module

from planner import Planner
from coder import Coder
from verifier import Verifier
from memory import MemoryManager
from metrics import MetricsTracker
import tool

from main import run_task


def run_one(workspace: str) -> dict:
    # Override globals so all modules see the right workspace
    instance_name = os.path.basename(workspace)
    config_module.PROJECT_NAME = instance_name
    config_module.WORKSPACE = workspace

    # Re-init memory bound to this instance's memory.json
    memory = MemoryManager(memory_file=os.path.join(workspace, "memory.json"))
    metrics = MetricsTracker() if ENABLE_METRICS else None

    planner  = Planner(metrics_tracker=metrics)
    coder    = Coder(metrics_tracker=metrics, memory=memory)
    verifier = Verifier(metrics_tracker=metrics, memory=memory)
    tool.set_memory_manager(memory)

    with open(os.path.join(workspace, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read()

    result_text = run_task(prompt, planner, coder, verifier, memory, metrics)
    last_record = memory.data["task_history"][-1] if memory.data["task_history"] else {}
    return {
        "instance": instance_name,
        "status": last_record.get("status", "unknown"),
        "attempts": last_record.get("attempts", 0),
        "files_changed": last_record.get("files_changed", []),
        "result_text": result_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="mbpp_*", help="glob under WORKSHOP")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default="mbpp_report.json")
    args = parser.parse_args()

    workspaces = sorted(glob.glob(os.path.join(WORKSHOP, args.pattern)))
    if args.limit:
        workspaces = workspaces[: args.limit]

    print(f"Running {len(workspaces)} MBPP instances ...")
    results = []
    for ws in workspaces:
        print(f"\n========== {os.path.basename(ws)} ==========")
        try:
            results.append(run_one(ws))
        except Exception as e:
            results.append({"instance": os.path.basename(ws), "status": "crashed", "error": str(e)})

    passed = sum(1 for r in results if r.get("status") == "passed")
    print(f"\n=== MBPP report: {passed}/{len(results)} passed ===")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
```

## P1.5 怎么跑

```bash
# 1) materialize first 10 MBPP test instances
python runners/setup_mbpp.py --split test --limit 10

# 2) run agent on them
python runners/run_mbpp.py --limit 10 --report mbpp_report.json
```

**注意一个 caveat**：`run_mbpp.py` 通过修改 `config.WORKSPACE` 全局变量来切换 workspace。这是 quick hack，**P1 够用**，但如果你之后想并行跑多 instance，需要把 WORKSPACE 改成 per-call 参数传递。这是 P3 的事，先标记为 TODO。

---

# P2 跑 SWE-bench 数据集

## P2.1 设计要点

- **数据集**：用 `SWE-bench/SWE-bench_Verified`（500 条人工核验过的）
- **不用官方 harness**，避免 120GB 磁盘要求
- **按需拉镜像 + 跑完即删**（一次只占 1 个 ~2-3 GB 镜像）
- 每个 instance：`git clone <repo>` → `git checkout <base_commit>` → `docker pull <image>` → 写 memory → 跑 agent → 跑测试 → `docker rmi`
- **关键设计**：verifier 不需要任何修改。SWE-bench instance 的 `test_command` 字段直接是一段 `docker run ...` 字符串，subprocess 跑这个字符串就会在 docker 容器里跑测试。verifier 完全不感知 docker 的存在。
- **per-repo timeout 表**写死在 setup 脚本里（数据集没给）
- **镜像名记在 sidecar 文件**：每个 instance 目录下放一个 `.docker_image` 文件（一行文本就是镜像名），run 阶段读出来在结束时 rmi。不写到 `project_context` 里，因为 `project_context` 是给 agent 看的，agent 不需要知道 docker。

## P2.2 新文件 `runners/setup_swebench.py`

```python
"""
Load SWE-bench Verified. For each instance:
- clone the repo at base_commit
- pull the docker image (so it's ready when run_swebench runs)
- write per-instance memory.json with a docker-wrapped test_command
- write a .docker_image sidecar file so run_swebench knows what to rmi
"""
import os
import json
import argparse
import subprocess
from datasets import load_dataset

from config import WORKSHOP


# Per-repo timeout table (seconds). Defaults to 300 if repo not listed.
# SWE-bench dataset has no timeout field; this is our manual table.
REPO_TIMEOUT = {
    "django/django": 300,
    "sympy/sympy": 600,
    "matplotlib/matplotlib": 300,
    "scikit-learn/scikit-learn": 600,
    "astropy/astropy": 900,
    "sphinx-doc/sphinx": 300,
    "pytest-dev/pytest": 300,
    "pylint-dev/pylint": 300,
    "pydata/xarray": 300,
    "psf/requests": 120,
    "pallets/flask": 120,
    "mwaskom/seaborn": 300,
}


def build_inner_pytest_command(fail_to_pass: list[str], pass_to_pass: list[str]) -> str:
    """Build the pytest command that runs INSIDE the docker container."""
    nodes = list(fail_to_pass) + list(pass_to_pass)
    if not nodes:
        return "pytest -q"
    quoted = " ".join(f'\\"{n}\\"' for n in nodes)  # escape for outer bash -c
    return f"pytest -q {quoted}"


def build_docker_test_command(workspace: str, image: str, inner_cmd: str) -> str:
    """Wrap the inner pytest command in a `docker run ...` invocation."""
    abs_ws = os.path.abspath(workspace)
    return (
        f'docker run --rm '
        f'-v "{abs_ws}":/testbed '
        f'-w /testbed '
        f'{image} '
        f'bash -c "{inner_cmd}"'
    )


def docker_image_for_instance(instance_id: str) -> str:
    """SWE-bench official image naming convention."""
    # NOTE: this format is approximate. The official mapping replaces some
    # characters; verify with `docker pull` before bulk use. See P2.5.
    safe = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{safe}:latest"


def materialize_instance(row: dict, repos_cache_dir: str, do_pull: bool = True) -> str:
    instance_id = row["instance_id"]
    repo = row["repo"]                       # e.g. "django/django"
    base_commit = row["base_commit"]
    workspace = os.path.join(WORKSHOP, instance_id)

    if os.path.exists(workspace):
        print(f"  skip (exists): {workspace}")
        return workspace

    # Clone (or reuse) bare cache, then clone from cache to workspace.
    cache_path = os.path.join(repos_cache_dir, repo.replace("/", "__"))
    if not os.path.exists(cache_path):
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", cache_path],
            check=True,
        )
    subprocess.run(["git", "-C", cache_path, "fetch", "--all"], check=True)
    os.makedirs(os.path.dirname(workspace), exist_ok=True)
    subprocess.run(["git", "clone", cache_path, workspace], check=True)
    subprocess.run(["git", "-C", workspace, "checkout", base_commit], check=True)

    # prompt.md (the user_prompt the agent will see)
    with open(os.path.join(workspace, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(f"# {instance_id}\n\n{row['problem_statement']}\n")

    # Resolve test nodes
    fail_to_pass = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
    pass_to_pass = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]

    image = docker_image_for_instance(instance_id)
    inner_cmd = build_inner_pytest_command(fail_to_pass, pass_to_pass)
    test_command = build_docker_test_command(workspace, image, inner_cmd)
    test_timeout = REPO_TIMEOUT.get(repo, 300)

    # Pull the image now so run_swebench doesn't have to wait for the network later
    if do_pull:
        print(f"  pulling {image} ...")
        result = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARN: docker pull failed: {result.stderr.strip()}")
            print(f"  (continuing; you may need to fix the image name in P2.5)")

    # memory.json — note: docker_image / repo / base_commit / fail_to_pass etc.
    # are NOT written here. project_context is what the agent sees, and the
    # agent doesn't need to know about docker.
    memory = {
        "project_context": {
            "project_name": instance_id,
            "workspace": workspace,
            "language": "python",
            "framework": "pytest",
            "test_command": test_command,
            "test_timeout": test_timeout,
            "updated_at": "",
        },
        "task_history": [],
        "facts": [],
    }
    with open(os.path.join(workspace, ".agent_memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    # Sidecar: image name for run_swebench to rmi after the run
    with open(os.path.join(workspace, ".docker_image"), "w", encoding="utf-8") as f:
        f.write(image + "\n")

    return workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--repos-cache", default=os.path.join(WORKSHOP, "_repos_cache"))
    parser.add_argument("--no-pull", action="store_true", help="skip docker pull during setup")
    args = parser.parse_args()

    os.makedirs(args.repos_cache, exist_ok=True)
    print(f"Loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)

    n = min(args.limit, len(ds)) if args.limit else len(ds)
    print(f"Materializing {n} instances ...")
    for i in range(n):
        try:
            ws = materialize_instance(ds[i], args.repos_cache, do_pull=not args.no_pull)
            print(f"  [{i+1}/{n}] {ws}")
        except subprocess.CalledProcessError as e:
            print(f"  [{i+1}/{n}] FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
```

## P2.3 新文件 `runners/run_swebench.py`

```python
"""
Run the agent over previously materialized SWE-bench instances.

For each instance: switch config.WORKSPACE to the instance dir, load
per-instance memory, run the agent, capture git diff, then docker rmi the
sidecar-recorded image to keep disk small (rolling mode).

Note: no SANDBOX_BACKEND switching is needed. The verifier always uses
SubprocessRunner; the docker invocation is baked into the test_command
string written by setup_swebench.py.
"""
import os
import json
import argparse
import glob
import subprocess

import config as config_module
from config import WORKSHOP, ENABLE_METRICS

from planner import Planner
from coder import Coder
from verifier import Verifier
from memory import MemoryManager
from metrics import MetricsTracker
import tool

from main import run_task


def _read_sidecar_image(workspace: str) -> str | None:
    p = os.path.join(workspace, ".docker_image")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip() or None


def run_one(workspace: str) -> dict:
    instance_name = os.path.basename(workspace)
    memory_file = os.path.join(workspace, ".agent_memory.json")

    # Switch globals for this instance (only WORKSPACE / PROJECT_NAME, no docker config)
    config_module.PROJECT_NAME = instance_name
    config_module.WORKSPACE = workspace

    memory = MemoryManager(memory_file=memory_file)

    metrics  = MetricsTracker() if ENABLE_METRICS else None
    planner  = Planner(metrics_tracker=metrics)
    coder    = Coder(metrics_tracker=metrics, memory=memory)
    verifier = Verifier(metrics_tracker=metrics, memory=memory)
    tool.set_memory_manager(memory)

    with open(os.path.join(workspace, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read()

    result_text = run_task(prompt, planner, coder, verifier, memory, metrics)

    # Capture the diff the agent produced as the candidate patch
    diff = subprocess.run(
        ["git", "-C", workspace, "diff"],
        capture_output=True, text=True,
    ).stdout

    last_record = memory.data["task_history"][-1] if memory.data["task_history"] else {}

    # Cleanup docker image to keep disk small (rolling mode)
    image = _read_sidecar_image(workspace)
    if image:
        print(f"  removing image {image} ...")
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    return {
        "instance_id": instance_name,
        "status": last_record.get("status", "unknown"),
        "attempts": last_record.get("attempts", 0),
        "files_changed": last_record.get("files_changed", []),
        "diff": diff,
        "result_text": result_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*__*", help="glob under WORKSHOP for instance dirs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default="swebench_report.jsonl")
    args = parser.parse_args()

    workspaces = sorted(
        d for d in glob.glob(os.path.join(WORKSHOP, args.pattern))
        if os.path.isfile(os.path.join(d, ".agent_memory.json"))
    )
    if args.limit:
        workspaces = workspaces[: args.limit]

    print(f"Running {len(workspaces)} SWE-bench instances ...")

    with open(args.report, "w", encoding="utf-8") as f:
        for ws in workspaces:
            print(f"\n========== {os.path.basename(ws)} ==========")
            try:
                result = run_one(ws)
            except Exception as e:
                result = {"instance_id": os.path.basename(ws), "status": "crashed", "error": str(e)}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
```

## P2.4 怎么跑

```bash
# 1) materialize first 5 SWE-bench Verified instances
#    (clones repos to cache + pulls docker images)
python runners/setup_swebench.py --limit 5

# 2) run agent on them (each instance: agent edits → docker run pytest → rmi)
python runners/run_swebench.py --limit 5 --report swebench_report.jsonl
```

任意时刻本地最多只占 1 个镜像（~2-3 GB），跑完一条立即清理，不会爆磁盘。

## P2.5 P2 阶段已知 caveats

1. **docker 镜像名格式**待确认。`docker_image_for_instance` 里写的 `swebench/sweb.eval.x86_64.<instance_id>:latest` 是 SWE-bench 文档的常见格式，但 instance_id 里的 `__` 在镜像 tag 中可能不允许，需要替换（我用了 `_1776_` 占位，实际规则待实测）。**Step 16 会先做单条 `docker pull` 验证**。如果跑不通，去 [https://hub.docker.com/r/swebench/sweb.eval.x86_64](https://hub.docker.com/r/swebench/sweb.eval.x86_64) 看看实际 tag 命名再回来调整 `docker_image_for_instance`。
2. **`environment_setup_commit`** 字段没用上。某些 instance 在 `base_commit` 上需要先回退到 `environment_setup_commit` 安装依赖再回到 `base_commit`，正确做法是参考官方 harness 的逻辑。**v1 先不管**，遇到具体 instance 装不上依赖再说。
3. **per-instance memory.json 命名**：MBPP 用 `memory.json`，SWE-bench 用 `.agent_memory.json`。原因是 SWE-bench workspace 是 git clone 出来的真实仓库，怕跟项目里可能存在的 `memory.json` 冲突，所以加点前缀。`.docker_image` 同理。
4. **`test_command` 里的引号转义**：`build_docker_test_command` 用的是 `bash -c "..."` 包裹内层 pytest，内层节点路径里有 `::` 这种字符，所以 `build_inner_pytest_command` 用 `\\"...\\"` 转义。如果你的某些 instance 的节点路径包含特殊字符（比如 `$`、`` ` ``），可能需要进一步转义。Step 19 跑通第一条之后再观察。

---

# 实施清单（分阶段）

按这个顺序写代码，每个阶段结束都能跑通做小验证：

## Step 0（前置改造）

- [ ] **Step 0** — `tool.py`：`from config import WORKSPACE` 改成 `import config`，所有 `WORKSPACE` 引用改成 `config.WORKSPACE`。详见 [P0.0 章节](#p00-step-0-前置改造--toolpy-切换到-import-config)。改完用 `grep -n "WORKSPACE" tool.py` 自检：所有引用都必须有 `config.` 前缀。

## P0 阶段（核心改造）

- [ ] **Step 1** — `config.py`：追加 verifier sandbox + memory B.2 的常量
- [ ] **Step 2** — 新增 `test_runner.py`（只有 SubprocessRunner，没有 docker 后端；用 `import config`）
- [ ] **Step 3** — 重写 `memory.py`（WorkingMemory + MemoryManager 两层；用 `import config`）
- [ ] **Step 4** — 改 `tool.py`：`save_memory` → working candidate、加 `_record_file_change` 和 `_record_observation`
- [ ] **Step 5** — 重写 `verifier.py`：接 `run_tests`，加 `files_changed` 参数；用 `import config` + `config.WORKSPACE`
- [ ] **Step 6** — 改 `coder.py`：注入 working memory snapshot
- [ ] **Step 7** — 改 `main.py`：`try / finally` + `begin_task` / `end_task` 包裹 `run_task`，给 Coder/Verifier 注入 memory
- [ ] **Step 8** — 在当前 `Execution/mini coding agent/` workspace 跑一个手工任务，确认：
  - [ ] verifier 输出里有 `[TestRunner]` block
  - [ ] task 结束后 `memory.json` 里 `task_history` 最后一条 `files_changed` 不为空
  - [ ] facts 只在 task 通过后才出现在 long-term

## P1 阶段（MBPP）

- [ ] **Step 9** — 装 `pip install datasets`（如果还没有）
- [ ] **Step 10** — 新增 `runners/__init__.py`（空文件）+ `runners/setup_mbpp.py`
- [ ] **Step 11** — 新增 `runners/run_mbpp.py`
- [ ] **Step 12** — `python runners/setup_mbpp.py --split test --limit 3` materialize 3 条（sanitized test split）
- [ ] **Step 13** — `python runners/run_mbpp.py --limit 3` 跑通
- [ ] **Step 14** — 看 `mbpp_report.json`，至少 1 条 passed 就算 P1 done

## P2 阶段（SWE-bench）

- [ ] **Step 15** — 新增 `runners/setup_swebench.py`
- [ ] **Step 16** — **先手动验证镜像名**：从 SWE-bench Verified 挑一条 instance，手动跑 `docker pull <你预测的镜像名>`，确认能拉下来。如果不行，调整 `setup_swebench.py:docker_image_for_instance` 的命名规则
- [ ] **Step 17** — `python runners/setup_swebench.py --limit 1` clone 1 条 + pull 镜像
- [ ] **Step 18** — 新增 `runners/run_swebench.py`
- [ ] **Step 19** — `python runners/run_swebench.py --limit 1`，看能不能跑完一条
- [ ] **Step 20** — 跑通 1 条 → 扩到 5 条 → 扩到 50 条

---

# 后续可选优化（不在本次范围）

- 真正的并行：现在 `run_mbpp.py` / `run_swebench.py` 是串行的，多核不利用
- 镜像 LRU 缓存：跑同一 repo 的多条 instance 时复用镜像而不是 rmi 后重新 pull
- Memory facts 加 embedding 检索：从 B.2 升级到 B.3
- Working memory 落盘 debug 模式：`--dump-working-memory` flag
- 接 SWE-bench 官方 `swebench.harness.run_evaluation` 出正式分数（论文用）

---

# 已知局限 (Limitations)

> 以下是当前 P0 设计中**有意接受**的局限。它们不是 bug，是工程取舍：为了让 P0 能快速跑通，主动放弃了一些高级能力，留给后续版本逐步修改。
> **重要**：未来如果发现这些"问题"，请先确认是不是已经在这个清单里——不要把"已知局限"当成"漏洞"去修复，那会破坏当前设计的内部一致性。

## L1. Memory：没有减分机制 (no demotion)

**现状**：fact 一旦进入 long-term，`confidence` 只能 `+0.2`，永远不会下降。即使这条 fact 是错的（例如 LLM 幻觉出来的表名/函数签名），系统也无法主动纠正它，只能等内存满时被淘汰策略被动清除——淘汰策略基于 `score = confidence × reinforce_count`，配合 `FACT_GRACE_PERIOD_TASKS` 保护新 fact。这个机制能让"加进来后从未被 reinforce 的烂 fact"逐步沉底淘汰，但**无法处理"被反复 reinforce 但其实是错的"fact**——这种 fact 会一直留下来。

**风险**：
- 错误 fact 可能被反复"验证"加分，最终 confidence 达到 1.0，比真实 fact 排得还前面
- 错误 fact 被注入未来 prompt，可能误导后续任务，产生连锁错误

**为什么接受**：
- 主动减分需要回答"凭什么减/减多少"——任务失败原因很多，归因很难
- 实现 LLM 自评/embedding 相似度检测会显著增加复杂度和 API 成本
- 当前依赖被动淘汰 + "宁可错杀"的 promote 策略作为补偿

**未来可能的改进方向**：
1. Verifier 在判失败时做归因，标记"哪条注入的 fact 与失败相关"，定向减分
2. 时间衰减：长时间未被验证的 fact 慢慢减分
3. LLM 裁判：promote 前用另一个 LLM 评估 fact 合理性

---

## L2. Memory：task 失败时 candidate facts 全部丢弃 (一刀切)

**现状**：task 失败时，working memory 里的所有 candidate facts 都被丢弃，不区分这条 fact 跟失败到底有没有关系。

**风险**：
- 任务可能因为一个跟 fact 无关的原因失败（例如算错边界值），却把同次任务里学到的好 fact（"项目用 pytest"）一起丢了
- 下次任务要重新学一遍同样的经验，浪费 token

**为什么接受**：
- 错杀代价低：重新读一次配置文件就能再学到
- 错放代价高：烂 fact 进 long-term 后**没有减分机制纠正**（见 L1）
- 在没有归因能力的前提下，保守策略是唯一安全选择

**未来可能的改进方向**：
- Verifier 做失败归因，只丢弃跟失败因果相关的 candidate fact
- 实现归因之后才能放宽这条策略

---

## L3. Memory：没有相关性检索 (no relevance retrieval)

**现状**：planner / coder 拉取 facts 注入 prompt 时，只按 confidence 排序取 top-N。没有根据当前任务内容做相关性筛选。

**风险**：
- 当前任务是修 auth 模块的 bug，但 memory 返回的可能是数据库相关的高 confidence fact，无关 facts 占用 prompt 空间
- 随着 facts 累积，注入的 context 信噪比逐渐下降

**为什么接受**：
- 相关性检索通常需要 embedding + 向量数据库，工程复杂度大
- P0 阶段优先把基础结构跑通，相关性检索属于 B.3 级别的增强

**未来可能的改进方向**：
- 升级到 B.3：facts 加 embedding，按 cosine 相似度检索
- 或者更简单：用关键词匹配做粗筛

---

## L4. Memory：fact 质量完全依赖 LLM 自觉 (soft constraint)

**现状**：`save_memory` 工具只是无脑写入接口，是否调用、写什么粒度的 fact 完全由 LLM 决定。系统没有任何机制阻止 LLM 写下任务细节级别的 fact（例如 `"user_id=42 是 admin"`），也没法阻止它存重复/低质内容。

**风险**：
- LLM 可能存一堆任务细节，污染 long-term memory
- LLM 可能完全不调用 save_memory，long-term 永远长不大

**为什么接受**：
- 加硬约束（例如正则检查、长度限制、分类强制）会让工具变臃肿，且容易误伤
- 当前依赖 prompt 工程：在 system prompt / tool description 里教 LLM "只存可泛化经验"
- 这是软约束，承认会有漂移

**未来可能的改进方向**：
- 在 promote 前加一道"fact 质量过滤"（关键词黑名单 / LLM 评分）
- 给 fact 加 `scope` 字段（task-local / project-wide），promote 时只升 project-wide 的

---

## L5. TestRunner：marker 探测只支持 Python

**现状**：`detect_test_command_by_marker` 只识别 `pytest.ini` / `pyproject.toml` / `tests/`，没支持 JS/Go/Rust 等其他语言的测试 marker。

**为什么接受**：
- 目标数据集（MBPP + SWE-bench）都是 Python 项目
- 多语言探测会让代码膨胀，违反 YAGNI

**未来可能的改进方向**：
- 加 `package.json` → `npm test` 等规则（按需添加）

---

## L6. TestRunner：不扫 workspace 根目录的 `test_*.py`

**现状**：marker 探测**故意不**扫 workspace 根目录散落的 `test_*.py` 文件。

**为什么接受**：
- 后期 MBPP / SWE-bench 的产物会放在 dataset-specific 子文件夹
- 不希望根目录一个 stray 测试文件就触发整个 pytest 探测
- 这是项目目录组织规约的一部分，不是技术限制

---

## L7. SubprocessRunner 是唯一后端

**现状**：test_runner 只有一个 SubprocessRunner，没有抽象基类、没有独立的 docker / k8s / 远程 runner。

**为什么接受**：
- SWE-bench 的 docker 用法是把 `docker run ...` 嵌进 `test_command` 字符串，subprocess 一样能跑
- 多 backend 抽象在当前需求下是过度设计

**未来可能的改进方向**：
- 真的需要远程执行/沙盒隔离时，再抽 `BaseRunner` 接口

---

文档已经融合所有补丁，可以直接按"实施清单" Step 0 → 20 顺序 implement。
