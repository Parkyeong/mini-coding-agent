"""Task judge role: decide whether a task needs planner-driven decomposition.

Used by the c1_judge experimental config: before run_task starts, the engine
asks this judge "should we plan, or just code?". For tasks the judge calls
"simple", we skip planner entirely and let coder go straight to ReAct loop.
For "complex" tasks we keep the full planner → coder → verifier → summarizer
pipeline.

The judge does NOT solve the task or extract spec; it ONLY classifies.
Output is one binary word: "simple" or "complex".

Architecturally parallel to summarizer.py / dedup.py: PROMPT plus a tiny
helper, wrapped in an LLMNode that engine.build_llm_nodes constructs once
and run_task uses (or skips) per task.
"""
from __future__ import annotations

import re

PROMPT = """你判断一个编程任务是否需要多步 planning,还是 coder 直接写代码就能搞定。

输入: 任务描述 + 测试用例
输出: 一个词 — "simple" 或 "complex"

判定标准:
─ SIMPLE (coder 直接写就行):
    · 单个函数实现
    · 输入/输出清晰
    · 测试用例就是 IO 对的形式 (assert f(x) == y)
    · 不需要分步 (实现 + 测试 + 重构 不算分步,实现就完事)

─ COMPLEX (需要 planner 拆解):
    · 多文件改动
    · 需要先理解现有代码结构再改
    · 任务里明显有先后依赖步骤 (先做 A,再用 A 的结果做 B)
    · 多个独立子目标

【重要】你只判复杂度,不要去反推算法或猜实现。

输出格式: 只输出一个词,small letters,无标点。
"""


def _build_input(prompt: str, test_block: str = "") -> str:
    parts = [f"Task description:\n{prompt.strip()}"]
    if test_block.strip():
        parts.append(f"\nTest cases:\n{test_block.strip()}")
    return "\n".join(parts)


def judge_complexity(node, prompt: str, test_block: str = "") -> str:
    """Returns 'simple' or 'complex'. Defaults to 'complex' on any parse
    failure (defensive: better to use planner unnecessarily than skip when
    needed)."""
    node.reset_message()
    user_input = _build_input(prompt, test_block)
    result = node.run(user_input)
    text = (result.get("text") or "").strip().lower()

    # Strip punctuation/code fences
    text = re.sub(r"[^a-z]", "", text)

    if "simple" in text:
        return "simple"
    if "complex" in text:
        return "complex"
    # Garbled output → default to "complex" so we still use the planner
    # (avoids skipping it when the judge fails to commit to an answer).
    return "complex"
