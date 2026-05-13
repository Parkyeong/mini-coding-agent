"""Summarizer role: distill project-level lessons from a completed task.

After a case passes verification, the engine calls `summarize()` once. The
summarizer reviews the FULL trace — what the planner planned, what the coder
did, what the verifier returned, what failures occurred and how they were
fixed — and outputs 1-2 generalizable facts about the project. These facts
go into WorkingMemory.candidate_facts, then through the same promotion path
that coder's save_memory used to feed.

Why a separate role (instead of letting coder write facts as it goes):
  - Coder's save_memory historically captured implementation details ("used
    re.search for this regex problem") that are useless for *future* tasks
  - Planner reads the global pool to make plans, so the writer should look
    at things from a whole-system perspective, not coder's tactical view
  - Removing save_memory from coder's toolbox also reduces the over-execution
    pattern (coder kept calling save_memory after tests already passed)

Architecturally parallel to planner.py / coder.py / dedup.py: PROMPT plus
a couple of helper functions, wrapped in an LLMNode by engine.build_llm_nodes.
"""
from __future__ import annotations

import json
import re

# Brain-facing metadata: summarizer is invoked by the engine post-task, not
# brain-dispatched, so it has no SUPPORTED_TASKS (empty list = invisible to
# brain). BRAIN_DESCRIPTION is omitted for the same reason.
SUPPORTED_TASKS: list[str] = []

PROMPT = """You review the trace of a COMPLETED coding task — including the plan that
was made, the code the agent wrote, the test results, and any retry/replan
that happened — and extract 1-2 generalizable lessons about the PROJECT.

You see the WHOLE process, not one role's view: what the planner decided,
what the coder did, what the verifier returned, what failures repeated,
what fixes worked. Your output joins a project-level knowledge base read by
future planners and coders BEFORE they tackle their own tasks.

GOOD lessons (project-level, transfer to OTHER tasks):
  - "this project uses pytest -q --disable-warnings; tests run quietly
     and warnings don't surface in failure output"
  - "function signatures (name + parameter count) must exactly match the
     imports/asserts in test_solution.py"
  - "test_solution.py is read-only; all fixes must go in solution.py"
  - "replace_in_file often fails on code blocks containing whitespace
     ambiguity — full write_file is more reliable for nontrivial edits"

BAD lessons (task-specific, useless for unrelated tasks):
  - "case 0011 uses a set to track seen characters"
  - "I imported re for this regex problem"
  - "the answer for this case is [1, 2, 3, 5, 7]"

Output ONE strong lesson when the task revealed something about the project.
Output two only if there are genuinely two distinct insights. If the task
was straightforward and revealed nothing new about the project conventions,
output an empty array [].

OUTPUT FORMAT — JSON only, no prose, no code fences, no explanation:

[{"fact": "...", "category": "..."}]

category should be one short tag like:
  testing | convention | debugging | implementation_pattern | tooling
"""


# ---------------------------------------------------------------------------
# Case trace builder — turns working_memory + workspace into a markdown that
# the summarizer LLM can read efficiently. We deliberately compress: long
# tool args / results get truncated, code dumps get a head+tail snippet, etc.
# ---------------------------------------------------------------------------

_TOOL_ARG_PREVIEW = 80      # chars per tool-call arg in the timeline
_TOOL_RESULT_PREVIEW = 240  # chars per tool-result in the timeline


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _format_tool_call(name: str, args: dict) -> str:
    """One-line preview of a tool call for the timeline."""
    if not isinstance(args, dict):
        return f"{name}(...)"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f"{k}={_short(v, _TOOL_ARG_PREVIEW)!r}")
        else:
            parts.append(f"{k}={v}")
    return f"{name}({', '.join(parts)})"


def _format_tool_result(result) -> str:
    if isinstance(result, dict):
        return _short(json.dumps(result), _TOOL_RESULT_PREVIEW)
    return _short(str(result), _TOOL_RESULT_PREVIEW)


def build_case_trace(memory, env=None) -> str:
    """Build a compact markdown trace of one task's execution.

    Includes: original prompt, plan(s), per-step action timeline (planner /
    coder / verifier events), final solution.py contents, final test result.
    Long fields are truncated so the trace stays under a few thousand tokens.
    """
    wm = memory.get_working() if memory is not None else None
    if wm is None:
        return ""

    parts: list[str] = []

    parts.append("# Task")
    parts.append("")
    parts.append(wm.user_prompt.strip())
    parts.append("")

    if wm.plan:
        parts.append("# Plan (final, after any replans)")
        parts.append("")
        for i, step in enumerate(wm.plan, 1):
            parts.append(f"{i}. {step}")
        parts.append("")

    parts.append("# Action timeline")
    parts.append("")
    for ev in wm.event_log.events:
        kind = ev.get("kind")
        p = ev.get("payload", {}) or {}
        if kind == "llm_call":
            role = p.get("role", "?")
            parts.append(f"- [{role}] llm_call (in={p.get('input_tokens', 0)}, "
                         f"out={p.get('output_tokens', 0)})")
        elif kind == "text":
            content = _short(p.get("content", ""), 200)
            if content:
                parts.append(f"  text: {content}")
        elif kind == "tool_call":
            call_str = _format_tool_call(p.get("name", "?"), p.get("args", {}))
            parts.append(f"  → {call_str}")
        elif kind == "tool_result":
            name = p.get("name", "?")
            res = _format_tool_result(p.get("result", ""))
            parts.append(f"  ← {name} → {res}")
    parts.append("")

    # Final solution.py — most important single piece of evidence.
    if env is not None:
        try:
            sol = env.read_file("solution.py")
            parts.append("# Final solution.py")
            parts.append("")
            parts.append("```python")
            parts.append(sol.rstrip())
            parts.append("```")
            parts.append("")
        except Exception:
            # File missing or unreadable — skip rather than erroring out.
            pass

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call + JSON parse
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """The PROMPT says 'no code fences' but LLMs sometimes ignore it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_facts(text: str) -> list[dict]:
    """Parse the model output into a list of {fact, category} dicts.
    Returns [] on any parse failure — fail-soft, summarization is best-effort."""
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    valid: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fact = (item.get("fact") or "").strip()
        category = (item.get("category") or "").strip() or "general"
        if fact:
            valid.append({"fact": fact, "category": category})
    # Cap at 3 — defends against runaway output even though prompt asks for 1-2.
    return valid[:3]


def summarize(node, memory, env=None) -> list[dict]:
    """Run the summarizer LLMNode on a completed task and return parsed facts.

    Engine should call this AFTER `overall_passed=True` and BEFORE `end_task`,
    so the resulting facts get added to WorkingMemory.candidate_facts and
    flow through the standard promotion path.
    """
    trace = build_case_trace(memory, env)
    if not trace.strip():
        return []
    node.reset_message()
    result = node.run(trace)
    return _parse_facts(result.get("text", ""))
