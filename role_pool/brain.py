"""Brain role for engine_fixed (plan 2 × tool 4 hardcoded pipeline).

Y vs X distinction (important — read before changing this file):

  In engine_fixed (Method Y), brain is a PLANNING role: it produces a JSON
  plan once (or twice with replan); host executes the plan in a fixed retry
  loop. Brain lives here in role_pool/ alongside worker roles, but it does
  NOT do any work — it only outputs a plan.

  In engine_brain (Method X, future), brain becomes an ORCHESTRATOR: it
  designs the entire workflow (including retry shapes), and consumes
  role_pool / tool_pool as a MENU. That brain will live OUTSIDE role_pool/
  (likely engine/brain.py once engine is split into a directory). The pools
  are still the menu; brain is independent of them.

  TL;DR: brain in role_pool/ is engine_fixed-specific. Don't confuse it
  with engine_brain's brain (which will live elsewhere).

Discovery mechanism:

  brain.py auto-discovers which roles/tools to show by reading SUPPORTED_TASKS
  and BRAIN_DESCRIPTION (or BRAIN_TOOLS) declared in each role/tool module.
  Add a new role for story task = add a new file under role_pool/ with
  SUPPORTED_TASKS = ["story"] and BRAIN_DESCRIPTION = "...". Brain will see
  it automatically next time build_system_prompt("story") is called.
"""

from __future__ import annotations

import json

from role_pool import (
    coder,
    dedup,
    planner,
    summarizer,
    verifier,
    writer,
)
from tool_pool import text_utils


# ---------------------------------------------------------------------------
# Discovery: role modules + tool modules brain may consult
# ---------------------------------------------------------------------------

# Map role name → module. brain.py reads each module's SUPPORTED_TASKS
# and BRAIN_DESCRIPTION attributes to decide what to show.
_ROLE_MODULES = {
    "writer": writer,
    "coder": coder,
    "planner": planner,
    "verifier": verifier,
    "summarizer": summarizer,
    "dedup": dedup,
}

# Tool modules. Each is expected to expose a BRAIN_TOOLS dict mapping
# tool_name → {"supported_tasks", "description", "callable"}. Modules
# without BRAIN_TOOLS are skipped (e.g. tool_pool/ops.py — those are
# worker-internal tools, not brain-dispatched).
_TOOL_MODULES = [text_utils]


def get_visible_roles(task_type: str) -> dict[str, str]:
    """Return {role_name: BRAIN_DESCRIPTION} for roles supporting task_type."""
    visible: dict[str, str] = {}
    for name, mod in _ROLE_MODULES.items():
        supported = getattr(mod, "SUPPORTED_TASKS", [])
        if task_type in supported:
            desc = getattr(mod, "BRAIN_DESCRIPTION", f"<no description for {name}>")
            visible[name] = desc
    return visible


def get_visible_tools(task_type: str) -> dict[str, str]:
    """Return {tool_name: description} for tools supporting task_type."""
    visible: dict[str, str] = {}
    for mod in _TOOL_MODULES:
        brain_tools = getattr(mod, "BRAIN_TOOLS", {})
        for tool_name, tool_meta in brain_tools.items():
            if task_type in tool_meta.get("supported_tasks", []):
                visible[tool_name] = tool_meta.get("description", "")
    return visible


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a planning orchestrator. Design a plan to
accomplish the user's task using the available roles and tools.

## Available roles (LLM workers)
{role_block}

## Available tools (Python functions, no LLM)
{tool_block}

## Execution context (FIXED — you cannot change this)
The host will execute your plan in a fixed retry loop:
  - Your plan specifies which role to call (e.g. writer) and the args
  - After each writer call, the host auto-verifies length with length_checker
  - If length != 241, host re-calls writer with feedback ("X chars, off by Y")
  - Up to 4 inner retries per brain attempt
  - If 4 retries fail, the host asks you to replan with the failure trace
  - Maximum 2 brain attempts total

## Your decision space
  - writer's guidance (the main lever you have)
  - strategy_notes (your reasoning, helps debugging — does not affect execution)

## Output format — JSON only, no markdown fences, no surrounding prose
{{
  "strategy_notes": "<your reasoning, e.g. 'writer tends to overshoot, aim for 230'>",
  "steps": [
    {{"role": "writer", "args": {{"theme": "<theme verbatim>", "guidance": "<your guidance>"}}}}
  ]
}}
"""


def build_system_prompt(task_type: str) -> str:
    """Build brain's system prompt with role / tool menus filtered by task_type."""
    roles = get_visible_roles(task_type)
    tools = get_visible_tools(task_type)
    role_block = "\n".join(roles.values()) or "(no roles available for this task type)"
    tool_block = "\n".join(tools.values()) or "(no tools available for this task type)"
    return PROMPT_TEMPLATE.format(role_block=role_block, tool_block=tool_block)


# ---------------------------------------------------------------------------
# Input builders — initial plan call vs replan call
# ---------------------------------------------------------------------------

def build_initial_input(theme_desc: str, target_len: int = 241) -> str:
    """User-message text for brain's FIRST call: just the task."""
    return (
        f"Task: Write a story about: {theme_desc}\n"
        f"Requirement: exactly {target_len} characters.\n\n"
        f"Design your plan. Output JSON only."
    )


def build_replan_input(previous_round: dict, target_len: int = 241) -> str:
    """User-message text for brain's SECOND call (replan).

    previous_round keys:
      plan            — the JSON plan brain produced last time
      inner_attempts  — list of {attempt: int, length: int, feedback_in: str}
      final_length    — length of the last attempt
    """
    plan = previous_round.get("plan")
    attempts = previous_round.get("inner_attempts", [])
    attempts_summary = "\n".join(
        f"  attempt {a['attempt']}: length={a['length']}, "
        f"feedback was: {a.get('feedback_in') or '(none — first attempt)'}"
        for a in attempts
    )
    final_length = previous_round.get("final_length", 0)
    diff = final_length - target_len
    return (
        f"Your previous plan failed. Trace:\n\n"
        f"Plan you proposed:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"Inner loop ran writer {len(attempts)} times with these results:\n"
        f"{attempts_summary}\n\n"
        f"Final attempt was {final_length} chars (target {target_len}, "
        f"off by {'+' if diff >= 0 else ''}{diff}).\n\n"
        f"Analyze why writer kept missing, then propose a new plan with different "
        f"guidance. Output JSON only."
    )


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------

def parse_plan(brain_output_text: str) -> dict | None:
    """Extract JSON plan from brain's text output. Returns None on parse failure.

    Tolerates: leading/trailing prose, ```json fences, etc. — tries best-effort
    extraction by finding the outermost JSON object braces.
    """
    text = (brain_output_text or "").strip()
    if not text:
        return None
    # Strip markdown fences if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: find outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
