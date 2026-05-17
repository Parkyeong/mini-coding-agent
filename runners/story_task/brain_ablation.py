"""brain_ablation.py — controlled ablation of method_brain's test0→test1 changes.

Three orthogonal dimensions:

    writer_mode   : v0 (simple desc, test0)            | v1 (two-mode desc, test1)
    example_mode  : v0_loop (single feedback loop)     | v1_bracket (bracket-and-fix)
                  | neutral (minimal DSL syntax demo only — no strategy hint)
    memory_mode   : v0 (flat last-cycle)               | v1 (last + nearest)

Four conditions, each varying ONE knob relative to the test0 baseline:

    a (baseline)        : writer=v0, example=v0_loop,    memory=v0
                          — reproduces test0 brain state
    b (memory only)     : writer=v0, example=v0_loop,    memory=v1
                          — isolates the cross-run memory upgrade
    c (writer only)     : writer=v1, example=v1_bracket, memory=v0
                          — isolates the writer-description + example upgrade
                          (these were bundled in the test0→test1 change)
    d (neutral example) : writer=v0, example=neutral,    memory=v0
                          — strips any strategy hint from the example; tests
                          what brain invents on its own with only DSL syntax
                          shown. Comparing d vs a tells us how much the v0
                          loop example anchored brain in test0.

Same outer loop shape as method_brain (8 cycles per run, brain re-called each
cycle, per-cycle writer cap = 3). All other infrastructure (DSL interpreter,
writer.py's host-side auto-direction, length_checker including absdiff,
config.py role configs) is shared with method_brain and stays at its current
state — this file only overrides the bits that varied between test0 and test1.

Independent from method_brain.py: it imports stable DSL primitives but never
mutates anything; method_brain runs unaffected.

Outputs:
    Execution/story_241/<exp>/brain_ablation_a/...
    Execution/story_241/<exp>/brain_ablation_b/...
    Execution/story_241/<exp>/brain_ablation_c/...

Usage:
    python -m runners.story_task.brain_ablation --condition a
    python -m runners.story_task.brain_ablation --condition b
    python -m runners.story_task.brain_ablation --condition c
    python -m runners.story_task.brain_ablation --condition d
    python -m runners.story_task.brain_ablation --all     # run all 4 sequentially

Env vars STORY_THEMES / STORY_RUNS_PER_THEME / STORY_EXP_NAME apply the same
way they do for method_brain (run_all sets them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP, ROLE_CONFIGS
from llm_node import LLMNode
from metrics import MetricsTracker
from role_pool import writer as writer_role
from role_pool.brain import get_visible_tools

# Reuse method_brain's stable DSL primitives (they're not part of what varies).
from runners.story_task.method_brain import (
    ExecutionContext,
    WorkflowError,
    execute,
    parse_workflow,
    _call_writer,            # noqa: F401  — implicitly used by `execute`
    _call_length_checker,    # noqa: F401
)


# ---------------------------------------------------------------------------
# Task spec — locked to match method_brain
# ---------------------------------------------------------------------------

_AVAILABLE_THEMES: list[tuple[str, str]] = [
    ("mountain_school",       "The lone teacher at a remote mountain school"),
    ("time_displaced_store",  "A convenience store displaced in time"),
    ("photo_studio_last_day", "The final day of an old photo studio"),
    ("rainy_night_bus",       "The last bus on a rainy night"),
]

_theme_filter = os.environ.get("STORY_THEMES", "").strip()
if _theme_filter:
    _wanted = {t.strip() for t in _theme_filter.split(",") if t.strip()}
    THEMES = [t for t in _AVAILABLE_THEMES if t[0] in _wanted]
else:
    THEMES = list(_AVAILABLE_THEMES)

RUNS_PER_THEME = int(os.environ.get("STORY_RUNS_PER_THEME", "4"))
TARGET_LEN = 241
TASK_TYPE = "story"
NUM_CYCLES = 8
WRITER_CALL_CAP_PER_CYCLE = 3

_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

# --- Writer description / example workflow: V0 = test0-era, V1 = test1-era ---

BRAIN_DESC_V0_WRITER = """writer (LLM worker, gpt-4o-mini):
    Args: theme (str, required), guidance (str, optional)
    Returns: draft story text
    Notes: tends to over/undershoot length targets by 5-30 characters.
"""

BRAIN_DESC_V1_WRITER = """writer (LLM worker, gpt-4o-mini):
    Args:
      theme (str, required)
      guidance (str, optional)         — your strategic instruction
      previous_attempt (str, optional) — full text of the prior draft

    TWO MODES (you choose by what you pass as previous_attempt):

    ─ FRESH DRAFT mode  (previous_attempt = "")
      Writer drafts from theme + your guidance. No host auto-direction.
      Use when you want a brand-new attempt unanchored to prior drafts.

    ─ MINIMAL EDIT mode  (previous_attempt = "<prior draft>")
      The HOST automatically computes diff = 241 − len(previous_attempt)
      and INJECTS a precise direction into writer's user message:

        "This is N characters — exactly M too long/short for the 241
         target. Output the SAME story with exactly M characters
         DELETED/APPENDED from the end (or tightened in place).
         Do NOT rewrite. Do NOT change the plot."

      Your `guidance` LAYERS ON TOP of this auto-direction. Examples:

      {
        "aligned_with_host": [
          "tighten the final clause",
          "remove an adverb or punctuation",
          "prefer cutting filler words near the end",
          "preserve plot and tone"
        ],
        "conflicting_with_host": [
          {"guidance": "rewrite the opening",
           "why": "host says do NOT rewrite"},
          {"guidance": "edit the middle sentence only",
           "why": "host says edit at the end"},
          {"guidance": "change the protagonist's name",
           "why": "host says only trim/append"},
          {"guidance": "compress phrases throughout",
           "why": "host narrows scope to the tail"}
        ]
      }

      If you actually need a different action (fresh start, structural
      change), use FRESH DRAFT mode instead (pass "" for
      previous_attempt). Don't try to override host inside MINIMAL EDIT.

    Notes:
      - $var only substitutes whole-string. "$last.diff chars" embedded
        inside a longer string stays LITERAL — useless. The host already
        injects the exact number in its auto-direction, so you don't
        need to.
      - Writer is gpt-4o-mini: minimal-edit accuracy is high when
        |diff| ≤ 15. For |diff| > 30, prefer FRESH DRAFT mode (cheaper
        than dragging a wrong draft toward the target step by step).
    Returns: story text
"""


EXAMPLE_V0 = """# Example workflow

A simple single-loop with feedback-driven revision. On each iteration after
the first, writer sees `$draft` (the previous story text) AND `$last.delta_text`
(the length error). On the first iteration both resolve to "" and writer
starts fresh.

```json
{{
  "strategy_notes": "Single phase. Writer revises $draft based on $last.delta_text. First iter both are empty so it starts fresh.",
  "workflow": {{
    "type": "loop",
    "max_iter": 8,
    "until": "$last.hit == true",
    "body": {{
      "type": "sequence",
      "steps": [
        {{
          "type": "call",
          "role": "writer",
          "args": {{
            "theme": "$theme",
            "guidance": "aim 241 exactly; story arc with strong ending",
            "previous_attempt": "$draft",
            "feedback": "$last.delta_text"
          }},
          "save_as": "draft"
        }},
        {{
          "type": "call",
          "tool": "length_checker",
          "args": {{"text": "$draft", "target": "$target_len"}},
          "save_as": "last"
        }}
      ]
    }}
  }}
}}
```

Writer's recognized args: `theme`, `guidance`, `feedback`, `previous_attempt`.
Pass `previous_attempt` for precise char-level revision (recommended for
length-constrained tasks).

You don't have to use this shape — you can design any control flow that fits
the task."""


EXAMPLE_V1 = """# Example workflow — "bracket-and-fix" (empirically effective pattern)

3-writer pattern shown to converge well within one cycle:
  1. FRESH DRAFT aiming UNDER target (~231 chars)
  2. FRESH DRAFT aiming OVER target (~251 chars)
  3. MINIMAL EDIT on the closer one (host auto-direction handles the
     precise delete/append based on the actual length)

```json
{{
  "strategy_notes": "Bracket-and-fix: two fresh drafts above and below target as a length envelope, then minimal-edit one of them to 241. Final-stage guidance reinforces host's tail-trim semantics.",
  "workflow": {{
    "type": "sequence",
    "steps": [
      {{"type": "call", "role": "writer",
        "args": {{"theme": "$theme",
                 "guidance": "Fresh concise draft. Aim ~231 chars.",
                 "previous_attempt": ""}},
        "save_as": "draft_short"}},
      {{"type": "call", "tool": "length_checker",
        "args": {{"text": "$draft_short", "target": "$target_len"}},
        "save_as": "check_short"}},
      {{"type": "if", "condition": "$check_short.hit == true",
        "then": {{"type": "return", "value": "$draft_short"}}}},
      {{"type": "call", "role": "writer",
        "args": {{"theme": "$theme",
                 "guidance": "Fresh vivid draft. Aim ~251 chars.",
                 "previous_attempt": ""}},
        "save_as": "draft_long"}},
      {{"type": "call", "tool": "length_checker",
        "args": {{"text": "$draft_long", "target": "$target_len"}},
        "save_as": "check_long"}},
      {{"type": "if", "condition": "$check_long.hit == true",
        "then": {{"type": "return", "value": "$draft_long"}}}},
      {{"type": "if",
        "condition": "$check_short.absdiff <= $check_long.absdiff",
        "then": {{"type": "call", "role": "writer",
                 "args": {{"theme": "$theme",
                          "guidance": "Tighten the ending only. Preserve plot and tone.",
                          "previous_attempt": "$draft_short"}},
                 "save_as": "draft_final"}},
        "else": {{"type": "call", "role": "writer",
                 "args": {{"theme": "$theme",
                          "guidance": "Tighten the ending only. Preserve plot and tone.",
                          "previous_attempt": "$draft_long"}},
                 "save_as": "draft_final"}}}},
      {{"type": "call", "tool": "length_checker",
        "args": {{"text": "$draft_final", "target": "$target_len"}},
        "save_as": "last"}}
    ]
  }}
}}
```

Use `$check.absdiff` (absolute distance) when comparing candidates'
closeness — `.diff` is signed and can flip intuition when one candidate
is short and the other is long."""


EXAMPLE_NEUTRAL = """# Example workflow (minimal — DSL syntax only)

This example is intentionally minimal: one writer call + one length check.
It does NOT suggest any strategy (no loop, no retry, no bracketing). Use
it as a syntax reference, then design your own workflow shape based on
the task, the within-run history, and the cross-run memory.

```json
{{
  "strategy_notes": "Minimal demonstration of DSL syntax only.",
  "workflow": {{
    "type": "sequence",
    "steps": [
      {{"type": "call", "role": "writer",
        "args": {{"theme": "$theme", "previous_attempt": ""}},
        "save_as": "draft"}},
      {{"type": "call", "tool": "length_checker",
        "args": {{"text": "$draft", "target": "$target_len"}},
        "save_as": "last"}}
    ]
  }}
}}
```

The DSL also supports `loop`, `if`, and `return` nodes (see node types
above). Whether to use them, how to combine them, and what shape of
strategy to design — all your choice."""


# --- Brain system prompt template (parameterized by writer / example) ---

BRAIN_PROMPT_TEMPLATE = """You are a workflow architect operating in an
ITERATIVE outer loop. Your job each turn is to design ONE workflow PROGRAM
(in the JSON DSL below) for THIS cycle of an 8-cycle agent loop.

You do NOT execute the workflow yourself — a Python interpreter will run it.
After execution, the host calls YOU AGAIN with the cycle's outcome appended
to the within-run history, and you design the NEXT cycle's workflow. The
loop ends early when any verifier returns HIT (length == 241).

# Workflow DSL

A workflow is a tree of nodes. Each node is a JSON object with a `type` field.

## Node types

### `call` — invoke a role or a tool
```
{{"type": "call", "role": "writer", "args": {{...}}, "save_as": "varname"}}
{{"type": "call", "tool": "length_checker", "args": {{...}}, "save_as": "varname"}}
```

### `sequence` — run children in order
```
{{"type": "sequence", "steps": [<node1>, <node2>, ...]}}
```

### `loop` — repeat body until condition (or max_iter)
```
{{"type": "loop", "max_iter": 8, "until": "<condition>", "body": <node>}}
```

### `if` — conditional branch
```
{{"type": "if", "condition": "<expr>", "then": <node>, "else": <node>}}
```

### `return` — terminate the workflow immediately
```
{{"type": "return", "value": "$some_var"}}
```

## Variables

- Initial variables: `$theme`, `$target_len` (= 241).
- SET via `save_as` on `call` nodes.
- READ via `$varname` or `$varname.field`.
- Missing variables silently resolve to empty string (no error).

## Conditions

Binary expressions: `<expr> OP <expr>` where OP ∈ {{==, !=, <, >, <=, >=}}.
Operands can be `$var`, `$var.field`, or a literal (int / bool / string).

# Available roles (LLM workers)

{role_block}

# Available tools (Python functions, no LLM)

{tool_block}

# Execution constraints (FIXED by host, you cannot bypass)

- **Per-cycle writer cap: {writer_cap} writer calls.** Within ONE cycle's
  workflow, host silently skips any writer call beyond #{writer_cap}
  (the skipped call's `save_as` won't be set). Cap resets next cycle —
  you have {total_cycles} cycles total, so up to {total_writer_max}
  writer calls across the whole run.
- SUCCESS: any length_checker call returning length == 241 ends the run
  immediately.
- Within a cycle, design a small focused workflow (1-3 writer calls).

{example_block}

# Output format

JSON only. No markdown fences, no surrounding prose, no explanation. Your
entire response must be a single JSON object with `strategy_notes` and
`workflow` keys.
"""


_EXAMPLES_BY_MODE = {
    "v0_loop":    EXAMPLE_V0,
    "v1_bracket": EXAMPLE_V1,
    "neutral":    EXAMPLE_NEUTRAL,
}


def build_brain_system_prompt(writer_mode: str, example_mode: str) -> str:
    """Render BRAIN_PROMPT_TEMPLATE with chosen writer description + example.

    writer_mode  : 'v0' (simple desc) | 'v1' (two-mode desc)
    example_mode : 'v0_loop' | 'v1_bracket' | 'neutral'

    These two knobs are independent so we can mix any (description, example)
    combination — e.g. v0 desc with neutral example (condition d).
    """
    writer_desc = (BRAIN_DESC_V1_WRITER if writer_mode == "v1"
                   else BRAIN_DESC_V0_WRITER)
    example = _EXAMPLES_BY_MODE.get(example_mode, EXAMPLE_V0)

    # Only writer + length_checker are in the menu (matches method_brain).
    role_block = writer_desc.strip()
    tools = get_visible_tools(TASK_TYPE)
    tool_block = "\n".join(tools.values()) or "(none)"

    return BRAIN_PROMPT_TEMPLATE.format(
        role_block=role_block,
        tool_block=tool_block,
        writer_cap=WRITER_CALL_CAP_PER_CYCLE,
        total_cycles=NUM_CYCLES,
        total_writer_max=NUM_CYCLES * WRITER_CALL_CAP_PER_CYCLE,
        example_block=example,
    )


# --- Memory entry building / rendering (v0 flat / v1 nested) ---

def _cycle_snapshot(c: dict | None) -> dict | None:
    if c is None:
        return None
    return {
        "cycle": c.get("cycle"),
        "final_length": c.get("final_length"),
        "hit": c.get("hit", False),
        "strategy_notes": c.get("strategy_notes", ""),
        "workflow_json": c.get("workflow_json"),
        "final_story": c.get("final_story", ""),
    }


def build_memory_entry(run_record: dict, memory_mode: str) -> dict:
    """Build the cross-run memory entry brain will see for THIS run.

    v0 (flat): only the run's last cycle, like test0.
    v1 (nested): last + nearest (smallest |final_length - target|).
    """
    cycles = run_record.get("cycles") or []
    last_cycle = cycles[-1] if cycles else None

    if memory_mode == "v1":
        nearest_cycle = (
            min(cycles, key=lambda c: abs(c.get("final_length", 0) - TARGET_LEN))
            if cycles else None
        )
        return {
            "run_idx": run_record["run_idx"],
            "cycles_used": run_record["cycles_used"],
            "hit": run_record["hit"],
            "final_length": run_record["final_length"],
            "last_cycle": _cycle_snapshot(last_cycle),
            "nearest_cycle": _cycle_snapshot(nearest_cycle),
        }
    # v0 flat shape
    return {
        "run_idx": run_record["run_idx"],
        "cycles_used": run_record["cycles_used"],
        "hit": run_record["hit"],
        "final_length": run_record["final_length"],
        "final_story": last_cycle.get("final_story", "") if last_cycle else "",
        "workflow_json": last_cycle.get("workflow_json") if last_cycle else None,
        "strategy_notes": last_cycle.get("strategy_notes", "") if last_cycle else "",
    }


def _render_cross_run_memory_v0(memory_context: list[dict]) -> str:
    if not memory_context:
        return ""
    lines = [
        "## Cross-run memory (previous runs of this SAME theme; last-cycle snapshot)",
        "",
        "If past runs kept missing, change strategy meaningfully.",
        "",
    ]
    for m in memory_context:
        outcome = "Pass" if m.get("hit") else "Fail"
        lines.append(f"### Run {m.get('run_idx', '?')} "
                     f"({m.get('cycles_used', '?')} cycles, {outcome}, "
                     f"final length {m.get('final_length', '?')})")
        sn = (m.get("strategy_notes") or "").strip()
        if sn:
            lines.append(f"strategy_notes: {sn}")
        wf = m.get("workflow_json")
        if wf is not None:
            try:
                wf_str = json.dumps(wf, ensure_ascii=False, indent=2)
            except Exception:
                wf_str = str(wf)
            lines.append("last workflow:")
            lines.append("```json")
            lines.append(wf_str)
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_cross_run_memory_v1(memory_context: list[dict]) -> str:
    if not memory_context:
        return ""

    def _render_snap(label: str, snap: dict | None) -> list[str]:
        if not snap:
            return []
        cyc = snap.get("cycle", "?")
        fl = snap.get("final_length", "?")
        passed = "Pass" if snap.get("hit") else "Fail"
        out = [f"**{label}** (cycle {cyc}, length {fl}, {passed}):"]
        sn = (snap.get("strategy_notes") or "").strip()
        if sn:
            out.append(f"strategy_notes: {sn}")
        wf = snap.get("workflow_json")
        if wf is not None:
            try:
                wf_str = json.dumps(wf, ensure_ascii=False, indent=2)
            except Exception:
                wf_str = str(wf)
            out.append("workflow:")
            out.append("```json")
            out.append(wf_str)
            out.append("```")
        return out

    lines = [
        "## Cross-run memory (previous runs of this SAME theme)",
        "",
        "Each previous run gives two snapshots:",
        "  - LAST cycle: where the run ended (safe fallback if Fail)",
        "  - NEAREST cycle: the closest-to-target attempt across that run",
        "    (= LAST when the run Passed; differs when an earlier cycle",
        "    was closer than where the run finally ended)",
        "",
    ]
    for m in memory_context:
        run_idx = m.get("run_idx", "?")
        cycles_used = m.get("cycles_used", "?")
        outcome = "Pass" if m.get("hit") else "Fail"
        lines.append(f"### Run {run_idx} ({cycles_used} cycles, {outcome})")
        last_c = m.get("last_cycle")
        near_c = m.get("nearest_cycle")
        lines.extend(_render_snap("LAST cycle", last_c))
        if near_c and last_c and near_c.get("cycle") != last_c.get("cycle"):
            lines.extend(_render_snap("NEAREST cycle", near_c))
        elif near_c and last_c and near_c.get("cycle") == last_c.get("cycle"):
            lines.append("(NEAREST cycle is the same as LAST cycle.)")
        lines.append("")
    return "\n".join(lines)


def _render_within_run_history(history: list[dict]) -> str:
    """Within-run history (unchanged between test0 and test1)."""
    if not history:
        return ""
    lines = [
        "## Within-run history (cycles you have already designed in THIS run)",
        "",
    ]
    for c in history:
        cycle_idx = c.get("cycle", "?")
        outcome = "Pass" if c.get("hit") else "Fail"
        final_len = c.get("final_length", "?")
        lines.append(f"### Cycle {cycle_idx} ({outcome}, final length {final_len})")
        sn = (c.get("strategy_notes") or "").strip()
        if sn:
            lines.append(f"strategy_notes: {sn}")
        wf = c.get("workflow_json")
        if wf is not None:
            try:
                wf_str = json.dumps(wf, ensure_ascii=False, indent=2)
            except Exception:
                wf_str = str(wf)
            lines.append("workflow:")
            lines.append("```json")
            lines.append(wf_str)
            lines.append("```")
        fs = (c.get("final_story") or "").strip()
        if fs:
            lines.append(f"final_story ({len(fs)} chars):")
            lines.append(f"  \"{fs}\"")
        lines.append("")
    return "\n".join(lines)


def build_cycle_input(theme_desc: str, cycle_idx: int, total_cycles: int,
                      memory_mode: str,
                      cross_run_memory: list[dict] | None = None,
                      within_run_history: list[dict] | None = None) -> str:
    parts = [
        f"## Task",
        f"Write a story about: {theme_desc}",
        f"Requirement: exactly {TARGET_LEN} characters.",
        "",
        f"## Cycle position",
        f"You are at the START of cycle {cycle_idx} of {total_cycles}.",
        f"Cycles remaining (including this one): {total_cycles - cycle_idx + 1}.",
        "",
    ]
    render_xrun = (_render_cross_run_memory_v1 if memory_mode == "v1"
                   else _render_cross_run_memory_v0)
    cross_block = render_xrun(cross_run_memory or [])
    if cross_block:
        parts.append(cross_block)
    within_block = _render_within_run_history(within_run_history or [])
    if within_block:
        parts.append(within_block)
    parts.append("## Your turn")
    if cross_run_memory or within_run_history:
        parts.append(
            f"Design the cycle-{cycle_idx} workflow JSON. If past attempts "
            "kept missing, change strategy meaningfully — don't repeat the "
            "same shape. Output JSON only."
        )
    else:
        parts.append(f"Design your cycle-{cycle_idx} workflow JSON. Output JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# One run (NUM_CYCLES cycles, brain re-called each)
# ---------------------------------------------------------------------------

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str,
            writer_mode: str, memory_mode: str, example_mode: str,
            memory_context: list[dict] | None = None) -> dict:
    metrics = MetricsTracker()
    memory_context = memory_context or []

    brain_cfg = ROLE_CONFIGS["brain"]
    sys_prompt = build_brain_system_prompt(writer_mode, example_mode)
    trajectory: list[dict] = []
    step_counter = [0]

    def next_step() -> int:
        step_counter[0] += 1
        return step_counter[0]

    cycles: list[dict] = []
    final_text = ""
    final_length = 0
    hit = False
    overall_error = None

    for cycle_idx in range(1, NUM_CYCLES + 1):
        brain_node = LLMNode(
            system_prompt=sys_prompt,
            role="brain",
            max_steps=brain_cfg["max_steps"],
            model=brain_cfg["model"],
            temperature=brain_cfg["temperature"],
            max_tokens=brain_cfg["max_tokens"],
            metrics_tracker=metrics,
        )
        prior_for_brain = [
            {
                "cycle": c["cycle"],
                "workflow_json": c["workflow_json"],
                "final_story": c["final_story"],
                "final_length": c["final_length"],
                "hit": c["hit"],
                "strategy_notes": c["strategy_notes"],
            }
            for c in cycles
        ]
        user_input = build_cycle_input(
            theme_desc=theme_desc,
            cycle_idx=cycle_idx,
            total_cycles=NUM_CYCLES,
            memory_mode=memory_mode,
            cross_run_memory=memory_context,
            within_run_history=prior_for_brain,
        )

        n0 = len(metrics.calls)
        brain_response = brain_node.run(user_input)
        brain_output = (brain_response.get("text") or "").strip()
        new_calls = metrics.calls[n0:]
        brain_tokens = {
            "in": sum(c.input_tokens for c in new_calls),
            "out": sum(c.output_tokens for c in new_calls),
        }
        trajectory.append({
            "step": next_step(),
            "role": "brain",
            "purpose": f"design cycle {cycle_idx} workflow",
            "input": user_input,
            "output": brain_output,
            "tokens": brain_tokens,
        })
        mem_note = (f" (sees {len(memory_context)} cross-run + "
                    f"{len(prior_for_brain)} within-run)"
                    if memory_context or prior_for_brain else "")
        print(f"  cycle {cycle_idx}: brain design{mem_note}", flush=True)

        plan = parse_workflow(brain_output)
        if plan is None or "workflow" not in plan:
            trajectory.append({
                "step": next_step(),
                "role": "system",
                "purpose": f"PLAN_UNPARSEABLE (cycle {cycle_idx})",
                "input": None,
                "output": "Could not parse brain output as JSON with a 'workflow' key",
                "tokens": {"in": 0, "out": 0},
            })
            cycles.append({
                "cycle": cycle_idx,
                "workflow_json": None,
                "strategy_notes": "",
                "final_story": final_text,
                "final_length": final_length,
                "hit": False,
                "writer_calls_used_in_cycle": 0,
                "strategy_validated": False,
                "error": "plan_unparseable",
            })
            overall_error = "plan_unparseable"
            continue

        workflow = plan.get("workflow")
        strategy_notes = plan.get("strategy_notes", "")

        ctx = ExecutionContext(
            theme=theme_desc, target_len=TARGET_LEN,
            metrics=metrics, trajectory=trajectory,
            starting_step=step_counter[0],
        )
        # Per-cycle cap (overrides ExecutionContext's default).
        ctx.writer_cap = WRITER_CALL_CAP_PER_CYCLE

        cycle_error = None
        try:
            execute(workflow, ctx)
        except WorkflowError as e:
            cycle_error = f"workflow_error: {e}"
            ctx.step += 1
            ctx.trajectory.append({
                "step": ctx.step,
                "role": "system",
                "purpose": f"INTERPRETER_ERROR (cycle {cycle_idx})",
                "input": None,
                "output": str(e),
                "tokens": {"in": 0, "out": 0},
            })
        step_counter[0] = ctx.step

        if ctx.last_text:
            final_text = ctx.last_text
            final_length = ctx.last_length
        cycle_hit = ctx.last_length == TARGET_LEN
        strategy_validated = ctx.writer_calls >= 1 and not (
            cycle_hit and ctx.writer_calls == 1
        )

        cycles.append({
            "cycle": cycle_idx,
            "workflow_json": workflow,
            "strategy_notes": strategy_notes,
            "final_story": final_text,
            "final_length": final_length,
            "hit": cycle_hit,
            "writer_calls_used_in_cycle": ctx.writer_calls,
            "strategy_validated": strategy_validated,
            "error": cycle_error,
        })

        if cycle_hit:
            hit = True
            print(f"  cycle {cycle_idx}: Pass — exiting run early", flush=True)
            break

    # Save story
    theme_dir = os.path.join(output_dir, theme_id)
    os.makedirs(theme_dir, exist_ok=True)
    with open(os.path.join(theme_dir, f"run_{run_idx}.txt"),
              "w", encoding="utf-8") as f:
        f.write(final_text)

    by_role = metrics.by_role()
    total_in = sum(r["input_tokens"] for r in by_role.values())
    total_out = sum(r["output_tokens"] for r in by_role.values())
    total_writer_calls = sum(c.get("writer_calls_used_in_cycle", 0) for c in cycles)

    return {
        "theme_id": theme_id,
        "theme_desc": theme_desc,
        "run_idx": run_idx,
        "hit": hit,
        "final_length": final_length,
        "cycles_used": len(cycles),
        "writer_calls_used": total_writer_calls,
        "cycles": cycles,
        "trajectory": trajectory,
        "memory_seen": list(memory_context or []),
        "error": overall_error,
        "tokens_by_role": by_role,
        "tokens_total_input": total_in,
        "tokens_total_output": total_out,
    }


# ---------------------------------------------------------------------------
# Per-condition driver
# ---------------------------------------------------------------------------

CONDITION_CONFIGS: dict[str, dict] = {
    "a": {
        "label": "a_baseline_test0",
        "writer_mode": "v0",
        "example_mode": "v0_loop",
        "memory_mode": "v0",
        "description": "writer=v0 (simple desc), example=v0_loop (single "
                       "feedback loop), memory=v0 (flat last-cycle) — "
                       "reproduces test0 brain state",
    },
    "b": {
        "label": "b_memory_only",
        "writer_mode": "v0",
        "example_mode": "v0_loop",
        "memory_mode": "v1",
        "description": "writer=v0, example=v0_loop, memory=v1 (last + "
                       "nearest cycles) — isolates the cross-run memory upgrade",
    },
    "c": {
        "label": "c_writer_only",
        "writer_mode": "v1",
        "example_mode": "v1_bracket",
        "memory_mode": "v0",
        "description": "writer=v1 (two-mode desc), example=v1_bracket "
                       "(bracket-and-fix), memory=v0 — isolates the writer "
                       "description + example upgrade (bundled in test0→test1)",
    },
    "d": {
        "label": "d_neutral_example",
        "writer_mode": "v0",
        "example_mode": "neutral",
        "memory_mode": "v0",
        "description": "writer=v0, example=neutral (DSL syntax only, no "
                       "strategy hint), memory=v0 — strips the v0_loop "
                       "example's anchoring to see what brain invents alone",
    },
}


def run_condition(cond_key: str) -> None:
    cfg = CONDITION_CONFIGS[cond_key]
    label = cfg["label"]
    writer_mode = cfg["writer_mode"]
    memory_mode = cfg["memory_mode"]
    example_mode = cfg["example_mode"]

    subdir = (
        os.path.join("story_241", _EXP_NAME, f"brain_ablation_{cond_key}")
        if _EXP_NAME
        else os.path.join("story_241", f"brain_ablation_{cond_key}")
    )
    output_dir = os.path.join(WORKSHOP, subdir)
    os.makedirs(output_dir, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    print("=" * 70)
    print(f"=== brain_ablation condition {cond_key.upper()}: {label} ===")
    print(f"=== started {started_at} ===")
    print("=" * 70)
    print(f"  description  : {cfg['description']}")
    print(f"  writer_mode  : {writer_mode}")
    print(f"  example_mode : {example_mode}")
    print(f"  memory_mode  : {memory_mode}")
    print(f"  themes      : {len(THEMES)}")
    print(f"  runs        : {RUNS_PER_THEME} per theme")
    print(f"  cycles      : {NUM_CYCLES} per run")
    print(f"  per-cycle   : up to {WRITER_CALL_CAP_PER_CYCLE} writer calls")
    print(f"  output      : {output_dir}")

    all_results: list[dict] = []
    theme_memory: dict[str, list[dict]] = {}

    for theme_id, theme_desc in THEMES:
        theme_memory[theme_id] = []
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(
                theme_id, theme_desc, run_idx, output_dir,
                writer_mode=writer_mode,
                memory_mode=memory_mode,
                example_mode=example_mode,
                memory_context=list(theme_memory[theme_id]),
            )
            all_results.append(r)
            theme_memory[theme_id].append(build_memory_entry(r, memory_mode))

            status = "Pass" if r["hit"] else f"Fail (len={r['final_length']})"
            err = f"  [error: {r['error']}]" if r.get("error") else ""
            by_role = r["tokens_by_role"]
            brain_in = by_role.get("brain", {}).get("input_tokens", 0)
            brain_out = by_role.get("brain", {}).get("output_tokens", 0)
            writer_in = by_role.get("writer", {}).get("input_tokens", 0)
            writer_out = by_role.get("writer", {}).get("output_tokens", 0)
            max_writer = NUM_CYCLES * WRITER_CALL_CAP_PER_CYCLE
            print(f"  → run result: {status}  | "
                  f"cycles: {r['cycles_used']}/{NUM_CYCLES}  | "
                  f"writer calls: {r['writer_calls_used']}/{max_writer}{err}")
            print(f"     tokens — brain: in={brain_in}, out={brain_out}  |  "
                  f"writer: in={writer_in}, out={writer_out}")

    total_runs = len(all_results)
    hits = sum(1 for r in all_results if r["hit"])
    sum_in = sum(r["tokens_total_input"] for r in all_results)
    sum_out = sum(r["tokens_total_output"] for r in all_results)
    grand_by_role: dict = {}
    for r in all_results:
        for role_name, m in r["tokens_by_role"].items():
            agg = grand_by_role.setdefault(
                role_name,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0},
            )
            agg["calls"] += m["calls"]
            agg["input_tokens"] += m["input_tokens"]
            agg["output_tokens"] += m["output_tokens"]

    summary = {
        "method": f"brain_ablation_{cond_key}",
        "ablation_label": label,
        "ablation_writer_mode": writer_mode,
        "ablation_example_mode": example_mode,
        "ablation_memory_mode": memory_mode,
        "ablation_description": cfg["description"],
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "target_len": TARGET_LEN,
            "runs_per_theme": RUNS_PER_THEME,
            "num_cycles": NUM_CYCLES,
            "writer_call_cap_per_cycle": WRITER_CALL_CAP_PER_CYCLE,
            "max_writer_calls_per_run": NUM_CYCLES * WRITER_CALL_CAP_PER_CYCLE,
            "themes": [{"id": t[0], "desc": t[1]} for t in THEMES],
            "writer_role_config": ROLE_CONFIGS["writer"],
            "brain_role_config": ROLE_CONFIGS["brain"],
        },
        "totals": {
            "runs": total_runs,
            "hits": hits,
            "hit_rate": hits / total_runs if total_runs else 0.0,
            "tokens_input": sum_in,
            "tokens_output": sum_out,
            "tokens_total": sum_in + sum_out,
            "tokens_by_role": grand_by_role,
        },
        "results": all_results,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Per-theme three-tier accuracy (same as method_brain)
    from runners.story_task._metrics import per_theme_counts, overall_counts
    by_theme = per_theme_counts(summary, "method_brain")
    overall = overall_counts(summary, "method_brain")

    def _fmt(num: int, den: int) -> str:
        return f"{num}/{den} ({num/den:.0%})" if den else "(no data)"

    print()
    print("=" * 60)
    print(f"brain_ablation {cond_key.upper()} summary  ({label})")
    print("=" * 60)
    print(f"  overall pass rate (per run)   : "
          f"{_fmt(overall['runs_hits'], overall['runs_total'])}")
    print(f"  overall pass rate (per cycle) : "
          f"{_fmt(overall['cycle_hits'], overall['cycle_total'])}")
    print(f"  overall pass rate (validated) : "
          f"{_fmt(overall['validated_hits'], overall['cycle_total'])}")
    print(f"  per-theme:")
    for tid, m in by_theme.items():
        print(f"    {tid:<26} "
              f"run {_fmt(m['runs_hits'], m['runs_total']):>13}  "
              f"cyc {_fmt(m['cycle_hits'], m['cycle_total']):>13}  "
              f"val {_fmt(m['validated_hits'], m['cycle_total']):>13}")
    print(f"  tokens total     : in={sum_in}, out={sum_out}, sum={sum_in + sum_out}")
    print(f"\nSaved: {summary_path}")

    # Auto-render comparison HTML (fail-soft — other conditions may be missing
    # if running a single --condition; html.py auto-discovers what's present).
    # html.main() re-reads STORY_EXP_NAME, so set the env var if --exp was used.
    try:
        if _EXP_NAME:
            os.environ["STORY_EXP_NAME"] = _EXP_NAME
        from runners.story_task import html as story_html
        story_html.main()
    except Exception as e:
        print(f"[warn] HTML render failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="brain_ablation: test isolated effect of test0→test1 changes "
                    "on method_brain.",
    )
    parser.add_argument(
        "--condition", choices=["a", "b", "c", "d"],
        help="Run a single condition. a=baseline (test0 repro), "
             "b=memory-only change, c=writer-only change, "
             "d=neutral example (no strategy hint).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all four conditions sequentially (~4x cost).",
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Runs per theme. Overrides STORY_RUNS_PER_THEME env var.",
    )
    parser.add_argument(
        "--themes", default="",
        help="Comma-separated theme ids. Overrides STORY_THEMES env var. "
             f"Available: {','.join(t[0] for t in _AVAILABLE_THEMES)}",
    )
    parser.add_argument(
        "--exp", default="",
        help="Experiment name. Overrides STORY_EXP_NAME; results go under "
             "Execution/story_241/<exp>/brain_ablation_<cond>/.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # CLI overrides take precedence over env vars. We mutate the module-level
    # globals so run_condition() / run_one() see the new values.
    global RUNS_PER_THEME, THEMES, _EXP_NAME
    if args.runs is not None:
        RUNS_PER_THEME = args.runs
    if args.themes:
        requested = [t.strip() for t in args.themes.split(",") if t.strip()]
        unknown = [t for t in requested
                   if t not in {x[0] for x in _AVAILABLE_THEMES}]
        if unknown:
            parser.error(f"unknown theme(s): {unknown}. "
                         f"Available: {[x[0] for x in _AVAILABLE_THEMES]}")
        THEMES = [t for t in _AVAILABLE_THEMES if t[0] in requested]
    if args.exp:
        _EXP_NAME = args.exp

    if args.all:
        for cond in ["a", "b", "c", "d"]:
            run_condition(cond)
    elif args.condition:
        run_condition(args.condition)
    else:
        parser.error("provide --condition {a,b,c,d} or --all")


if __name__ == "__main__":
    main()
