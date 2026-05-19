"""method_brain — per cycle: brain designs a workflow, host executes it.

Per (theme, run): the host runs NUM_CYCLES = 8 outer cycles. Each cycle:

    1. brain (gpt-5-mini) designs a workflow JSON program in the DSL
       (sequence / loop / if / call / return + variables). Brain sees:
         - the theme
         - cross-run memory (previous runs of this theme; last-cycle
           snapshot only)
         - within-run history (all previous cycles in THIS run: each
           cycle's workflow JSON, final story, length, hit, strategy_notes)
    2. host executes that workflow via the DSL interpreter, invoking
       writer / length_checker as the workflow directs.
    3. cycle outcome (final story + length + hit) is recorded; appended
       to the within-run history so the NEXT cycle's brain sees it.

Comparison fairness across methods is on the "agent main loop iteration
count" axis (= 8), matching baseline (8 writer-verify retries) and
method_fixed (8 cycles of writer-verify × 3 + textplanner).

Per-cycle writer cap: brain's workflow may call writer AT MOST 3 times
per cycle (host hard cap, matching method_fixed's writer-per-cycle).
Writer calls beyond the cap within a single cycle are silently skipped
(BUDGET_EXHAUSTED recorded in trajectory). Cap resets each cycle.

HIT in any cycle's verifier = run ends successfully (no further cycles).

Usage:
  python -m runners.story_task.method_brain
"""

from __future__ import annotations

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
# Reuse discovery (not the role itself — these are utility functions that
# inspect role/tool modules for SUPPORTED_TASKS / BRAIN_DESCRIPTION fields).
from role_pool.brain import get_visible_roles, get_visible_tools
from tool_pool.text_utils import length_checker


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------

# All 4 canonical themes — filtered at runtime by STORY_THEMES env var.
_AVAILABLE_THEMES: list[tuple[str, str]] = [
    ("mountain_school",       "The lone teacher at a remote mountain school"),
    ("time_displaced_store",  "A convenience store displaced in time"),
    ("photo_studio_last_day", "The final day of an old photo studio"),
    ("rainy_night_bus",       "The last bus on a rainy night"),
]

# Runtime filtering — set STORY_THEMES / STORY_RUNS_PER_THEME (typically via
# run_all.py --themes / --runs) to scope the experiment.
_theme_filter = os.environ.get("STORY_THEMES", "").strip()
if _theme_filter:
    _wanted = {t.strip() for t in _theme_filter.split(",") if t.strip()}
    THEMES = [t for t in _AVAILABLE_THEMES if t[0] in _wanted]
else:
    THEMES = list(_AVAILABLE_THEMES)

RUNS_PER_THEME = int(os.environ.get("STORY_RUNS_PER_THEME", "4"))
TARGET_LEN = 241
TASK_TYPE = "story"

# Outer agent loop: 8 cycles, matching baseline's 8 retries and method_fixed's
# 8 cycles. Per-cycle writer cap is the host hard limit on writer calls within
# one workflow execution; cap resets each cycle. With NUM_CYCLES=8 and
# WRITER_CALL_CAP_PER_CYCLE=3, worst case = 24 writer calls + 8 brain calls
# per run (no early HIT). HIT in any cycle exits the run early.
NUM_CYCLES = 8
WRITER_CALL_CAP_PER_CYCLE = 3

# If STORY_EXP_NAME is set (typically by run_all.py --exp), put results
# under story_241/<exp_name>/<method>/.
_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
OUTPUT_SUBDIR = (
    os.path.join("story_241", _EXP_NAME, "method_brain") if _EXP_NAME
    else os.path.join("story_241", "method_brain")
)


# ---------------------------------------------------------------------------
# BRAIN PROMPT
# ---------------------------------------------------------------------------

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
```
or
```
{{"type": "call", "tool": "length_checker", "args": {{...}}, "save_as": "varname"}}
```
`save_as` is optional. If given, the call's return value is stored in the
variable `varname` for later reference.

### `sequence` — run children in order
```
{{"type": "sequence", "steps": [<node1>, <node2>, ...]}}
```

### `loop` — repeat body until condition (or max_iter)
```
{{"type": "loop", "max_iter": 8, "until": "<condition>", "body": <node>}}
```
`until` is optional; without it the loop runs max_iter times.

### `if` — conditional branch
```
{{"type": "if", "condition": "<condition>", "then": <node>, "else": <node>}}
```
`else` is optional.

### `return` — terminate the workflow immediately
```
{{"type": "return", "value": "$some_var"}}
```

## Variables

- Initial variables provided to your workflow:
    $theme — the story theme (string)
    $target_len — 241 (int)
- You can SET variables by adding `save_as` to a `call` node.
- You can READ variables in args / conditions with `$varname` or `$varname.field`
  (dot access for dicts).
- If a variable doesn't exist yet (e.g., `$last` on the first loop iteration),
  it silently resolves to an empty string — no error. This lets you write
  feedback-driven loops that work cleanly on the first iter.

## Conditions

Conditions are strings of the form `<expr> OP <expr>`:
  OP: == != < > <= >=
  expr: `$var`, `$var.field`, or a literal (int / bool / string)

Examples:
  "$last.hit == true"
  "$last.length == 241"
  "$retry_count >= 5"

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
  immediately. Use length_checker after every writer call so the host can
  detect HIT.
- Within a cycle, design a small focused workflow (1-3 writer calls).
  The OUTER loop is what re-tries with new strategy — don't try to bake
  the whole task into one cycle's workflow.

# Example workflow — "bracket-and-fix" (empirically effective pattern)

This is a 3-writer pattern that has been shown to converge well within one
cycle. The shape:

  1. FRESH DRAFT aiming UNDER target (~231 chars)
  2. FRESH DRAFT aiming OVER target (~251 chars)
  3. MINIMAL EDIT on one of them (host auto-direction does the precise
     delete/append based on its actual length)

Why it works:
  - Two opposing fresh drafts bracket the target — the closer one is at
    most ~10 chars away. Host's auto-direction shines for small N.
  - Each fresh draft is INDEPENDENT (no shared previous_attempt) so writer
    doesn't get stuck in a fixed point (a recurring failure mode of
    single-draft retry loops).
  - The final-stage `guidance` REINFORCES host's tail-edit semantics
    ("tighten the ending") rather than fighting it.

```json
{{
  "strategy_notes": "Bracket-and-fix: produce two fresh drafts above and below the target as a length envelope, then minimal-edit one of them down/up to 241. Final-stage guidance reinforces host's tail-trim semantics.",
  "workflow": {{
    "type": "sequence",
    "steps": [
      {{
        "type": "call",
        "role": "writer",
        "args": {{
          "theme": "$theme",
          "guidance": "Fresh concise draft. Aim ~231 chars (just under target).",
          "previous_attempt": ""
        }},
        "save_as": "draft_short"
      }},
      {{
        "type": "call",
        "tool": "length_checker",
        "args": {{"text": "$draft_short", "target": "$target_len"}},
        "save_as": "check_short"
      }},
      {{
        "type": "if",
        "condition": "$check_short.hit == true",
        "then": {{"type": "return", "value": "$draft_short"}}
      }},
      {{
        "type": "call",
        "role": "writer",
        "args": {{
          "theme": "$theme",
          "guidance": "Fresh vivid draft. Aim ~251 chars (just over target).",
          "previous_attempt": ""
        }},
        "save_as": "draft_long"
      }},
      {{
        "type": "call",
        "tool": "length_checker",
        "args": {{"text": "$draft_long", "target": "$target_len"}},
        "save_as": "check_long"
      }},
      {{
        "type": "if",
        "condition": "$check_long.hit == true",
        "then": {{"type": "return", "value": "$draft_long"}}
      }},
      {{
        "type": "if",
        "condition": "$check_short.absdiff <= $check_long.absdiff",
        "then": {{
          "type": "call",
          "role": "writer",
          "args": {{
            "theme": "$theme",
            "guidance": "Tighten the ending only. Preserve plot and tone.",
            "previous_attempt": "$draft_short"
          }},
          "save_as": "draft_final"
        }},
        "else": {{
          "type": "call",
          "role": "writer",
          "args": {{
            "theme": "$theme",
            "guidance": "Tighten the ending only. Preserve plot and tone.",
            "previous_attempt": "$draft_long"
          }},
          "save_as": "draft_final"
        }}
      }},
      {{
        "type": "call",
        "tool": "length_checker",
        "args": {{"text": "$draft_final", "target": "$target_len"}},
        "save_as": "last"
      }}
    ]
  }}
}}
```

NOTE on the candidate-comparison: we use `$check_short.absdiff <=
$check_long.absdiff` (absolute distance) NOT `$check_short.diff <=
$check_long.diff`. The signed diff comparison flips intuition — short
draft.diff is negative, long draft.diff is positive, so signed comparison
would always pick the short branch regardless of which is actually closer.
Use `.absdiff` whenever you're comparing closeness across candidates.

Writer's recognized args: `theme`, `guidance`, `previous_attempt`. See the
writer role description above for FRESH DRAFT vs MINIMAL EDIT mode rules.

You don't have to use this shape. Other valid patterns:
  - Single-draft + iterative minimal-edit (simple, slower convergence)
  - Multi-phase: try one strategy, branch via `if` to a different
    strategy if the first misses
  - Style-constrained drafts (e.g. "three short sentences, no names")
    to reduce variance before minimal-edit

You have multiple cycles in the outer loop, so don't try to bake the
whole task into a single cycle's workflow. Pick a focused 1-3 writer
call shape per cycle.

# Output format

JSON only. No markdown fences, no surrounding prose, no explanation. Your
entire response must be a single JSON object with `strategy_notes` and
`workflow` keys.
"""


def build_brain_system_prompt() -> str:
    """Render BRAIN_PROMPT_TEMPLATE with the discovered role/tool menus."""
    roles = get_visible_roles(TASK_TYPE)
    tools = get_visible_tools(TASK_TYPE)
    role_block = "\n".join(roles.values()) or "(none)"
    tool_block = "\n".join(tools.values()) or "(none)"
    return BRAIN_PROMPT_TEMPLATE.format(
        role_block=role_block,
        tool_block=tool_block,
        writer_cap=WRITER_CALL_CAP_PER_CYCLE,
        total_cycles=NUM_CYCLES,
        total_writer_max=NUM_CYCLES * WRITER_CALL_CAP_PER_CYCLE,
    )


def _fmt_trace_entry(t: dict) -> str:
    """One-line compact rendering of a trace step for brain to consume.

    MUST match method_brain_code._fmt_trace_entry exactly so the two
    methods present identical feedback density to brain. No story text;
    only writer guidance + lengths and length_checker diff/hit.
    """
    kind = t.get("kind", "?")
    if kind == "writer":
        prev_len = t.get("previous_attempt_len")
        mode = "edit" if prev_len else "fresh"
        gd = (t.get("guidance") or "").replace("\n", " ").strip()
        if len(gd) > 90:
            gd = gd[:87] + "..."
        out_len = t.get("output_len", 0)
        prev_part = f", prev_len={prev_len}" if prev_len else ""
        return f"writer({mode}{prev_part}, guidance={gd!r}) → len={out_len}"
    if kind == "writer_skipped":
        return "writer(SKIPPED — per-cycle cap reached) → \"\""
    if kind == "length_checker":
        length = t.get("length", 0)
        diff = t.get("diff")
        hit = t.get("hit")
        if diff is None:
            return f"length_checker → length={length}"
        diff_str = f"{diff:+d}" if diff is not None else "?"
        return (f"length_checker → length={length}, diff={diff_str}, "
                f"hit={'Pass' if hit else 'Fail'}")
    return f"{kind}: {t}"


def _render_cross_run_memory_block(memory_context: list[dict]) -> str:
    """Cross-run memory: previous RUNS of the same theme.

    Each record contains TWO cycle snapshots:
      - last_cycle:    the cycle where the run ended (Pass cycle for
                       successful runs; safe fallback for failed runs).
      - nearest_cycle: the cycle with the smallest |final_length - 241|
                       across that run. For Pass runs this equals last_cycle.
                       For Fail runs this exposes the most promising attempt
                       that didn't quite make it (preserves info that would
                       be lost if we stored only last).

    Record shape:
      {run_idx, cycles_used, hit, final_length,
       last_cycle: {cycle, final_length, hit, strategy_notes, workflow_json},
       nearest_cycle: {same fields}}
    """
    if not memory_context:
        return ""

    def _render_snapshot(label: str, snap: dict | None) -> list[str]:
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
        trace = snap.get("trace") or []
        if trace:
            out.append("trace:")
            for t in trace:
                out.append(f"  - {_fmt_trace_entry(t)}")
        return out

    lines = [
        "## Cross-run memory (previous runs of this SAME theme)",
        "",
        "Snapshots are filtered to STRATEGY-VALIDATED cycles only — i.e.",
        "cycles where the planned multi-step workflow actually ran (not",
        "cycles where writer #1 luck-hit 241 before any later step had",
        "a chance to execute). This filters out false positives so you",
        "learn from real strategy contributions, not coincidences.",
        "",
        "Each previous run gives up to two snapshots:",
        "  - LAST cycle: the last validated cycle in that run",
        "  - NEAREST cycle: validated cycle with the smallest",
        "    |final_length - 241| (= LAST if same; differs when an earlier",
        "    cycle was closer than where the run finally ended)",
        "",
        "If a run had NO validated cycles, snapshots are absent — that",
        "run's apparent Pass (if any) was first-writer cold-start luck",
        "and carries NO strategy signal. Treat that run as 'no learning'.",
        "",
        "EXPLORATION NOTE: a strategy that came close ONCE may fail again.",
        "Don't just copy nearest-cycle workflows verbatim — consider",
        "variations, hybrid approaches, or genuinely different shapes when",
        "two or more past runs converge on the same near-miss pattern.",
        "",
    ]
    for m in memory_context:
        run_idx = m.get("run_idx", "?")
        cycles_used = m.get("cycles_used", "?")
        outcome = "Pass" if m.get("hit") else "Fail"
        had_validated = m.get("had_validated_strategy", True)
        lines.append(f"### Run {run_idx} ({cycles_used} cycles, {outcome})")

        if not had_validated:
            lines.append(
                f"(No validated cycle in this run — any Pass was "
                f"cold-start luck on writer #1; no strategy signal.)"
            )
            lines.append("")
            continue

        last_c = m.get("last_cycle")
        near_c = m.get("nearest_cycle")
        lines.extend(_render_snapshot("LAST cycle (validated)", last_c))

        if near_c and last_c and near_c.get("cycle") != last_c.get("cycle"):
            lines.extend(_render_snapshot("NEAREST cycle (validated)", near_c))
        elif near_c and last_c and near_c.get("cycle") == last_c.get("cycle"):
            lines.append("(NEAREST cycle is the same as LAST cycle.)")
        lines.append("")
    return "\n".join(lines)


def _render_within_run_history_block(history: list[dict]) -> str:
    """Within-run history: each cycle of THIS run that has already executed.

    Record shape: {cycle, workflow_json, trace, final_length, hit,
                   strategy_notes}

    Brain sees ALL past cycles in this run: each cycle's workflow JSON
    (the plan) + a compact execution trace (writer guidance + lengths,
    length_checker diff/hit). NO story text — that was a leak of the
    actual prose, but for strategy refinement brain only needs lengths
    + structural feedback. Kept in lockstep with method_brain_code so
    the two methods present identical feedback density to brain.
    """
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
        trace = c.get("trace") or []
        if trace:
            lines.append("trace:")
            for t in trace:
                lines.append(f"  - {_fmt_trace_entry(t)}")
        lines.append("")
    return "\n".join(lines)


def build_cycle_input(theme_desc: str, cycle_idx: int, total_cycles: int,
                      target_len: int = TARGET_LEN,
                      cross_run_memory: list[dict] | None = None,
                      within_run_history: list[dict] | None = None) -> str:
    """Build the user-message text for brain at the start of one cycle."""
    parts = [
        f"## Task",
        f"Write a story about: {theme_desc}",
        f"Requirement: exactly {target_len} characters.",
        "",
        f"## Cycle position",
        f"You are at the START of cycle {cycle_idx} of {total_cycles}.",
        f"Cycles remaining (including this one): {total_cycles - cycle_idx + 1}.",
        "",
    ]
    cross_block = _render_cross_run_memory_block(cross_run_memory or [])
    if cross_block:
        parts.append(cross_block)
    within_block = _render_within_run_history_block(within_run_history or [])
    if within_block:
        parts.append(within_block)
    parts.append("## Your turn")
    if (cross_run_memory or within_run_history):
        parts.append(
            "Design the cycle-" + str(cycle_idx) + " workflow JSON. If past "
            "attempts kept missing, change strategy meaningfully — don't repeat "
            "the same shape. Output JSON only."
        )
    else:
        parts.append("Design your cycle-1 workflow JSON. Output JSON only.")
    return "\n".join(parts)


def parse_workflow(brain_output_text: str) -> dict | None:
    """Extract JSON from brain output. Returns None on parse failure.
    Tolerates leading/trailing prose and ```json fences."""
    text = (brain_output_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# DSL Interpreter
# ---------------------------------------------------------------------------

class ExecutionContext:
    """State carried through workflow execution.

    - vars: variable store ({"theme": "...", "target_len": 241, ...} initially)
    - metrics: MetricsTracker for token accounting
    - trajectory: flat per-step record (input/output of every call)
    - writer_calls: counter for budget enforcement
    - step: global step counter for trajectory indexing
    - returned / return_value: set by `return` nodes to short-circuit ancestors
    - last_text / last_length: convenience accessors (the latest writer output
        and its measured length). last_length is updated whenever a
        length_checker call saves into a variable referenced by ... well, by
        anyone — for simplicity we track the most recent length_checker result.
    """

    def __init__(self, theme: str, target_len: int, metrics: MetricsTracker,
                 trajectory: list, starting_step: int):
        self.vars: dict = {"theme": theme, "target_len": target_len}
        self.metrics = metrics
        self.trajectory = trajectory
        self.writer_calls = 0
        self.writer_cap = WRITER_CALL_CAP_PER_CYCLE
        self.step = starting_step
        self.returned = False
        self.return_value = None
        self.last_text: str = ""
        self.last_length: int = 0
        # Per-cycle compact trace surfaced to next-cycle brain. Mirrors
        # method_brain_code's trace shape so the two methods present
        # equally-dense execution feedback to brain (writer guidance +
        # lengths, length_checker diff/hit — NO story text). Reset by
        # the host between cycles via a fresh ExecutionContext.
        self.trace: list[dict] = []


# ---- Value / arg resolution ----------------------------------------------

_SENTINEL = object()


def _lookup_var(ref: str, ctx: ExecutionContext):
    """Resolve a single `$var` or `$var.field.subfield` reference.
    Returns _SENTINEL if any part of the path doesn't resolve."""
    path = ref.split(".")
    val = ctx.vars.get(path[0], _SENTINEL)
    if val is _SENTINEL:
        return _SENTINEL
    for p in path[1:]:
        if isinstance(val, dict):
            val = val.get(p, _SENTINEL)
            if val is _SENTINEL:
                return _SENTINEL
        else:
            return _SENTINEL
    return val


def resolve_value(v, ctx: ExecutionContext):
    """Resolve a JSON value. Strings starting with $ are variable references.
    Missing variables silently resolve to empty string ""."""
    if not isinstance(v, str):
        return v
    if not v.startswith("$"):
        return v
    ref = v[1:]
    looked = _lookup_var(ref, ctx)
    if looked is _SENTINEL:
        return ""
    return looked


def resolve_args(args, ctx: ExecutionContext):
    """Recursively resolve $vars inside an args dict (or nested structures)."""
    if isinstance(args, dict):
        return {k: resolve_args(v, ctx) for k, v in args.items()}
    if isinstance(args, list):
        return [resolve_args(x, ctx) for x in args]
    return resolve_value(args, ctx)


# ---- Condition evaluation ------------------------------------------------

_OPS = ("==", "!=", "<=", ">=", "<", ">")


def _coerce(v):
    """Try to coerce string to int / float / bool for comparison."""
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return s
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null" or low == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return v


def eval_condition(expr: str, ctx: ExecutionContext) -> bool:
    """Evaluate a simple binary expression. Supports == != < > <= >=.
    Left/right operands may be $vars or literals."""
    expr = (expr or "").strip()
    if not expr:
        return False
    # Order matters: 2-char ops before 1-char to avoid partial match.
    for op in _OPS:
        idx = expr.find(op)
        if idx >= 0:
            left = expr[:idx].strip()
            right = expr[idx + len(op):].strip()
            l_val = _coerce(resolve_value(left, ctx))
            r_val = _coerce(resolve_value(right, ctx))
            try:
                if op == "==":
                    return l_val == r_val
                if op == "!=":
                    return l_val != r_val
                if op == "<=":
                    return l_val <= r_val
                if op == ">=":
                    return l_val >= r_val
                if op == "<":
                    return l_val < r_val
                if op == ">":
                    return l_val > r_val
            except TypeError:
                return False
    return False


# ---- Call dispatch -------------------------------------------------------

def _call_writer(args: dict, ctx: ExecutionContext) -> str:
    """One writer LLM call. Fresh LLMNode. Records trajectory + tokens."""
    if ctx.writer_calls >= ctx.writer_cap:
        ctx.step += 1
        ctx.trajectory.append({
            "step": ctx.step,
            "role": "writer",
            "purpose": "BUDGET_EXHAUSTED (writer call skipped)",
            "input": args,
            "output": None,
            "tokens": {"in": 0, "out": 0},
        })
        ctx.trace.append({"step": ctx.step, "kind": "writer_skipped"})
        return ""

    cfg = ROLE_CONFIGS["writer"]
    node = LLMNode(
        system_prompt=writer_role.PROMPT,
        role="writer",
        max_steps=cfg["max_steps"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        metrics_tracker=ctx.metrics,
    )
    user_input = writer_role.build_input(
        theme=str(args.get("theme") or ""),
        guidance=str(args.get("guidance") or ""),
        feedback=str(args.get("feedback") or ""),
        previous_attempt=str(args.get("previous_attempt") or ""),
    )

    n0 = len(ctx.metrics.calls)
    node.reset_message()
    response = node.run(user_input)
    story = (response.get("text") or "").strip()
    new_calls = ctx.metrics.calls[n0:]
    tokens = {
        "in": sum(c.input_tokens for c in new_calls),
        "out": sum(c.output_tokens for c in new_calls),
    }

    ctx.writer_calls += 1
    ctx.last_text = story
    ctx.step += 1
    ctx.trajectory.append({
        "step": ctx.step,
        "role": "writer",
        "purpose": f"writer call #{ctx.writer_calls}",
        "input": user_input,
        "output": story,
        "tokens": tokens,
    })
    prev_attempt = str(args.get("previous_attempt") or "")
    ctx.trace.append({
        "step": ctx.step,
        "kind": "writer",
        "guidance": str(args.get("guidance") or ""),
        "previous_attempt_len": len(prev_attempt) if prev_attempt else 0,
        "output_len": len(story),
    })
    # Note: verifier line follows in _call_length_checker, so we don't log here.
    # Brain's workflow always pairs each writer with a length_checker call.
    return story


def _call_length_checker(args: dict, ctx: ExecutionContext) -> dict:
    """Length-checker tool call. Pure Python, no tokens, records trajectory."""
    text = args.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    target_arg = args.get("target")
    try:
        target = int(target_arg) if target_arg is not None and target_arg != "" else None
    except (ValueError, TypeError):
        target = None
    result = length_checker(text, target=target)
    ctx.last_length = result.get("length", 0)
    ctx.step += 1
    ctx.trajectory.append({
        "step": ctx.step,
        "role": "length_checker",
        "purpose": "verify length",
        "input": {"text": text, "target": target},
        "output": result,
        "tokens": {"in": 0, "out": 0},
    })
    trace_entry = {
        "step": ctx.step,
        "kind": "length_checker",
        "length": result.get("length", 0),
    }
    if "diff" in result:
        trace_entry["diff"] = result["diff"]
        trace_entry["hit"] = result.get("hit", False)
    ctx.trace.append(trace_entry)
    # Per-call live log (paired with the most recent writer call).
    # Only pass/fail + length + diff (token info goes to summary.json).
    if result.get("target") is not None:
        diff_str = (f"diff={result['diff']:+d}" if not result.get("hit")
                    else "diff=  0")
        status = "Pass" if result.get("hit") else "Fail"
        print(f"  writer #{ctx.writer_calls}→verify: {status}  "
              f"len={result['length']:>3}  {diff_str}", flush=True)
    else:
        print(f"  writer #{ctx.writer_calls}→verify: len={result['length']:>3}",
              flush=True)
    return result


# ---- Main executor -------------------------------------------------------

class WorkflowError(Exception):
    pass


_MAX_NODES = 200  # global hard cap on nodes executed, prevents infinite loops


def execute(node, ctx: ExecutionContext, depth: int = 0) -> None:
    if ctx.returned:
        return
    if not isinstance(node, dict):
        raise WorkflowError(f"node is not a dict: {type(node).__name__}")
    if depth > 20:
        raise WorkflowError("max nesting depth (20) exceeded")

    t = node.get("type")

    if t == "sequence":
        for step in node.get("steps", []) or []:
            execute(step, ctx, depth + 1)
            if ctx.returned:
                return

    elif t == "loop":
        max_iter = int(node.get("max_iter", 10))
        until_expr = node.get("until")
        body = node.get("body")
        if body is None:
            return
        # Hard safety cap: never let max_iter exceed per-cycle cap * 4.
        max_iter = min(max_iter, WRITER_CALL_CAP_PER_CYCLE * 4)
        for _ in range(max_iter):
            execute(body, ctx, depth + 1)
            if ctx.returned:
                return
            if until_expr and eval_condition(until_expr, ctx):
                break
            # Stop spinning if budget exhausted (writer skipped, condition
            # may never become true).
            if ctx.writer_calls >= ctx.writer_cap:
                break

    elif t == "if":
        cond = node.get("condition", "")
        if eval_condition(cond, ctx):
            then_node = node.get("then")
            if then_node is not None:
                execute(then_node, ctx, depth + 1)
        else:
            else_node = node.get("else")
            if else_node is not None:
                execute(else_node, ctx, depth + 1)

    elif t == "return":
        ctx.returned = True
        val = node.get("value")
        ctx.return_value = resolve_value(val, ctx) if val is not None else None

    elif t == "call":
        resolved_args = resolve_args(node.get("args", {}) or {}, ctx)
        save_as = node.get("save_as")
        target = node.get("role") or node.get("tool")
        if target == "writer":
            result = _call_writer(resolved_args, ctx)
        elif target == "length_checker":
            result = _call_length_checker(resolved_args, ctx)
        else:
            ctx.step += 1
            ctx.trajectory.append({
                "step": ctx.step,
                "role": str(target),
                "purpose": "UNKNOWN_DISPATCH (call skipped)",
                "input": resolved_args,
                "output": None,
                "tokens": {"in": 0, "out": 0},
            })
            return
        if save_as:
            ctx.vars[save_as] = result

    else:
        raise WorkflowError(f"unknown node type: {t!r}")


# ---------------------------------------------------------------------------
# Per-run flow
# ---------------------------------------------------------------------------

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str,
            memory_context: list[dict] | None = None) -> dict:
    """One (theme, run_idx) pass through NUM_CYCLES outer cycles.

    Each cycle: brain (fresh LLMNode) → designs workflow → host executes it
    via the DSL interpreter → outcome recorded. The next cycle's brain sees
    all prior cycles' workflows / outcomes / final stories.

    `memory_context` — cross-run memory: previous runs of this theme. Each
    record is the LAST-cycle snapshot of that run (per Q4 design decision).
    """
    metrics = MetricsTracker()
    memory_context = memory_context or []

    brain_cfg = ROLE_CONFIGS["brain"]
    trajectory: list[dict] = []
    step_counter = [0]

    def next_step() -> int:
        step_counter[0] += 1
        return step_counter[0]

    cycles: list[dict] = []          # full per-cycle records (this run)
    final_text = ""
    final_length = 0
    hit = False
    overall_error = None

    for cycle_idx in range(1, NUM_CYCLES + 1):
        # ---- Brain call: fresh LLMNode so message history doesn't grow ----
        brain_node = LLMNode(
            system_prompt=build_brain_system_prompt(),
            role="brain",
            max_steps=brain_cfg["max_steps"],
            model=brain_cfg["model"],
            temperature=brain_cfg["temperature"],
            max_tokens=brain_cfg["max_tokens"],
            metrics_tracker=metrics,
        )
        # Slim view of past cycles for brain's prompt (drop bulky fields like
        # full trajectory — keep just what the prompt template uses).
        prior_for_brain = [
            {
                "cycle": c["cycle"],
                "workflow_json": c["workflow_json"],
                "trace": c.get("trace", []),
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
            target_len=TARGET_LEN,
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
                "error": "plan_unparseable",
            })
            overall_error = "plan_unparseable"
            # Continue to next cycle — brain might recover
            continue

        workflow = plan.get("workflow")
        strategy_notes = plan.get("strategy_notes", "")

        # ---- Execute this cycle's workflow ----
        # Fresh ExecutionContext per cycle so the per-cycle writer cap resets.
        ctx = ExecutionContext(
            theme=theme_desc, target_len=TARGET_LEN,
            metrics=metrics, trajectory=trajectory,
            starting_step=step_counter[0],
        )
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
        # Sync step counter to ctx's
        step_counter[0] = ctx.step

        # Cycle outcome — empty workflow execution leaves ctx.last_text="".
        # In that case keep the carry-over text from prior cycle so the next
        # cycle still has something to look at.
        if ctx.last_text:
            final_text = ctx.last_text
            final_length = ctx.last_length
        cycle_hit = ctx.last_length == TARGET_LEN

        # strategy_validated: did the cycle's planned multi-step strategy
        # actually contribute to the outcome? Defined as: NOT (Pass AND
        # writer_calls == 1). A Pass on the very first writer call means
        # the strategy was cut short by a lucky cold-start hit before any
        # later step (length_checker → if → minimal-edit) could run.
        strategy_validated = ctx.writer_calls >= 1 and not (
            cycle_hit and ctx.writer_calls == 1
        )
        cycles.append({
            "cycle": cycle_idx,
            "workflow_json": workflow,
            "strategy_notes": strategy_notes,
            "trace": ctx.trace,
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

    # ---- Save story (HIT or last MISS) ----
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.join(WORKSHOP, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    max_writer = NUM_CYCLES * WRITER_CALL_CAP_PER_CYCLE
    print(f"=== method_brain (story task) started {started_at} ===")
    print(f"  themes  : {len(THEMES)}")
    print(f"  runs    : {RUNS_PER_THEME} per theme")
    print(f"  loop    : {NUM_CYCLES} cycles per run "
          f"(each cycle: brain → workflow → execute)")
    print(f"  budget  : {WRITER_CALL_CAP_PER_CYCLE} writer calls per cycle "
          f"(max {max_writer} writer calls per run)")
    print(f"  target  : exactly {TARGET_LEN} characters")
    print(f"  output  : {output_dir}")

    all_results: list[dict] = []
    # Per-theme cross-run memory. Per Q4 decision, each previous run's record
    # is the LAST-cycle snapshot (which transitively contains the learning
    # from earlier cycles in that run, so older snapshots are redundant).
    theme_memory: dict[str, list[dict]] = {}
    for theme_id, theme_desc in THEMES:
        theme_memory[theme_id] = []
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir,
                        memory_context=list(theme_memory[theme_id]))
            all_results.append(r)

            # Build cross-run memory entry: last + nearest cycle, but ONLY
            # among cycles where the strategy actually ran to validation
            # (not first-call cold-start luck). This keeps brain from
            # learning false positives — a Pass that came from writer #1
            # luck doesn't carry information about the strategy.
            #
            #   last_cycle    — last validated cycle in the run
            #   nearest_cycle — validated cycle with smallest |length - target|
            #
            # If the run had NO validated cycles (e.g. all passes were
            # cold-start luck, or workflow errors throughout), both
            # snapshots are None; brain sees "run X: hit=Pass overall,
            # but no validated cycle data — the pass was luck".
            def _cycle_snapshot(c: dict | None) -> dict | None:
                if c is None:
                    return None
                return {
                    "cycle": c.get("cycle"),
                    "final_length": c.get("final_length"),
                    "hit": c.get("hit", False),
                    "strategy_notes": c.get("strategy_notes", ""),
                    "workflow_json": c.get("workflow_json"),
                    "trace": c.get("trace", []),
                    "final_story": c.get("final_story", ""),
                    "strategy_validated": c.get("strategy_validated", True),
                }

            cycles = r["cycles"] or []
            validated_cycles = [c for c in cycles if c.get("strategy_validated")]
            last_cycle = validated_cycles[-1] if validated_cycles else None
            nearest_cycle = (
                min(validated_cycles,
                    key=lambda c: abs(c.get("final_length", 0) - TARGET_LEN))
                if validated_cycles else None
            )
            mem_entry = {
                "run_idx": run_idx,
                "cycles_used": r["cycles_used"],
                "hit": r["hit"],
                "final_length": r["final_length"],
                "had_validated_strategy": bool(validated_cycles),
                "last_cycle": _cycle_snapshot(last_cycle),
                "nearest_cycle": _cycle_snapshot(nearest_cycle),
            }
            theme_memory[theme_id].append(mem_entry)

            status = "Pass" if r["hit"] else f"Fail (len={r['final_length']})"
            err = f"  [error: {r['error']}]" if r.get("error") else ""
            by_role = r["tokens_by_role"]
            brain_in = by_role.get("brain", {}).get("input_tokens", 0)
            brain_out = by_role.get("brain", {}).get("output_tokens", 0)
            writer_in = by_role.get("writer", {}).get("input_tokens", 0)
            writer_out = by_role.get("writer", {}).get("output_tokens", 0)
            print(f"  → run result: {status}  | "
                  f"cycles: {r['cycles_used']}/{NUM_CYCLES}  | "
                  f"writer calls: {r['writer_calls_used']}/{max_writer}{err}")
            print(f"     tokens — brain: in={brain_in}, out={brain_out}  |  "
                  f"writer: in={writer_in}, out={writer_out}")

    # ---- Aggregate ----
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
        "method": "method_brain",
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

    # Per-theme accuracy — two metrics (see _metrics.py)
    from runners.story_task._metrics import per_theme_counts, overall_counts
    by_theme = per_theme_counts(summary, "method_brain")
    overall = overall_counts(summary, "method_brain")

    def _fmt(num: int, den: int) -> str:
        return f"{num}/{den} ({num/den:.0%})" if den else "(no data)"

    print()
    print("=" * 60)
    print("method_brain summary")
    print("=" * 60)
    print(f"  overall pass rate (per run)       : "
          f"{_fmt(overall['runs_hits'], overall['runs_total'])}")
    print(f"  overall pass rate (per cycle)     : "
          f"{_fmt(overall['cycle_hits'], overall['cycle_total'])}")
    print(f"  overall pass rate (validated)     : "
          f"{_fmt(overall['validated_hits'], overall['cycle_total'])}")
    print(f"  per-theme:")
    for tid, m in by_theme.items():
        print(f"    {tid:<26} "
              f"run {_fmt(m['runs_hits'], m['runs_total']):>13}  "
              f"cyc {_fmt(m['cycle_hits'], m['cycle_total']):>13}  "
              f"val {_fmt(m['validated_hits'], m['cycle_total']):>13}")
    print(f"  tokens total     : in={sum_in}, out={sum_out}, sum={sum_in + sum_out}")
    print(f"\nSaved: {summary_path}")

    # Auto-render comparison HTML (fail-soft — other methods may be missing)
    try:
        from runners.story_task import html as story_html
        story_html.main()
    except Exception as e:
        print(f"[warn] HTML render failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
