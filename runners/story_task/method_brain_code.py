"""method_brain_code — per cycle: brain writes a Python function, host runs it.

Same outer-loop shape as method_brain (8 cycles per run, per-cycle writer
cap = 3). The ONLY difference is brain's output medium:

  method_brain      : brain → JSON DSL workflow → host parses → host's DSL
                      interpreter walks the JSON tree dispatching to
                      writer / length_checker.

  method_brain_code : brain → Python source defining `solve(theme, target_len,
                      writer, length_checker) -> str`. Host execs the code,
                      injects wrapped `writer` / `length_checker` callables
                      (the wrappers enforce per-cycle caps + record
                      trajectory), then invokes `solve(...)`.

The optimization signal is the same as method_brain — the brain re-sees its
own past outputs (now: source code) and re-designs. The function's docstring
serves as the natural-language strategy description (equivalent to
method_brain's `strategy_notes` field).

What brain sees about prior cycles (within-run + cross-run memory):
  - the function source code
  - the strategy_notes (function docstring, extracted for at-a-glance scan)
  - a compact step trace (writer guidance + lengths; length_checker results)
  - final length + hit/miss
  NOT: the final story text (per design — user wants strategy + length only).

Per-cycle caps (host hard limits, brain cannot bypass):
  - writer calls: 3 per cycle (matches method_brain). Beyond the cap, the
    writer wrapper returns "" instead of calling the LLM and logs a
    BUDGET_EXHAUSTED step. Resets each cycle.
  - length_checker calls: 30 per cycle. Pure-Python, no token cost, but a
    safety belt against `while True: length_checker(...)`. Raises
    BudgetExceeded which we catch at the cycle boundary.

HIT in any cycle's final length_checker = run ends successfully.

Usage:
  python -m runners.story_task.method_brain_code
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP, ROLE_CONFIGS
from llm_node import LLMNode
from metrics import MetricsTracker
from role_pool import writer as writer_role
from tool_pool.text_utils import length_checker as _length_checker_fn


# ---------------------------------------------------------------------------
# Task spec — mirrors method_brain
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
LENGTH_CHECKER_CALL_CAP_PER_CYCLE = 30   # safety belt vs runaway loops

_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
OUTPUT_SUBDIR = (
    os.path.join("story_241", _EXP_NAME, "method_brain_code") if _EXP_NAME
    else os.path.join("story_241", "method_brain_code")
)


# ---------------------------------------------------------------------------
# BRAIN PROMPT — teaches Python output instead of JSON DSL
# ---------------------------------------------------------------------------

BRAIN_PROMPT_TEMPLATE = """You are a workflow architect operating in an
ITERATIVE outer loop. Your job each turn is to write ONE Python function
that, when executed by the host, attempts to produce a 241-character story.

You do NOT execute the function yourself — the host execs your code and
invokes the function. After execution, the host calls YOU AGAIN with the
cycle's outcome appended to the within-run history, and you write the
NEXT cycle's function. The loop ends early when any length_checker call
in your function returns hit == True (length == 241).

# Required function signature

```python
def solve(theme, target_len, writer, length_checker):
    \"\"\"<strategy description — this is your strategy_notes>\"\"\"
    ...
    return final_story   # str
```

  - `theme`         — the story theme (str)
  - `target_len`    — {target_len} (int)
  - `writer`        — host-provided wrapped callable (see API below)
  - `length_checker`— host-provided wrapped callable (see API below)

The DOCSTRING of `solve` is your strategy description. The host extracts
and surfaces it separately as `strategy_notes` for at-a-glance scanning
across cycles. Be clear about WHY this strategy — past failure mode you're
addressing, hypothesis you're testing, etc.

Return the final story string. Host will length-check whatever you return
and use that as the cycle's outcome. (You typically also length-check it
inside `solve` so you can early-exit on hit, but the host's final check
is the canonical outcome.)

# Available callables (host-provided, do NOT re-import or redefine)

## `writer(theme, guidance="", previous_attempt="") -> str`
  LLM worker (gpt-4o-mini). Returns the story text.

  TWO MODES (chosen by previous_attempt):

  - FRESH DRAFT mode  (previous_attempt = "")
      Drafts from theme + guidance. No host auto-direction.
      Use for a brand-new attempt unanchored to prior drafts.

  - MINIMAL EDIT mode  (previous_attempt = "<prior draft text>")
      Host computes diff = target_len − len(previous_attempt) and INJECTS
      a precise direction into writer's user message:
          "This is N chars — exactly M too long/short for the 241 target.
           Output the SAME story with exactly M characters DELETED/APPENDED
           from the end (or tightened in place). Do NOT rewrite. Do NOT
           change the plot."
      Your `guidance` LAYERS ON TOP of this auto-direction.
      Aligned guidance:    "tighten the final clause", "prefer cutting
                            filler words near the end", "preserve plot".
      Conflicting (bad):   "rewrite the opening", "change protagonist's
                            name", "edit the middle sentence".

  Accuracy is high when |diff| ≤ 15. For |diff| > 30, prefer FRESH DRAFT.

## `length_checker(text, target=None) -> dict`
  Pure Python, no tokens. Returns:
    target given : {{"length": int, "target": int, "diff": int,
                     "absdiff": int, "hit": bool, "delta_text": str}}
    target None  : {{"length": int}}
  Use `.absdiff` when comparing two candidates' closeness — `.diff` is
  signed and flips intuition (a short draft has diff<0, a long has diff>0,
  so signed comparison always picks the short one regardless of which is
  actually closer).

# Execution constraints (FIXED by host)

- **Per-cycle writer cap: {writer_cap} writer calls.** The wrapper silently
  returns "" beyond the cap. Cap resets next cycle — you have
  {total_cycles} cycles, so up to {total_writer_max} writer calls per run.
- **Per-cycle length_checker cap: {lc_cap}** — safety belt only; you should
  never hit it. If a runaway loop calls length_checker {lc_cap}+ times the
  host raises and the cycle ends.
- SUCCESS: any length_checker call returning hit == True ends the run
  immediately. Call length_checker after every writer call so the host
  can detect HIT.
- Keep the function small and focused (1–3 writer calls). The OUTER loop
  retries with new strategy across cycles — don't try to bake the whole
  task into a single cycle's function.
- Don't import anything. Don't redefine `writer` or `length_checker`.
  Don't read/write files. Don't catch all exceptions silently — let them
  surface so you can debug from the trace in the next cycle.

# Example function — "bracket-and-fix" (empirically effective pattern)

```python
def solve(theme, target_len, writer, length_checker):
    \"\"\"Bracket-and-fix: produce two fresh drafts above and below the target
    as a length envelope, then minimal-edit the closer one. Two opposing
    fresh drafts cap distance to ~10 chars; final tighten leans on host's
    tail-edit semantics rather than fighting it.\"\"\"
    short = writer(theme=theme, previous_attempt="",
                   guidance="Fresh concise draft. Aim ~231 chars (just under target).")
    s_check = length_checker(short, target=target_len)
    if s_check["hit"]:
        return short

    long_ = writer(theme=theme, previous_attempt="",
                   guidance="Fresh vivid draft. Aim ~251 chars (just over target).")
    l_check = length_checker(long_, target=target_len)
    if l_check["hit"]:
        return long_

    base = short if s_check["absdiff"] <= l_check["absdiff"] else long_
    final = writer(theme=theme, previous_attempt=base,
                   guidance="Tighten the ending only. Preserve plot and tone.")
    length_checker(final, target=target_len)
    return final
```

You don't have to use this shape. Other valid patterns:
  - Single-draft + iterative minimal-edit
  - Multi-phase: try one strategy, branch to a different shape on miss
  - Style-constrained drafts (e.g. "three short sentences, no names") to
    reduce variance before minimal-edit
  - A loop over `previous_attempt` with bounded retries

# Output format

Python source only. No markdown fences, no surrounding prose, no
explanation. Your entire response must be code defining a `solve`
function with the exact signature above. The host will `exec()` it.
"""


def build_brain_system_prompt() -> str:
    return BRAIN_PROMPT_TEMPLATE.format(
        target_len=TARGET_LEN,
        writer_cap=WRITER_CALL_CAP_PER_CYCLE,
        lc_cap=LENGTH_CHECKER_CALL_CAP_PER_CYCLE,
        total_cycles=NUM_CYCLES,
        total_writer_max=NUM_CYCLES * WRITER_CALL_CAP_PER_CYCLE,
    )


# ---------------------------------------------------------------------------
# Memory / history blocks shown to brain
# ---------------------------------------------------------------------------

def _render_snapshot(label: str, snap: dict | None) -> list[str]:
    if not snap:
        return []
    cyc = snap.get("cycle", "?")
    fl = snap.get("final_length", "?")
    passed = "Pass" if snap.get("hit") else "Fail"
    out = [f"**{label}** (cycle {cyc}, length {fl}, {passed}):"]
    sn = (snap.get("strategy_notes") or "").strip()
    if sn:
        out.append("strategy_notes:")
        out.append(f"  {sn}")
    code = snap.get("code")
    if code:
        out.append("code:")
        out.append("```python")
        out.append(code.rstrip())
        out.append("```")
    trace = snap.get("trace") or []
    if trace:
        out.append("trace:")
        for t in trace:
            out.append(f"  - {_fmt_trace_entry(t)}")
    return out


def _fmt_trace_entry(t: dict) -> str:
    """One-line compact rendering of a trace step for brain to consume.

    Crucially does NOT include the writer output text — only its length.
    The user spec is: brain sees strategy + lengths + trace shape, not
    the story text.
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
    """Cross-run memory: previous runs of this SAME theme.

    Same schema as method_brain but the per-cycle artifact is the
    Python source code (not the JSON workflow). Filtered to strategy-
    validated cycles only (not first-writer cold-start luck).
    """
    if not memory_context:
        return ""
    lines = [
        "## Cross-run memory (previous runs of this SAME theme)",
        "",
        "Snapshots are filtered to STRATEGY-VALIDATED cycles only — i.e.",
        "cycles where the planned multi-step function actually ran (not",
        "cycles where writer #1 luck-hit 241 before any later step had",
        "a chance to execute).",
        "",
        "Each previous run gives up to two snapshots:",
        "  - LAST cycle: the last validated cycle in that run",
        "  - NEAREST cycle: validated cycle with the smallest",
        "    |final_length - 241|",
        "",
        "EXPLORATION NOTE: a strategy that came close ONCE may fail again.",
        "Consider variations / hybrids / genuinely different shapes when",
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
                "(No validated cycle in this run — any Pass was cold-start "
                "luck on writer #1; no strategy signal.)"
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
    """Within-run history: cycles already executed in THIS run.

    For each cycle: strategy_notes (docstring), code source, trace, final
    length, hit/fail. NO final_story text.
    """
    if not history:
        return ""
    lines = [
        "## Within-run history (cycles you have already written in THIS run)",
        "",
    ]
    for c in history:
        cycle_idx = c.get("cycle", "?")
        outcome = "Pass" if c.get("hit") else "Fail"
        final_len = c.get("final_length", "?")
        lines.append(f"### Cycle {cycle_idx} ({outcome}, final length {final_len})")
        sn = (c.get("strategy_notes") or "").strip()
        if sn:
            lines.append("strategy_notes:")
            lines.append(f"  {sn}")
        code = c.get("code")
        if code:
            lines.append("code:")
            lines.append("```python")
            lines.append(code.rstrip())
            lines.append("```")
        trace = c.get("trace") or []
        if trace:
            lines.append("trace:")
            for t in trace:
                lines.append(f"  - {_fmt_trace_entry(t)}")
        err = c.get("error")
        if err:
            lines.append(f"error: {err}")
        lines.append("")
    return "\n".join(lines)


def build_cycle_input(theme_desc: str, cycle_idx: int, total_cycles: int,
                      target_len: int = TARGET_LEN,
                      cross_run_memory: list[dict] | None = None,
                      within_run_history: list[dict] | None = None) -> str:
    parts = [
        "## Task",
        f"Write a story about: {theme_desc}",
        f"Requirement: exactly {target_len} characters.",
        "",
        "## Cycle position",
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
    if cross_run_memory or within_run_history:
        parts.append(
            f"Write the cycle-{cycle_idx} `solve` function. If past attempts "
            "kept missing, change strategy meaningfully — don't repeat the "
            "same shape. Update the docstring to describe the NEW strategy "
            "and what hypothesis it tests. Output Python code only."
        )
    else:
        parts.append(
            "Write your cycle-1 `solve` function. Output Python code only."
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def parse_code(brain_output_text: str) -> str | None:
    """Extract Python source from brain's text output.

    Tolerates leading/trailing prose and ```python / ``` fences. Returns
    None if no plausible code is found.
    """
    text = (brain_output_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    if "def solve" not in text:
        # Last-ditch: maybe brain wrapped code in a fence somewhere inside
        fence_start = text.find("```")
        if fence_start >= 0:
            rest = text[fence_start + 3:]
            nl = rest.find("\n")
            if nl >= 0:
                rest = rest[nl + 1:]
            fence_end = rest.find("```")
            if fence_end >= 0:
                rest = rest[:fence_end]
            if "def solve" in rest:
                return rest.strip()
        return None
    return text


# ---------------------------------------------------------------------------
# Wrapped writer / length_checker — host-controlled budget + trajectory
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when length_checker is called more than its per-cycle cap.
    Caught at the cycle boundary; halts the current `solve` call cleanly."""


class _ExecState:
    """State threaded through the wrappers for one cycle's exec."""

    def __init__(self):
        self.writer_calls = 0
        self.length_checker_calls = 0
        self.last_text = ""
        self.last_length = 0


def _make_wrappers(theme_desc: str, metrics: MetricsTracker,
                   trajectory: list, step_counter: list,
                   trace: list, exec_state: _ExecState):
    """Build the (writer, length_checker) callables injected into solve()."""

    writer_cfg = ROLE_CONFIGS["writer"]

    def writer(theme=None, guidance="", previous_attempt=""):
        # Default theme to the cycle theme if caller didn't pass it.
        if theme is None or theme == "":
            theme = theme_desc

        if exec_state.writer_calls >= WRITER_CALL_CAP_PER_CYCLE:
            step_counter[0] += 1
            trajectory.append({
                "step": step_counter[0],
                "role": "writer",
                "purpose": "BUDGET_EXHAUSTED (writer call skipped)",
                "input": {"theme": theme, "guidance": guidance,
                          "previous_attempt": previous_attempt},
                "output": None,
                "tokens": {"in": 0, "out": 0},
            })
            trace.append({
                "step": step_counter[0],
                "kind": "writer_skipped",
            })
            return ""

        node = LLMNode(
            system_prompt=writer_role.PROMPT,
            role="writer",
            max_steps=writer_cfg["max_steps"],
            model=writer_cfg["model"],
            temperature=writer_cfg["temperature"],
            max_tokens=writer_cfg["max_tokens"],
            metrics_tracker=metrics,
        )
        user_input = writer_role.build_input(
            theme=str(theme or ""),
            guidance=str(guidance or ""),
            feedback="",
            previous_attempt=str(previous_attempt or ""),
        )

        n0 = len(metrics.calls)
        node.reset_message()
        response = node.run(user_input)
        story = (response.get("text") or "").strip()
        new_calls = metrics.calls[n0:]
        tokens = {
            "in": sum(c.input_tokens for c in new_calls),
            "out": sum(c.output_tokens for c in new_calls),
        }

        exec_state.writer_calls += 1
        exec_state.last_text = story
        step_counter[0] += 1
        trajectory.append({
            "step": step_counter[0],
            "role": "writer",
            "purpose": f"writer call #{exec_state.writer_calls}",
            "input": user_input,
            "output": story,
            "tokens": tokens,
        })
        trace.append({
            "step": step_counter[0],
            "kind": "writer",
            "guidance": guidance or "",
            "previous_attempt_len": (len(previous_attempt)
                                     if previous_attempt else 0),
            "output_len": len(story),
        })
        return story

    def length_checker(text, target=None):
        if exec_state.length_checker_calls >= LENGTH_CHECKER_CALL_CAP_PER_CYCLE:
            raise BudgetExceeded(
                f"length_checker called > {LENGTH_CHECKER_CALL_CAP_PER_CYCLE} "
                f"times in one cycle (likely a runaway loop)"
            )
        if not isinstance(text, str):
            text = str(text)
        try:
            target_int = int(target) if target is not None else None
        except (TypeError, ValueError):
            target_int = None
        result = _length_checker_fn(text, target=target_int)
        exec_state.length_checker_calls += 1
        exec_state.last_length = result.get("length", 0)
        step_counter[0] += 1
        trajectory.append({
            "step": step_counter[0],
            "role": "length_checker",
            "purpose": "verify length",
            "input": {"text": text, "target": target_int},
            "output": result,
            "tokens": {"in": 0, "out": 0},
        })
        trace_entry = {
            "step": step_counter[0],
            "kind": "length_checker",
            "length": result.get("length", 0),
        }
        if "diff" in result:
            trace_entry["diff"] = result["diff"]
            trace_entry["hit"] = result.get("hit", False)
        trace.append(trace_entry)

        # Per-call live log (matches method_brain's style).
        if result.get("target") is not None:
            diff_str = (f"diff={result['diff']:+d}" if not result.get("hit")
                        else "diff=  0")
            status = "Pass" if result.get("hit") else "Fail"
            print(f"  writer #{exec_state.writer_calls}→verify: {status}  "
                  f"len={result['length']:>3}  {diff_str}", flush=True)
        else:
            print(f"  writer #{exec_state.writer_calls}→verify: "
                  f"len={result['length']:>3}", flush=True)
        return result

    return writer, length_checker


# ---------------------------------------------------------------------------
# Per-cycle exec
# ---------------------------------------------------------------------------

def _exec_solve(code: str, theme_desc: str, metrics: MetricsTracker,
                trajectory: list, step_counter: list) -> dict:
    """Compile + run brain's code for one cycle.

    Returns dict:
      strategy_notes  — docstring of solve() (or "" if none)
      solve_return    — return value of solve(...) (str or None)
      exec_state      — _ExecState (writer_calls, last_text, last_length)
      trace           — per-call compact trace
      error           — None on success, else a string ("syntax_error: ...",
                        "no_solve_function", "budget_exceeded: ...",
                        "runtime_error: ...")
    """
    exec_state = _ExecState()
    trace: list[dict] = []
    writer, length_checker = _make_wrappers(
        theme_desc, metrics, trajectory, step_counter, trace, exec_state
    )

    namespace: dict = {
        "__name__": "method_brain_code_cycle",
        "writer": writer,
        "length_checker": length_checker,
        "TARGET_LEN": TARGET_LEN,
    }
    error: str | None = None
    solve_fn = None
    strategy_notes = ""
    solve_return = None

    try:
        compiled = compile(code, "<brain_code>", "exec")
    except SyntaxError as e:
        return {
            "strategy_notes": "",
            "solve_return": None,
            "exec_state": exec_state,
            "trace": trace,
            "error": f"syntax_error: {e}",
        }

    try:
        exec(compiled, namespace)
    except Exception as e:
        return {
            "strategy_notes": "",
            "solve_return": None,
            "exec_state": exec_state,
            "trace": trace,
            "error": f"toplevel_error: {type(e).__name__}: {e}",
        }

    solve_fn = namespace.get("solve")
    if not callable(solve_fn):
        return {
            "strategy_notes": "",
            "solve_return": None,
            "exec_state": exec_state,
            "trace": trace,
            "error": "no_solve_function",
        }

    strategy_notes = (getattr(solve_fn, "__doc__", "") or "").strip()

    try:
        solve_return = solve_fn(
            theme_desc, TARGET_LEN, writer, length_checker,
        )
    except BudgetExceeded as e:
        error = f"budget_exceeded: {e}"
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        error = f"runtime_error: {type(e).__name__}: {e}\n{tb}"

    return {
        "strategy_notes": strategy_notes,
        "solve_return": solve_return,
        "exec_state": exec_state,
        "trace": trace,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Per-run flow
# ---------------------------------------------------------------------------

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str,
            memory_context: list[dict] | None = None) -> dict:
    metrics = MetricsTracker()
    memory_context = memory_context or []

    brain_cfg = ROLE_CONFIGS["brain"]
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
            system_prompt=build_brain_system_prompt(),
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
                "code": c["code"],
                "strategy_notes": c["strategy_notes"],
                "trace": c["trace"],
                "final_length": c["final_length"],
                "hit": c["hit"],
                "error": c.get("error"),
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
            "purpose": f"design cycle {cycle_idx} code",
            "input": user_input,
            "output": brain_output,
            "tokens": brain_tokens,
        })
        mem_note = (f" (sees {len(memory_context)} cross-run + "
                    f"{len(prior_for_brain)} within-run)"
                    if memory_context or prior_for_brain else "")
        print(f"  cycle {cycle_idx}: brain design{mem_note}", flush=True)

        code = parse_code(brain_output)
        if code is None:
            trajectory.append({
                "step": next_step(),
                "role": "system",
                "purpose": f"CODE_UNPARSEABLE (cycle {cycle_idx})",
                "input": None,
                "output": "Could not extract a `def solve` from brain output",
                "tokens": {"in": 0, "out": 0},
            })
            cycles.append({
                "cycle": cycle_idx,
                "code": None,
                "strategy_notes": "",
                "trace": [],
                "final_length": final_length,
                "hit": False,
                "writer_calls_used_in_cycle": 0,
                "strategy_validated": False,
                "error": "code_unparseable",
            })
            overall_error = "code_unparseable"
            continue

        # ---- Execute this cycle's code ----
        result = _exec_solve(
            code=code, theme_desc=theme_desc,
            metrics=metrics, trajectory=trajectory,
            step_counter=step_counter,
        )
        exec_state: _ExecState = result["exec_state"]
        strategy_notes = result["strategy_notes"]
        cycle_error = result["error"]
        cycle_trace = result["trace"]

        # Final outcome resolution. Prefer solve()'s return value if it
        # returned a non-empty string AND no error fired AFTER writer ran.
        # Otherwise fall back to the last writer output observed.
        solve_return = result["solve_return"]
        if isinstance(solve_return, str) and solve_return:
            # Length-check whatever solve returned (canonical outcome).
            check = _length_checker_fn(solve_return, target=TARGET_LEN)
            step_counter[0] += 1
            trajectory.append({
                "step": step_counter[0],
                "role": "length_checker",
                "purpose": f"host final check of solve() return (cycle {cycle_idx})",
                "input": {"text": solve_return, "target": TARGET_LEN},
                "output": check,
                "tokens": {"in": 0, "out": 0},
            })
            cycle_trace.append({
                "step": step_counter[0],
                "kind": "length_checker",
                "length": check["length"],
                "diff": check["diff"],
                "hit": check["hit"],
            })
            cycle_final_text = solve_return
            cycle_final_length = check["length"]
            cycle_hit = check["hit"]
            if check.get("target") is not None:
                diff_str = (f"diff={check['diff']:+d}" if not check["hit"]
                            else "diff=  0")
                status = "Pass" if check["hit"] else "Fail"
                print(f"  cycle {cycle_idx} final return → {status}  "
                      f"len={check['length']:>3}  {diff_str}", flush=True)
        elif exec_state.last_text:
            cycle_final_text = exec_state.last_text
            cycle_final_length = exec_state.last_length
            cycle_hit = cycle_final_length == TARGET_LEN
        else:
            # No writer call ever ran or it returned "".
            cycle_final_text = final_text
            cycle_final_length = final_length
            cycle_hit = False

        if cycle_final_text:
            final_text = cycle_final_text
            final_length = cycle_final_length

        # strategy_validated: did the planned multi-step strategy actually
        # contribute? NOT (Pass AND writer_calls == 1). Mirrors method_brain.
        strategy_validated = exec_state.writer_calls >= 1 and not (
            cycle_hit and exec_state.writer_calls == 1
        )

        cycles.append({
            "cycle": cycle_idx,
            "code": code,
            "strategy_notes": strategy_notes,
            "trace": cycle_trace,
            "final_length": cycle_final_length,
            "hit": cycle_hit,
            "writer_calls_used_in_cycle": exec_state.writer_calls,
            "length_checker_calls_used_in_cycle": exec_state.length_checker_calls,
            "strategy_validated": strategy_validated,
            "error": cycle_error,
        })

        if cycle_error:
            print(f"  cycle {cycle_idx}: error — {cycle_error.splitlines()[0]}",
                  flush=True)

        if cycle_hit:
            hit = True
            print(f"  cycle {cycle_idx}: Pass — exiting run early", flush=True)
            break

    # ---- Save final story ----
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
    print(f"=== method_brain_code (story task) started {started_at} ===")
    print(f"  themes  : {len(THEMES)}")
    print(f"  runs    : {RUNS_PER_THEME} per theme")
    print(f"  loop    : {NUM_CYCLES} cycles per run "
          f"(each cycle: brain → python code → exec)")
    print(f"  budget  : {WRITER_CALL_CAP_PER_CYCLE} writer calls per cycle "
          f"(max {max_writer} per run); length_checker safety belt = "
          f"{LENGTH_CHECKER_CALL_CAP_PER_CYCLE}/cycle")
    print(f"  target  : exactly {TARGET_LEN} characters")
    print(f"  output  : {output_dir}")

    all_results: list[dict] = []
    theme_memory: dict[str, list[dict]] = {}
    for theme_id, theme_desc in THEMES:
        theme_memory[theme_id] = []
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir,
                        memory_context=list(theme_memory[theme_id]))
            all_results.append(r)

            def _cycle_snapshot(c: dict | None) -> dict | None:
                if c is None:
                    return None
                return {
                    "cycle": c.get("cycle"),
                    "final_length": c.get("final_length"),
                    "hit": c.get("hit", False),
                    "strategy_notes": c.get("strategy_notes", ""),
                    "code": c.get("code"),
                    "trace": c.get("trace", []),
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
        "method": "method_brain_code",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "target_len": TARGET_LEN,
            "runs_per_theme": RUNS_PER_THEME,
            "num_cycles": NUM_CYCLES,
            "writer_call_cap_per_cycle": WRITER_CALL_CAP_PER_CYCLE,
            "length_checker_call_cap_per_cycle": LENGTH_CHECKER_CALL_CAP_PER_CYCLE,
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

    from runners.story_task._metrics import per_theme_counts, overall_counts
    by_theme = per_theme_counts(summary, "method_brain_code")
    overall = overall_counts(summary, "method_brain_code")

    def _fmt(num: int, den: int) -> str:
        return f"{num}/{den} ({num/den:.0%})" if den else "(no data)"

    print()
    print("=" * 60)
    print("method_brain_code summary")
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


if __name__ == "__main__":
    main()
