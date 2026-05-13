"""method_brain — brain designs a workflow program, host interprets it.

Single-file architecture for X2: BRAIN_PROMPT + DSL interpreter + runner.

How it differs from method_fixed:
  - method_fixed: brain outputs simple plan (writer guidance), host has
    hardcoded retry loop (4 inner × 2 brain attempts = 8 writer calls)
  - method_brain: brain outputs a full workflow PROGRAM in a small JSON DSL
    (sequence/loop/call/if/return + variables). Host is a generic
    interpreter for this DSL — it does NOT hardcode the retry shape.
    Brain decides loop counts, when to stop, multi-phase strategies, etc.

The brain (4.1-mini) is INDEPENDENT of role_pool — it's not a role itself,
it consumes role_pool / tool_pool as a menu via discovery functions. Workers
(writer, length_checker) do the actual work, dispatched by host's executor.

Budget cap: 8 writer calls per (theme, run). Matches method_fixed / baseline.
If brain writes a workflow that tries to call writer 9+ times, host silently
skips the excess and records BUDGET_EXHAUSTED in the trajectory.

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

THEMES: list[tuple[str, str]] = [
    ("mountain_school",       "The lone teacher at a remote mountain school"),
    ("time_displaced_store",  "A convenience store displaced in time"),
    ("photo_studio_last_day", "The final day of an old photo studio"),
    ("rainy_night_bus",       "The last bus on a rainy night"),
]
TARGET_LEN = 241
TASK_TYPE = "story"
RUNS_PER_THEME = 4
WRITER_CALL_CAP = 8
OUTPUT_SUBDIR = os.path.join("story_241", "method_brain")


# ---------------------------------------------------------------------------
# BRAIN PROMPT
# ---------------------------------------------------------------------------

BRAIN_PROMPT_TEMPLATE = """You are a workflow architect. Design a workflow
PROGRAM (in the JSON DSL below) to accomplish the user's task.

You do NOT execute the workflow yourself — a Python interpreter will run it.
Your job is to design what shape the workflow takes (loops, conditionals,
sequencing, when to stop) and which roles/tools each step invokes.

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

- **Budget: max {writer_cap} writer calls total per task.** If your workflow
  tries to invoke writer beyond this, host will silently skip the excess
  calls and continue (the variable `save_as` for the skipped call will not
  be set, so subsequent `$<var>` references resolve to empty).
- SUCCESS condition: the final `save_as` value of the last successful writer
  call must produce a story of EXACTLY 241 characters. Use length_checker to
  verify; the host considers the run a HIT iff the latest writer output has
  length 241.
- Aim to use few writer calls when possible (token cost matters).

# Example workflow

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
the task. Multi-phase strategies (try guidance A first, then switch to B if
A keeps overshooting) are legitimate uses of the DSL.

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
        writer_cap=WRITER_CALL_CAP,
    )


def build_initial_input(theme_desc: str, target_len: int = TARGET_LEN) -> str:
    return (
        f"Task: Write a story about: {theme_desc}\n"
        f"Requirement: exactly {target_len} characters.\n\n"
        f"Design your workflow. Output JSON only."
    )


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
        self.writer_cap = WRITER_CALL_CAP
        self.step = starting_step
        self.returned = False
        self.return_value = None
        self.last_text: str = ""
        self.last_length: int = 0


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
        # Hard safety cap: never let max_iter exceed WRITER_CALL_CAP * 4.
        max_iter = min(max_iter, WRITER_CALL_CAP * 4)
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

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str) -> dict:
    metrics = MetricsTracker()

    # ---- Brain (single shot) ----
    brain_cfg = ROLE_CONFIGS["brain"]
    brain = LLMNode(
        system_prompt=build_brain_system_prompt(),
        role="brain",
        max_steps=brain_cfg["max_steps"],
        model=brain_cfg["model"],
        temperature=brain_cfg["temperature"],
        max_tokens=brain_cfg["max_tokens"],
        metrics_tracker=metrics,
    )
    initial_input = build_initial_input(theme_desc, TARGET_LEN)

    n0 = len(metrics.calls)
    brain_response = brain.run(initial_input)
    brain_output = (brain_response.get("text") or "").strip()
    brain_calls = metrics.calls[n0:]
    brain_tokens = {
        "in": sum(c.input_tokens for c in brain_calls),
        "out": sum(c.output_tokens for c in brain_calls),
    }

    trajectory: list[dict] = [{
        "step": 1,
        "role": "brain",
        "purpose": "design workflow",
        "input": initial_input,
        "output": brain_output,
        "tokens": brain_tokens,
    }]

    plan = parse_workflow(brain_output)
    if plan is None or "workflow" not in plan:
        # Brain failed to produce a parseable workflow — terminate here.
        trajectory.append({
            "step": 2,
            "role": "system",
            "purpose": "PLAN_UNPARSEABLE",
            "input": None,
            "output": "Could not parse brain output as JSON containing a 'workflow' key",
            "tokens": {"in": 0, "out": 0},
        })
        return _build_run_record(
            theme_id, theme_desc, run_idx, output_dir,
            hit=False, final_text="", final_length=0,
            trajectory=trajectory, strategy_notes="",
            workflow=None, writer_calls_used=0,
            error="plan_unparseable", metrics=metrics,
        )

    workflow = plan.get("workflow")
    strategy_notes = plan.get("strategy_notes", "")

    # ---- Execute workflow ----
    ctx = ExecutionContext(
        theme=theme_desc, target_len=TARGET_LEN,
        metrics=metrics, trajectory=trajectory, starting_step=len(trajectory),
    )
    error = None
    try:
        execute(workflow, ctx)
    except WorkflowError as e:
        error = f"workflow_error: {e}"
        ctx.step += 1
        ctx.trajectory.append({
            "step": ctx.step,
            "role": "system",
            "purpose": "INTERPRETER_ERROR",
            "input": None,
            "output": str(e),
            "tokens": {"in": 0, "out": 0},
        })

    # ---- Determine outcome (HIT iff last_length == 241) ----
    hit = ctx.last_length == TARGET_LEN
    final_text = ctx.last_text
    final_length = ctx.last_length

    return _build_run_record(
        theme_id, theme_desc, run_idx, output_dir,
        hit=hit, final_text=final_text, final_length=final_length,
        trajectory=ctx.trajectory, strategy_notes=strategy_notes,
        workflow=workflow, writer_calls_used=ctx.writer_calls,
        error=error, metrics=metrics,
    )


def _build_run_record(theme_id, theme_desc, run_idx, output_dir, *,
                      hit, final_text, final_length, trajectory,
                      strategy_notes, workflow, writer_calls_used,
                      error, metrics) -> dict:
    theme_dir = os.path.join(output_dir, theme_id)
    os.makedirs(theme_dir, exist_ok=True)
    with open(os.path.join(theme_dir, f"run_{run_idx}.txt"), "w", encoding="utf-8") as f:
        f.write(final_text)

    by_role = metrics.by_role()
    total_in = sum(r["input_tokens"] for r in by_role.values())
    total_out = sum(r["output_tokens"] for r in by_role.values())

    return {
        "theme_id": theme_id,
        "theme_desc": theme_desc,
        "run_idx": run_idx,
        "hit": hit,
        "final_length": final_length,
        "writer_calls_used": writer_calls_used,
        "strategy_notes": strategy_notes,
        "workflow": workflow,
        "trajectory": trajectory,
        "error": error,
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
    print(f"=== method_brain (story task) started {started_at} ===")
    print(f"  themes  : {len(THEMES)}")
    print(f"  runs    : {RUNS_PER_THEME} per theme")
    print(f"  budget  : {WRITER_CALL_CAP} writer calls per run")
    print(f"  target  : exactly {TARGET_LEN} characters")
    print(f"  output  : {output_dir}")

    all_results: list[dict] = []
    for theme_id, theme_desc in THEMES:
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir)
            all_results.append(r)
            status = "HIT" if r["hit"] else f"MISS (len={r['final_length']})"
            err = f"  [error: {r['error']}]" if r.get("error") else ""
            print(f"  {status}  | writer calls used: {r['writer_calls_used']}{err}")
            print(f"  tokens: in={r['tokens_total_input']}, out={r['tokens_total_output']}")
            for role_name, m in r["tokens_by_role"].items():
                print(f"    [{role_name}] {m['calls']} calls, "
                      f"in={m['input_tokens']}, out={m['output_tokens']}")

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
            "writer_call_cap": WRITER_CALL_CAP,
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

    print()
    print("=" * 60)
    print(f"method_brain summary")
    print("=" * 60)
    print(f"  hit rate     : {hits}/{total_runs} ({summary['totals']['hit_rate']:.0%})")
    print(f"  tokens total : in={sum_in}, out={sum_out}, sum={sum_in + sum_out}")
    print(f"  per-role     :")
    for role_name, m in grand_by_role.items():
        print(f"    [{role_name}] {m['calls']} calls, "
              f"in={m['input_tokens']}, out={m['output_tokens']}")
    print(f"\nSaved: {summary_path}")

    # Auto-render comparison HTML (fail-soft — other methods may be missing)
    try:
        from runners.story_task import html as story_html
        story_html.main()
    except Exception as e:
        print(f"[warn] HTML render failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
