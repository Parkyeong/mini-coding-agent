"""method_fixed — fixed agent loop with periodic textplanner ("coach").

Per (theme, run): the host runs NUM_CYCLES = 8 cycles, plus one trailing
writer-verify at the end to consume the final cycle's textplanner advice.
Comparison fairness across methods is on the "agent main loop iteration
count" axis (= 8), NOT on writer call count.

    cycle 1:  writer-verify × 3       (cold; no textplanner advice yet)
              ↓
              textplanner              (reads latest attempt + verifier;
                                        advice persists into next cycle)
    cycle 2:  writer-verify × 3       (FIRST call here uses cycle-1 textplanner advice)
              ↓
              textplanner
    ...
    cycle 8:  writer-verify × 3       (uses cycle-7 textplanner advice)
              ↓
              textplanner              (last advice)
    trailing: writer-verify × 1       (consumes cycle-8 textplanner advice)

Any verifier HIT (length == 241) immediately breaks the whole run — the 8
cycles is a MAX, not a floor.

Budget per run (no early-HIT case):
    8 cycles × (3 writer + 1 textplanner)  +  1 trailing writer
  = 25 writer calls + 8 textplanner calls

This is different from baseline (8 writer calls) and method_brain's current
WRITER_CALL_CAP (8). The cross-method comparison is now on "main loop
iterations", not writer count.

Reference: Parkyeong/general_agent/runners/agent_run.py — the "coach" fires
every N writer failures and its advice persists in the conversation. The
shape is host-fixed — textplanner has NO control over how many writer calls
happen or when it itself runs.

Outputs (under WORKSHOP/story_241/method_fixed/):
  <theme_id>/run_<N>.txt   — final story text (HIT or last MISS)
  summary.json             — flat trajectory (every writer / length_checker /
                             textplanner step with full input/output) +
                             per-cycle summary + per-role token totals

Usage:
  python -m runners.story_task.method_fixed
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
from role_pool import textplanner as textplanner_role
from role_pool import writer as writer_role
from tool_pool.text_utils import length_checker


# ---------------------------------------------------------------------------
# Task spec — locked across all three story-task methods
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

# Loop shape constants — fixed by host.
#
# NUM_CYCLES is the "agent main loop iteration count" that we compare across
# methods (baseline = 8 writer-verify iters; method_fixed = 8 of these cycles;
# method_brain workflow = 8 inner iters).
#
# Each cycle = WRITERS_PER_CYCLE writer-verify calls, then one textplanner
# call. The textplanner's advice carries into the NEXT cycle's first writer.
# After the final (8th) cycle, one trailing writer-verify consumes that last
# textplanner's advice — otherwise it'd be wasted.
#
# Worst case per run (no early HIT):
#   NUM_CYCLES * WRITERS_PER_CYCLE + 1 = 8 * 3 + 1 = 25 writer calls
#   NUM_CYCLES                         = 8         textplanner calls
NUM_CYCLES = 8
WRITERS_PER_CYCLE = 3
MAX_WRITER_CALLS = NUM_CYCLES * WRITERS_PER_CYCLE + 1   # = 25

# If STORY_EXP_NAME is set (typically by run_all.py --exp), put results
# under story_241/<exp_name>/<method>/.
_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
OUTPUT_SUBDIR = (
    os.path.join("story_241", _EXP_NAME, "method_fixed") if _EXP_NAME
    else os.path.join("story_241", "method_fixed")
)


# ---------------------------------------------------------------------------
# Per-call dispatch — fresh LLMNode each call so message history doesn't
# accumulate across iterations (each call rebuilds its user message from
# scratch with the latest previous_attempt + planner advice baked in).
# ---------------------------------------------------------------------------

def _call_textplanner(theme_desc: str, previous_attempt: str,
                      verifier_result: dict | None,
                      metrics: MetricsTracker) -> tuple[str, str, dict]:
    """One textplanner invocation. Returns (advice_text, user_input, tokens)."""
    cfg = ROLE_CONFIGS["textplanner"]
    node = LLMNode(
        system_prompt=textplanner_role.PROMPT,
        role="textplanner",
        max_steps=cfg["max_steps"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        metrics_tracker=metrics,
    )
    user_input = textplanner_role.build_input(
        theme=theme_desc,
        target_len=TARGET_LEN,
        previous_attempt=previous_attempt,
        verifier_result=verifier_result,
    )
    n0 = len(metrics.calls)
    node.reset_message()
    response = node.run(user_input)
    advice = (response.get("text") or "").strip()
    new_calls = metrics.calls[n0:]
    tokens = {
        "in": sum(c.input_tokens for c in new_calls),
        "out": sum(c.output_tokens for c in new_calls),
    }
    return advice, user_input, tokens


def _call_writer(theme_desc: str, planner_advice: str, previous_attempt: str,
                 metrics: MetricsTracker) -> tuple[str, str, dict]:
    """One writer invocation. Fresh LLMNode. Returns (story, user_input, tokens).

    planner_advice — most recent textplanner advice (empty string before the
        first textplanner runs). Passed as `guidance` so the writer prompt
        layers it above the previous_attempt + host-computed direction.
    previous_attempt — prior story text (empty on the very first writer call).
        writer_role.build_input auto-adds the host-computed exact direction
        ("trim N chars from end") when previous_attempt is non-empty.
    """
    cfg = ROLE_CONFIGS["writer"]
    node = LLMNode(
        system_prompt=writer_role.PROMPT,
        role="writer",
        max_steps=cfg["max_steps"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        metrics_tracker=metrics,
    )
    user_input = writer_role.build_input(
        theme=theme_desc,
        guidance=planner_advice,
        feedback="",                        # already encoded via auto-direction
        previous_attempt=previous_attempt,
        target_len=TARGET_LEN,
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
    return story, user_input, tokens


# ---------------------------------------------------------------------------
# One run: NUM_CYCLES × (writer-verify × 3, textplanner, writer-verify × 1),
# early-exit on HIT.
# ---------------------------------------------------------------------------

class _Hit(Exception):
    """Raised to unwind nested loops on a HIT. Carries no payload — final
    state is on the closure variables in run_one()."""


def run_one(theme_id: str, theme_desc: str, run_idx: int,
            output_dir: str) -> dict:
    metrics = MetricsTracker()
    trajectory: list[dict] = []
    step_counter = [0]

    def append_step(role, purpose, input_, output_, tokens):
        step_counter[0] += 1
        trajectory.append({
            "step": step_counter[0],
            "role": role,
            "purpose": purpose,
            "input": input_,
            "output": output_,
            "tokens": tokens,
        })

    # Mutable state threaded through the loop.
    state = {
        "previous_attempt": "",
        "verifier_result": None,
        "planner_advice": "",
        "writer_calls_used": 0,
        "final_text": "",
        "final_length": 0,
        "hit": False,
    }
    cycles_data: list[dict] = []   # per-cycle summary for the HTML view

    def _do_writer_verify(target_list, cycle_idx: int, phase: str, sub_idx: int):
        """One writer→length_checker pair. Appends the attempt record to
        `target_list` BEFORE possibly raising _Hit (so the winning attempt
        is always preserved in the data). Updates state."""
        state["writer_calls_used"] += 1
        wc = state["writer_calls_used"]

        text, w_input, w_tokens = _call_writer(
            theme_desc,
            planner_advice=state["planner_advice"],
            previous_attempt=state["previous_attempt"],
            metrics=metrics,
        )
        append_step("writer",
                    f"cycle {cycle_idx} {phase} #{sub_idx} (writer call {wc})",
                    w_input, text, w_tokens)

        check = length_checker(text, target=TARGET_LEN)
        append_step("length_checker",
                    f"verify cycle {cycle_idx} {phase} #{sub_idx}",
                    {"text": text, "target": TARGET_LEN}, check,
                    {"in": 0, "out": 0})

        diff_str = (f"diff={check['diff']:+d}" if not check["hit"]
                    else "diff=  0")
        status = "Pass" if check["hit"] else "Fail"
        print(f"  cycle {cycle_idx} {phase} #{sub_idx} "
              f"(writer #{wc}): {status}  "
              f"len={check['length']:>3}  {diff_str}", flush=True)

        attempt = {
            "length": check["length"],
            "diff": check["diff"],
            "hit": check["hit"],
        }
        # Record BEFORE the _Hit raise so the winning attempt is preserved.
        if target_list is not None:
            target_list.append(attempt)

        state["final_text"] = text
        state["final_length"] = check["length"]
        state["previous_attempt"] = text
        state["verifier_result"] = check

        if check["hit"]:
            state["hit"] = True
            raise _Hit()

    trailing_attempt: dict | None = None   # the final writer-verify after cycle 8
    trailing_box: list[dict] = []          # populated in the trailing phase

    try:
        for cycle in range(1, NUM_CYCLES + 1):
            cycle_record = {
                "cycle": cycle,
                "attempts": [],          # writer-verify outcomes this cycle
                "textplanner_advice": "",  # given at END of cycle
            }
            cycles_data.append(cycle_record)

            # ---- Phase 1: writer-verify × WRITERS_PER_CYCLE ----
            for sub in range(1, WRITERS_PER_CYCLE + 1):
                _do_writer_verify(cycle_record["attempts"], cycle, "w-v", sub)

            # ---- Phase 2: textplanner — every cycle, including the last.
            # The last cycle's advice is consumed by the trailing writer
            # after the loop. ----
            advice, tp_input, tp_tokens = _call_textplanner(
                theme_desc,
                previous_attempt=state["previous_attempt"],
                verifier_result=state["verifier_result"],
                metrics=metrics,
            )
            append_step("textplanner", f"end of cycle {cycle} (coach)",
                        tp_input, advice, tp_tokens)
            print(f"  cycle {cycle} → textplanner (coach)", flush=True)
            state["planner_advice"] = advice
            cycle_record["textplanner_advice"] = advice

        # ---- Trailing writer-verify: consumes cycle-8 textplanner advice ----
        # _do_writer_verify appends into trailing_box BEFORE possibly raising
        # _Hit (append-before-raise contract).
        _do_writer_verify(trailing_box, NUM_CYCLES + 1, "trailing", 1)
    except _Hit:
        # The winning attempt is already recorded — in cycle_record["attempts"]
        # if it happened inside a cycle, or in trailing_box if it was trailing.
        pass

    if trailing_box:
        trailing_attempt = trailing_box[0]

    # ---- Save story (HIT or last MISS) ----
    theme_dir = os.path.join(output_dir, theme_id)
    os.makedirs(theme_dir, exist_ok=True)
    with open(os.path.join(theme_dir, f"run_{run_idx}.txt"),
              "w", encoding="utf-8") as f:
        f.write(state["final_text"])

    by_role = metrics.by_role()
    total_in = sum(r["input_tokens"] for r in by_role.values())
    total_out = sum(r["output_tokens"] for r in by_role.values())

    return {
        "theme_id": theme_id,
        "theme_desc": theme_desc,
        "run_idx": run_idx,
        "hit": state["hit"],
        "final_length": state["final_length"],
        "writer_calls_used": state["writer_calls_used"],
        "cycles": cycles_data,
        "trailing_attempt": trailing_attempt,
        "trajectory": trajectory,
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
    print(f"=== method_fixed (story task) started {started_at} ===")
    print(f"  themes  : {len(THEMES)}")
    print(f"  runs    : {RUNS_PER_THEME} per theme")
    print(f"  cycle   : writer-verify × {WRITERS_PER_CYCLE} → textplanner")
    print(f"  loop    : {NUM_CYCLES} cycles + 1 trailing writer-verify "
          f"(max {MAX_WRITER_CALLS} writer calls, {NUM_CYCLES} textplanner calls)")
    print(f"  target  : exactly {TARGET_LEN} characters")
    print(f"  output  : {output_dir}")

    all_results: list[dict] = []
    for theme_id, theme_desc in THEMES:
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir)
            all_results.append(r)
            status = "Pass" if r["hit"] else f"Fail (len={r['final_length']})"
            by_role = r["tokens_by_role"]
            tp_in = by_role.get("textplanner", {}).get("input_tokens", 0)
            tp_out = by_role.get("textplanner", {}).get("output_tokens", 0)
            w_in = by_role.get("writer", {}).get("input_tokens", 0)
            w_out = by_role.get("writer", {}).get("output_tokens", 0)
            print(f"  → run result: {status}  | "
                  f"writer calls: {r['writer_calls_used']}/{MAX_WRITER_CALLS}  "
                  f"| cycles entered: {len(r['cycles'])}/{NUM_CYCLES}")
            print(f"     tokens — textplanner: in={tp_in}, out={tp_out}  |  "
                  f"writer: in={w_in}, out={w_out}")

    # ----- Aggregate -----
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
        "method": "method_fixed",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "target_len": TARGET_LEN,
            "runs_per_theme": RUNS_PER_THEME,
            "num_cycles": NUM_CYCLES,
            "writers_per_cycle": WRITERS_PER_CYCLE,
            "max_writer_calls": MAX_WRITER_CALLS,
            "themes": [{"id": t[0], "desc": t[1]} for t in THEMES],
            "writer_role_config": ROLE_CONFIGS["writer"],
            "textplanner_role_config": ROLE_CONFIGS["textplanner"],
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
    by_theme = per_theme_counts(summary, "method_fixed")
    overall = overall_counts(summary, "method_fixed")

    def _fmt(num: int, den: int) -> str:
        return f"{num}/{den} ({num/den:.0%})" if den else "(no data)"

    print()
    print("=" * 60)
    print("method_fixed summary")
    print("=" * 60)
    print(f"  overall pass rate (per run)   : "
          f"{_fmt(overall['runs_hits'], overall['runs_total'])}")
    print(f"  overall pass rate (per cycle) : "
          f"{_fmt(overall['cycle_hits'], overall['cycle_total'])}")
    print(f"  per-theme:")
    for tid, m in by_theme.items():
        print(f"    {tid:<26} "
              f"run {_fmt(m['runs_hits'], m['runs_total']):>13}  "
              f"cyc {_fmt(m['cycle_hits'], m['cycle_total']):>13}")
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
