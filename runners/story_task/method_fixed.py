"""method_fixed — brain plans, host runs a fixed pipeline.

(File is "method_fixed" not "engine_fixed" because for story task there's
no dataset to iterate, so runner + engine collapse into one method file.
Conceptually it still does engine work — the `run_one()` function is the
per-(theme, run) orchestration. MBPP keeps runner/engine split because the
dataset has 257 cases that need batched iteration.)

Method shape per (theme, run):

    brain attempt 1 (initial plan):
        brain.run(initial_input)  → JSON plan with writer args (guidance)
        parse plan
        inner loop, up to 4 retries:
            fresh writer LLMNode → run with theme + guidance + feedback
            length_checker(story, target=241) → result dict
            if hit: SUCCESS, break out of entire run
            else: build feedback ("X chars, off by Y") for next retry

    if 4 inner retries failed:
        brain attempt 2 (replan, SAME LLMNode, messages accumulate):
            brain.run(replan_input with failure trace)
            new plan
            inner loop again (up to 4 retries)

    if both brain attempts × 4 retries failed → MISS, save last story

Hardcoded by host (brain has NO control over these):
  - INNER_MAX_RETRIES = 4
  - BRAIN_MAX_ATTEMPTS = 2
  - The shape: writer → length_checker → retry on miss
  - The feedback format ("X chars, off by Y. Adjust to exactly 241.")

Brain has control over:
  - writer guidance (the main lever)
  - strategy_notes (for trace inspection only)

Per task spec: gpt-4o-mini is locked for writer (the worker producing the
prose). Brain uses gpt-4.1-mini (set in ROLE_CONFIGS).

Outputs (under WORKSHOP/story_241/method_fixed/):
  <theme_id>/run_<N>.txt   — final story text (HIT or last MISS)
  summary.json             — full per-run trace + per-role token totals

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
from role_pool import brain as brain_role
from role_pool import writer as writer_role
from tool_pool.text_utils import length_checker


# ---------------------------------------------------------------------------
# Task spec — locked across all three story-task methods
# ---------------------------------------------------------------------------

THEMES: list[tuple[str, str]] = [
    ("mountain_school",       "The lone teacher at a remote mountain school"),
    ("time_displaced_store",  "A convenience store displaced in time"),
    ("photo_studio_last_day", "The final day of an old photo studio"),
    ("rainy_night_bus",       "The last bus on a rainy night"),
]
TARGET_LEN = 241
TASK_TYPE = "story"
RUNS_PER_THEME = 3
INNER_MAX_RETRIES = 4
BRAIN_MAX_ATTEMPTS = 2
OUTPUT_SUBDIR = os.path.join("story_241", "method_fixed")


# ---------------------------------------------------------------------------
# Writer dispatch — fresh LLMNode per call, shares the run's MetricsTracker
# ---------------------------------------------------------------------------

def run_writer_fresh(theme: str, guidance: str, feedback: str,
                     metrics_tracker: MetricsTracker) -> str:
    """One writer invocation. Fresh LLMNode each call: no message accumulation,
    feedback / guidance go into the user message via writer_role.build_input."""
    cfg = ROLE_CONFIGS["writer"]
    node = LLMNode(
        system_prompt=writer_role.PROMPT,
        role="writer",
        max_steps=cfg["max_steps"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        metrics_tracker=metrics_tracker,
    )
    return writer_role.run_writer(
        node, theme=theme, guidance=guidance, feedback=feedback,
    )


# ---------------------------------------------------------------------------
# One run: brain attempts × inner retries
# ---------------------------------------------------------------------------

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str) -> dict:
    """One full attempt sequence for one (theme, run_idx)."""
    metrics = MetricsTracker()

    # Brain LLMNode — same instance across both brain attempts so its
    # messages accumulate (initial plan → replan sees its own previous plan).
    brain_cfg = ROLE_CONFIGS["brain"]
    brain = LLMNode(
        system_prompt=brain_role.build_system_prompt(TASK_TYPE),
        role="brain",
        max_steps=brain_cfg["max_steps"],
        model=brain_cfg["model"],
        temperature=brain_cfg["temperature"],
        max_tokens=brain_cfg["max_tokens"],
        metrics_tracker=metrics,
    )

    rounds: list[dict] = []
    hit = False
    final_text = ""
    final_length = 0

    for brain_attempt in range(1, BRAIN_MAX_ATTEMPTS + 1):
        # ---- Brain call: initial or replan ----
        if brain_attempt == 1:
            user_input = brain_role.build_initial_input(theme_desc, TARGET_LEN)
        else:
            user_input = brain_role.build_replan_input(rounds[-1], TARGET_LEN)

        # IMPORTANT: do NOT call brain.reset_message() between attempts —
        # brain's accumulated context (its prev plan + failure trace) is
        # essential for replan quality.
        brain_response = brain.run(user_input)
        brain_output = (brain_response.get("text") or "").strip()

        plan = brain_role.parse_plan(brain_output)

        if plan is None:
            # Plan unparseable: terminate this run with the failure recorded.
            rounds.append({
                "brain_attempt": brain_attempt,
                "brain_output": brain_output,
                "plan": None,
                "error": "plan unparseable",
                "inner_attempts": [],
                "final_length": final_length,
                "hit": False,
            })
            break

        # ---- Extract writer guidance from plan ----
        writer_args: dict = {}
        for step in plan.get("steps", []) or []:
            if step.get("role") == "writer":
                writer_args = step.get("args", {}) or {}
                break
        guidance = writer_args.get("guidance", "") or ""

        # ---- Inner loop: writer × INNER_MAX_RETRIES, verify each ----
        inner_attempts: list[dict] = []
        feedback = ""
        round_hit = False
        last_text = ""
        last_length = 0

        for inner_attempt in range(1, INNER_MAX_RETRIES + 1):
            text = run_writer_fresh(theme_desc, guidance, feedback, metrics)
            check = length_checker(text, target=TARGET_LEN)

            inner_attempts.append({
                "attempt": inner_attempt,
                "length": check["length"],
                "diff": check["diff"],
                "feedback_in": feedback,
                "guidance": guidance,
            })
            last_text = text
            last_length = check["length"]

            if check["hit"]:
                round_hit = True
                break

            feedback = (
                f"Previous attempt was {check['length']} characters, "
                f"{check['delta_text']}. Adjust to exactly {TARGET_LEN}."
            )

        rounds.append({
            "brain_attempt": brain_attempt,
            "brain_output": brain_output,
            "plan": plan,
            "strategy_notes": plan.get("strategy_notes", ""),
            "guidance_used": guidance,
            "inner_attempts": inner_attempts,
            "final_length": last_length,
            "hit": round_hit,
        })

        # Track latest text/length for the run's final output.
        final_text = last_text
        final_length = last_length

        if round_hit:
            hit = True
            break

    # ---- Save story (HIT or last MISS) ----
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
        "rounds": rounds,
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
    print(f"  brain   : up to {BRAIN_MAX_ATTEMPTS} attempts")
    print(f"  inner   : up to {INNER_MAX_RETRIES} retries per brain attempt")
    print(f"  target  : exactly {TARGET_LEN} characters")
    print(f"  output  : {output_dir}")

    all_results: list[dict] = []
    for theme_id, theme_desc in THEMES:
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir)
            all_results.append(r)
            status = "HIT" if r["hit"] else f"MISS (len={r['final_length']})"
            print(f"  {status}  | brain rounds: {len(r['rounds'])}")
            for round_idx, rnd in enumerate(r["rounds"], 1):
                hit_str = "HIT" if rnd.get("hit") else "miss"
                inner_lens = [a["length"] for a in rnd.get("inner_attempts", [])]
                err = rnd.get("error")
                if err:
                    print(f"    round {round_idx}: ERR ({err})")
                else:
                    print(f"    round {round_idx}: {hit_str}, inner lengths: {inner_lens}")
            print(f"  tokens: in={r['tokens_total_input']}, out={r['tokens_total_output']}")
            for role_name, m in r["tokens_by_role"].items():
                print(f"    [{role_name}] {m['calls']} calls, "
                      f"in={m['input_tokens']}, out={m['output_tokens']}")

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
            "inner_max_retries": INNER_MAX_RETRIES,
            "brain_max_attempts": BRAIN_MAX_ATTEMPTS,
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
    print(f"method_fixed summary")
    print("=" * 60)
    print(f"  hit rate     : {hits}/{total_runs} ({summary['totals']['hit_rate']:.0%})")
    print(f"  tokens total : in={sum_in}, out={sum_out}, sum={sum_in + sum_out}")
    print(f"  per-role     :")
    for role_name, m in grand_by_role.items():
        print(f"    [{role_name}] {m['calls']} calls, "
              f"in={m['input_tokens']}, out={m['output_tokens']}")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
