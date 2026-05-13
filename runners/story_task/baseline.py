"""Baseline story task: writer LLMNode + length-feedback retry loop.

No engine, no brain — this is the simplest "agent" configuration: ONE LLMNode
(gpt-4o-mini, system_prompt = writer PROMPT) called repeatedly with feedback
until len == 241 or the writer-call budget is exhausted.

Per task spec (locked across baseline / method_fixed / method_brain):
  - 4 fixed English themes
  - Strict 241 characters, zero tolerance
  - 4 runs per theme to characterize variance
  - Max 8 writer calls per run (matches method_fixed inner budget,
    method_brain WRITER_CALL_CAP)
  - Worker model locked to gpt-4o-mini

Outputs (under WORKSHOP/story_241/baseline/):
  <theme_id>/run_<N>.txt   — final story text (HIT or last MISS)
  summary.json             — per-run trajectory (every writer + verify step
                              with full input/output) + per-role token totals

Usage:
  python -m runners.story_task.baseline
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
from tool_pool.text_utils import length_checker


# ---------------------------------------------------------------------------
# Task spec — locked across all three story-task methods
# ---------------------------------------------------------------------------

THEMES: list[tuple[str, str]] = [
    # Currently scoped to 1 theme for smoke-testing. Uncomment the rest for
    # the full 4-theme experiment.
    ("mountain_school",       "The lone teacher at a remote mountain school"),
    # ("time_displaced_store",  "A convenience store displaced in time"),
    # ("photo_studio_last_day", "The final day of an old photo studio"),
    # ("rainy_night_bus",       "The last bus on a rainy night"),
]
TARGET_LEN = 241
WRITER_CALL_CAP = 8                  # max writer calls per (theme, run)
RUNS_PER_THEME = 4

# If STORY_EXP_NAME is set (typically by run_all.py --exp), put results
# under story_241/<exp_name>/<method>/. Otherwise default to story_241/<method>/.
_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
OUTPUT_SUBDIR = (
    os.path.join("story_241", _EXP_NAME, "baseline") if _EXP_NAME
    else os.path.join("story_241", "baseline")
)


# ---------------------------------------------------------------------------
# One run = up to WRITER_CALL_CAP attempts on a single (theme, run_idx)
# ---------------------------------------------------------------------------

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str) -> dict:
    """One attempt sequence on one theme. Returns metrics dict with trajectory."""
    metrics = MetricsTracker()
    cfg = ROLE_CONFIGS["writer"]

    trajectory: list[dict] = []
    step = 0

    feedback = ""
    previous_attempt = ""    # the full text of the last attempt — passed to next
    final_story = ""
    final_length = 0
    hit = False

    for attempt_idx in range(1, WRITER_CALL_CAP + 1):
        # Fresh LLMNode each attempt: no message-history accumulation across
        # retries (we re-build the prompt from scratch with the new feedback
        # + previous_attempt baked in via writer_role.build_input).
        writer = LLMNode(
            system_prompt=writer_role.PROMPT,
            role="writer",
            max_steps=cfg["max_steps"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            metrics_tracker=metrics,
        )
        # Build the actual user message so we can record it as the "input"
        # (writer_role.build_input is the same fn run_writer uses internally).
        user_input = writer_role.build_input(
            theme=theme_desc, guidance="",
            feedback=feedback, previous_attempt=previous_attempt,
        )

        n0 = len(metrics.calls)
        writer.reset_message()
        response = writer.run(user_input)
        story = (response.get("text") or "").strip()
        new_calls = metrics.calls[n0:]
        writer_tokens = {
            "in": sum(c.input_tokens for c in new_calls),
            "out": sum(c.output_tokens for c in new_calls),
        }

        step += 1
        trajectory.append({
            "step": step,
            "role": "writer",
            "purpose": f"attempt #{attempt_idx}",
            "input": user_input,
            "output": story,
            "tokens": writer_tokens,
        })

        # Host-side verify with length_checker (Python, no LLM cost)
        verify = length_checker(story, target=TARGET_LEN)
        step += 1
        trajectory.append({
            "step": step,
            "role": "length_checker",
            "purpose": f"verify attempt #{attempt_idx}",
            "input": {"text": story, "target": TARGET_LEN},
            "output": verify,
            "tokens": {"in": 0, "out": 0},
        })

        # Per-attempt live log
        diff_str = f"(off {verify['diff']:+d})" if not verify["hit"] else "        "
        status = "HIT" if verify["hit"] else "MISS"
        print(f"  attempt {attempt_idx}: in={writer_tokens['in']:>5}  "
              f"out={writer_tokens['out']:>4}  len={verify['length']:>3}  {diff_str}  {status}",
              flush=True)

        # Always remember the latest attempt — even if all retries miss, we
        # save the last one (so you can read what the model gave up on).
        final_story = story
        final_length = verify["length"]

        if verify["hit"]:
            hit = True
            break

        # Feed BOTH the length-diff feedback AND the full previous story
        # text into the next attempt, so writer can revise directly.
        feedback = (
            f"Previous attempt was {verify['length']} characters, "
            f"{verify['delta_text']}. Adjust to exactly {TARGET_LEN}."
        )
        previous_attempt = story

    # Save the final story for inspection
    theme_dir = os.path.join(output_dir, theme_id)
    os.makedirs(theme_dir, exist_ok=True)
    with open(os.path.join(theme_dir, f"run_{run_idx}.txt"), "w", encoding="utf-8") as f:
        f.write(final_story)

    by_role = metrics.by_role()
    total_in = sum(r["input_tokens"] for r in by_role.values())
    total_out = sum(r["output_tokens"] for r in by_role.values())

    return {
        "theme_id": theme_id,
        "theme_desc": theme_desc,
        "run_idx": run_idx,
        "hit": hit,
        "final_length": final_length,
        "writer_calls_used": sum(1 for s in trajectory if s["role"] == "writer"),
        "trajectory": trajectory,
        "tokens_by_role": by_role,
        "tokens_total_input": total_in,
        "tokens_total_output": total_out,
    }


# ---------------------------------------------------------------------------
# Main: iterate themes × runs, write summary
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.join(WORKSHOP, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"=== baseline (story task) started {started_at} ===")
    print(f"  themes : {len(THEMES)}")
    print(f"  runs   : {RUNS_PER_THEME} per theme")
    print(f"  budget : {WRITER_CALL_CAP} writer calls per run")
    print(f"  target : exactly {TARGET_LEN} characters")
    print(f"  output : {output_dir}")

    all_results: list[dict] = []
    for theme_id, theme_desc in THEMES:
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir)
            all_results.append(r)
            status = "HIT" if r["hit"] else f"MISS (len={r['final_length']})"
            print(f"  → run result: {status}  | writer calls used: {r['writer_calls_used']}/{WRITER_CALL_CAP}"
                  f"  | tokens: in={r['tokens_total_input']}, out={r['tokens_total_output']}")

    # Aggregate
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
        "method": "baseline",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "target_len": TARGET_LEN,
            "writer_call_cap": WRITER_CALL_CAP,
            "runs_per_theme": RUNS_PER_THEME,
            "themes": [{"id": t[0], "desc": t[1]} for t in THEMES],
            "writer_role_config": ROLE_CONFIGS["writer"],
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
    print(f"baseline summary")
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
