"""Baseline story task: writer LLMNode + length-feedback retry loop.

No engine.run_task involvement — this is the simplest possible "agent"
configuration: ONE LLMNode (gpt-4o-mini, system_prompt = writer PROMPT)
called repeatedly with feedback until len == 241 or retry budget exhausted.

Per task spec:
  - 4 fixed English themes (locked across baseline / Track A / Track B)
  - Strict 241 characters, zero tolerance
  - 3 runs per theme to characterize variance
  - Max 10 retries per run
  - Worker model locked to gpt-4o-mini

Outputs (under WORKSHOP/story_241/baseline/):
  <theme_id>/run_<N>.txt   — final story text (last attempt, hit or miss)
  summary.json             — full per-run metrics + per-role token aggregation

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
MAX_RETRIES = 10
RUNS_PER_THEME = 3
OUTPUT_SUBDIR = os.path.join("story_241", "baseline")


# ---------------------------------------------------------------------------
# One run = up to MAX_RETRIES attempts on a single (theme, run_idx)
# ---------------------------------------------------------------------------

def run_one(theme_id: str, theme_desc: str, run_idx: int, output_dir: str) -> dict:
    """One attempt sequence on one theme. Returns metrics dict."""
    metrics = MetricsTracker()
    cfg = ROLE_CONFIGS["writer"]

    feedback = ""
    final_story = ""
    final_len = 0
    hit = False

    for attempt_idx in range(1, MAX_RETRIES + 1):
        # Fresh LLMNode each attempt: no message-history accumulation across
        # retries (we re-build the prompt from scratch with the new feedback
        # baked in via writer_role.build_input).
        writer = LLMNode(
            system_prompt=writer_role.PROMPT,
            role="writer",
            max_steps=cfg["max_steps"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            metrics_tracker=metrics,
        )
        story = writer_role.run_writer(writer, theme_desc, feedback)
        actual_len = len(story)

        # Always remember the latest attempt — even if all 10 miss, we save
        # the last one (so you can read what the model gave up on).
        final_story = story
        final_len = actual_len

        if actual_len == TARGET_LEN:
            hit = True
            break

        feedback = writer_role.build_length_feedback(actual_len, TARGET_LEN)

    # Save the final story text for inspection.
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
        "final_length": final_len,
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
    print(f"=== Baseline story task started {started_at} ===")
    print(f"  themes : {len(THEMES)}")
    print(f"  runs   : {RUNS_PER_THEME} per theme")
    print(f"  retries: max {MAX_RETRIES}")
    print(f"  target : exactly {TARGET_LEN} characters")
    print(f"  output : {output_dir}")

    all_results: list[dict] = []
    for theme_id, theme_desc in THEMES:
        for run_idx in range(1, RUNS_PER_THEME + 1):
            print(f"\n--- {theme_id} run {run_idx}/{RUNS_PER_THEME} ---")
            r = run_one(theme_id, theme_desc, run_idx, output_dir)
            all_results.append(r)
            status = "HIT" if r["hit"] else f"MISS (len={r['final_length']})"
            print(f"  {status}")
            print(f"  tokens: in={r['tokens_total_input']}, out={r['tokens_total_output']}")
            for role_name, m in r["tokens_by_role"].items():
                print(f"    [{role_name}] {m['calls']} calls, "
                      f"in={m['input_tokens']}, out={m['output_tokens']}")

    # Aggregate
    total_runs = len(all_results)
    hits = sum(1 for r in all_results if r["hit"])
    sum_in = sum(r["tokens_total_input"] for r in all_results)
    sum_out = sum(r["tokens_total_output"] for r in all_results)

    # Per-role aggregation across all runs (same role names sum together).
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
            "max_retries": MAX_RETRIES,
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
    print(f"Baseline summary")
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
