"""Run all 3 story-task methods sequentially.

Each method is run as a separate Python process (subprocess) so a crash in
one doesn't affect the others. The HTML comparison is auto-generated at the
end of each method's main(), so you can refresh the browser anytime to see
partial progress.

Usage:
    python -m runners.story_task.run_all                # default output dir
    python -m runners.story_task.run_all --exp my_v1    # named experiment

When --exp is given, results go under:
    Execution/story_241/<exp_name>/baseline/
    Execution/story_241/<exp_name>/method_fixed/
    Execution/story_241/<exp_name>/method_brain/
    Execution/story_241/<exp_name>/comparison.html

Without --exp, results go to Execution/story_241/{baseline,method_fixed,
method_brain}/ (re-runs overwrite). Use --exp to keep multiple runs side
by side for comparison.

THEMES / RUNS_PER_THEME / WRITER_CALL_CAP live in each method file — edit
them there if you want to scope down (e.g., 1 theme for a quick look).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


METHODS = [
    "runners.story_task.baseline",
    "runners.story_task.method_fixed",
    "runners.story_task.method_brain",
]


def run_one(module_path: str, env: dict) -> tuple[bool, float]:
    """Run one method as a subprocess. Returns (success, elapsed_seconds).
    `env` is the environment passed to the subprocess (carries STORY_EXP_NAME)."""
    print()
    print("=" * 72)
    print(f"▶  Running: python -m {module_path}")
    print("=" * 72, flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=_PROJECT_ROOT,
        env=env,
        # Inherit stdout/stderr so we see live output, including the method's
        # own per-run logs as it goes.
    )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    print(flush=True)
    print(f"◼  {module_path}: {'OK' if ok else f'FAILED (rc={proc.returncode})'} "
          f"in {elapsed:.1f}s", flush=True)
    return ok, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 3 story-task methods sequentially.",
    )
    parser.add_argument(
        "--exp", default="",
        help="Optional experiment name. Results will be saved under "
             "Execution/story_241/<exp_name>/. Without --exp, results go to "
             "Execution/story_241/{baseline,method_fixed,method_brain}/ "
             "and re-runs overwrite.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # Pass STORY_EXP_NAME to subprocesses so each method + html.py pick up
    # the same output directory prefix.
    env = os.environ.copy()
    if args.exp:
        env["STORY_EXP_NAME"] = args.exp

    started = datetime.now()
    print(f"=== run_all started {started.isoformat(timespec='seconds')} ===")
    if args.exp:
        print(f"Experiment name: {args.exp}")
        print(f"Output dir: Execution/story_241/{args.exp}/")
    else:
        print("No --exp given; results go to default (story_241/<method>/).")
    print(f"Methods to run (in order): {len(METHODS)}")
    for m in METHODS:
        print(f"  - {m}")

    statuses: list[tuple[str, bool, float]] = []
    for module_path in METHODS:
        ok, elapsed = run_one(module_path, env)
        statuses.append((module_path, ok, elapsed))
        # We do NOT abort on failure — the rest of the methods still run.

    finished = datetime.now()
    total_elapsed = (finished - started).total_seconds()

    print()
    print("=" * 72)
    print("run_all summary")
    print("=" * 72)
    for module_path, ok, elapsed in statuses:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {module_path:<40}  {elapsed:>7.1f}s")
    print()
    print(f"Total elapsed: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")

    html_path = (
        os.path.join(_PROJECT_ROOT, "Execution", "story_241", args.exp, "comparison.html")
        if args.exp
        else os.path.join(_PROJECT_ROOT, "Execution", "story_241", "comparison.html")
    )
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
