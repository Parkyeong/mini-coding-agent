"""Run all 3 story-task methods sequentially.

Each method is run as a separate Python process (subprocess) so that a crash
in one doesn't affect the others. The HTML comparison is auto-generated at
the end of each method's main(), so you can refresh the browser anytime to
see partial progress.

Usage:
    python -m runners.story_task.run_all

Reads THEMES / RUNS_PER_THEME directly from each method file — change them
there if you want to scope down (e.g., 1 theme for a quick look).
"""

from __future__ import annotations

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


def run_one(module_path: str) -> tuple[bool, float]:
    """Run one method as a subprocess. Returns (success, elapsed_seconds)."""
    print()
    print("=" * 72)
    print(f"▶  Running: python -m {module_path}")
    print("=" * 72, flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=_PROJECT_ROOT,
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
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    started = datetime.now()
    print(f"=== run_all started {started.isoformat(timespec='seconds')} ===")
    print(f"Methods to run (in order): {len(METHODS)}")
    for m in METHODS:
        print(f"  - {m}")

    statuses: list[tuple[str, bool, float]] = []
    for module_path in METHODS:
        ok, elapsed = run_one(module_path)
        statuses.append((module_path, ok, elapsed))
        # We do NOT abort on failure — the rest of the methods still run.
        # The HTML auto-render is fail-soft inside each method, so partial
        # data is fine.

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
    print(f"HTML: {os.path.join(_PROJECT_ROOT, 'Execution', 'story_241', 'comparison.html')}")


if __name__ == "__main__":
    main()
