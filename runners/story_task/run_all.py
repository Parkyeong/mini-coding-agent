"""Run all 3 story-task methods sequentially.

Each method is run as a separate Python process (subprocess) so a crash in
one doesn't affect the others. The HTML comparison is auto-generated at the
end of each method's main(), so you can refresh the browser anytime to see
partial progress.

Usage:
    # default: all 4 themes, 4 runs each, results overwrite story_241/<method>/
    python -m runners.story_task.run_all

    # named experiment (results go under story_241/<exp>/)
    python -m runners.story_task.run_all --exp my_v1

    # scope down: only 1 theme, 4 runs each
    python -m runners.story_task.run_all --themes mountain_school --runs 4

    # multiple specific themes
    python -m runners.story_task.run_all --themes mountain_school,rainy_night_bus --runs 2

    # combine all knobs
    python -m runners.story_task.run_all --exp test_v3 --themes mountain_school --runs 2

Parameter control flows: run_all CLI args → env vars (STORY_*) → each method
file reads env vars at module load → applied to THEMES / RUNS_PER_THEME /
OUTPUT_SUBDIR. Single source of truth = run_all.
"""

from __future__ import annotations

import argparse
import json
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

AVAILABLE_THEMES = [
    "mountain_school",
    "time_displaced_store",
    "photo_studio_last_day",
    "rainy_night_bus",
]


def run_one(module_path: str, env: dict) -> tuple[bool, float]:
    """Run one method as a subprocess. Returns (success, elapsed_seconds)."""
    print()
    print("=" * 72)
    print(f"▶  Running: python -m {module_path}")
    print("=" * 72, flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=_PROJECT_ROOT,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    print(flush=True)
    print(f"◼  {module_path}: {'OK' if ok else f'FAILED (rc={proc.returncode})'} "
          f"in {elapsed:.1f}s", flush=True)
    return ok, elapsed


def per_theme_summary(summary_path: str, method_name: str) -> dict:
    """Read a method's summary.json and return per-theme counts.

    Each entry is {runs_hits, runs_total, cycle_hits, cycle_total}.
    See _metrics.py for the two-axis rationale (run-level pass@N vs
    cycle-level per-iteration pass rate).
    """
    if not os.path.exists(summary_path):
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        return {}
    from runners.story_task._metrics import per_theme_counts
    return per_theme_counts(s, method_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 3 story-task methods sequentially.",
    )
    parser.add_argument(
        "--exp", default="",
        help="Optional experiment name. Results saved to "
             "Execution/story_241/<exp_name>/. Without --exp, results "
             "overwrite Execution/story_241/<method>/.",
    )
    parser.add_argument(
        "--themes", default="",
        help=f"Comma-separated theme ids to scope down. Empty = all 4. "
             f"Available: {','.join(AVAILABLE_THEMES)}",
    )
    parser.add_argument(
        "--runs", type=int, default=4,
        help="Runs per theme (default: 4)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # Validate --themes (catch typos before launching subprocesses).
    if args.themes:
        requested = [t.strip() for t in args.themes.split(",") if t.strip()]
        unknown = [t for t in requested if t not in AVAILABLE_THEMES]
        if unknown:
            print(f"[error] unknown theme(s): {unknown}", file=sys.stderr)
            print(f"[hint] available: {AVAILABLE_THEMES}", file=sys.stderr)
            sys.exit(2)
        active_themes = requested
    else:
        active_themes = list(AVAILABLE_THEMES)

    # Pass STORY_* env vars to subprocesses.
    env = os.environ.copy()
    if args.exp:
        env["STORY_EXP_NAME"] = args.exp
    if args.themes:
        env["STORY_THEMES"] = args.themes
    env["STORY_RUNS_PER_THEME"] = str(args.runs)

    started = datetime.now()
    print(f"=== run_all started {started.isoformat(timespec='seconds')} ===")
    print(f"  themes  : {len(active_themes)}  ({', '.join(active_themes)})")
    print(f"  runs    : {args.runs} per theme")
    print(f"  total   : {len(active_themes) * args.runs} cases per method × "
          f"{len(METHODS)} methods")
    print(f"  exp     : {args.exp if args.exp else '(none — overwrites default dirs)'}")
    out_root = os.path.join(_PROJECT_ROOT, "Execution", "story_241",
                            args.exp) if args.exp else os.path.join(
        _PROJECT_ROOT, "Execution", "story_241")
    print(f"  output  : {out_root}/")
    print(f"  methods : {len(METHODS)} (in order)")
    for m in METHODS:
        print(f"    - {m}")

    statuses: list[tuple[str, bool, float]] = []
    for module_path in METHODS:
        ok, elapsed = run_one(module_path, env)
        statuses.append((module_path, ok, elapsed))

    finished = datetime.now()
    total_elapsed = (finished - started).total_seconds()

    # ---------- Cross-method per-theme summary ----------
    # Metrics per cell, stacked:
    #   run: pass@N — fraction of runs that ended with a Pass.
    #   cyc: per-cycle pass — fraction of main-loop iterations that
    #        produced a 241-char output (denominator = cycles executed).
    #   val: strategy-validated pass — only for method_brain, since it's
    #        the only method whose cycle internals are designed by the LLM.
    #        baseline (single-attempt cycles) and method_fixed (fixed
    #        pipeline) have trivial / pre-defined "validation" so the
    #        number adds no signal there.
    print()
    print("=" * 78)
    print("Cross-method per-theme accuracy  (run = pass@N, cyc = per-iter; "
          "val = strategy-driven, method_brain only)")
    print("=" * 78)
    col_w = 16
    header = f"  {'theme / metric':<26}"
    for module_path in METHODS:
        method_name = module_path.rsplit(".", 1)[-1]
        header += f"  {method_name:>{col_w}}"
    print(header)
    print("  " + "-" * (26 + (col_w + 2) * len(METHODS)))

    # Collect per-method per-theme stats
    method_themes: dict = {}
    for module_path in METHODS:
        method_name = module_path.rsplit(".", 1)[-1]
        summary_path = os.path.join(out_root, method_name, "summary.json")
        method_themes[method_name] = per_theme_summary(summary_path, method_name)

    def _fmt(num: int, den: int) -> str:
        if den == 0:
            return "(no data)"
        return f"{num}/{den} ({num / den:>4.0%})"

    SHOW_VAL = "method_brain"   # val metric only renders for this method

    for tid in active_themes:
        run_row = f"  {tid:<22} run "
        cyc_row = f"  {'':<22} cyc "
        val_row = f"  {'':<22} val "
        for module_path in METHODS:
            method_name = module_path.rsplit(".", 1)[-1]
            m = method_themes.get(method_name, {}).get(tid)
            if m is None:
                run_row += f"  {'(no data)':>{col_w}}"
                cyc_row += f"  {'(no data)':>{col_w}}"
                val_row += f"  {'—':>{col_w}}"
            else:
                run_row += f"  {_fmt(m['runs_hits'], m['runs_total']):>{col_w}}"
                cyc_row += f"  {_fmt(m['cycle_hits'], m['cycle_total']):>{col_w}}"
                if method_name == SHOW_VAL:
                    val_row += f"  {_fmt(m['validated_hits'], m['cycle_total']):>{col_w}}"
                else:
                    val_row += f"  {'—':>{col_w}}"
        print(run_row)
        print(cyc_row)
        print(val_row)

    # Totals row
    print("  " + "-" * (26 + (col_w + 2) * len(METHODS)))
    tot_run = f"  {'TOTAL':<22} run "
    tot_cyc = f"  {'':<22} cyc "
    tot_val = f"  {'':<22} val "
    for module_path in METHODS:
        method_name = module_path.rsplit(".", 1)[-1]
        all_themes = method_themes.get(method_name, {})
        rh = sum(m["runs_hits"] for m in all_themes.values())
        rt = sum(m["runs_total"] for m in all_themes.values())
        ch = sum(m["cycle_hits"] for m in all_themes.values())
        ct = sum(m["cycle_total"] for m in all_themes.values())
        vh = sum(m["validated_hits"] for m in all_themes.values())
        tot_run += f"  {_fmt(rh, rt):>{col_w}}"
        tot_cyc += f"  {_fmt(ch, ct):>{col_w}}"
        if method_name == SHOW_VAL:
            tot_val += f"  {_fmt(vh, ct):>{col_w}}"
        else:
            tot_val += f"  {'—':>{col_w}}"
    print(tot_run)
    print(tot_cyc)
    print(tot_val)

    # ---------- Per-method run status ----------
    print()
    print("=" * 72)
    print("Per-method run status")
    print("=" * 72)
    for module_path, ok, elapsed in statuses:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {module_path:<40}  {elapsed:>7.1f}s")
    print()
    print(f"Total elapsed: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")

    html_path = os.path.join(out_root, "comparison.html")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
