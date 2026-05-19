"""Run the method_brain vs method_brain_code A/B comparison experiment.

Research question: does directly emitting Python code vs JSON DSL change
brain's effective performance? Outer loop / writer model / writer cap /
cross-run memory shape / per-cycle feedback density are all locked
identical between the two methods (method_brain.py was modified to also
show an execution trace, so brain sees the same info density as
method_brain_code).

What's left as the lone independent variable: the OUTPUT MEDIUM brain
emits each cycle. method_brain emits a JSON DSL workflow → host parses →
host's DSL interpreter walks it. method_brain_code emits a Python `solve`
function → host execs → host invokes it directly.

Sample size: 4 themes × 8 runs/theme = 32 runs/method, 64 total. At
binary pass/fail granularity, SE ≈ 9% on overall pass rate; cycle-level
(160 attempts/method, no early exit) tightens to ≈ 4%. The validated
pass rate (filters writer-#1 cold-start luck) is the most apples-to-
apples comparison.

What this script does:

  1. Launches each method as a subprocess with STORY_EXP_NAME=brain_vs_code,
     STORY_RUNS_PER_THEME=8. Outputs land at
     story_241/brain_vs_code/method_brain/ and
     story_241/brain_vs_code/method_brain_code/.
  2. Reads both summary.json files.
  3. Prints + saves a text comparison report:
       - 3-tier pass rates (run / cycle / validated), per-theme + overall
       - error class distribution (DSL parse vs Python parse vs runtime
         vs budget)
       - token cost (brain in/out, writer in/out, total)
       - strategy evolution snapshots: for each theme, prints the
         strategy_notes + plan + final_length at 4 corners — (run 1,
         cycle 1), (run 1, last cycle), (last run, cycle 1), (last run,
         last cycle). Lets you eyeball whether either method's brain
         converges or explores within/across runs.
  4. Calls html.py to render the side-by-side comparison.html (uses the
     existing per-theme/per-run UX; method_brain_code plans show as
     Python <pre>, method_brain plans as the JSON workflow tree).

Usage:
    python -m runners.story_task.run_brain_code_comparison

Skip a method if its summary already exists:
    python -m runners.story_task.run_brain_code_comparison --skip-existing

Override scope (e.g. for smoke test):
    python -m runners.story_task.run_brain_code_comparison --themes mountain_school --runs 1
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

from config import WORKSHOP
from runners.story_task._metrics import per_theme_counts, overall_counts


DEFAULT_EXP_NAME = "brain_vs_code"
METHODS = [
    ("method_brain",      "runners.story_task.method_brain"),
    ("method_brain_code", "runners.story_task.method_brain_code"),
]
AVAILABLE_THEMES = [
    "mountain_school",
    "time_displaced_store",
    "photo_studio_last_day",
    "rainy_night_bus",
]
DEFAULT_RUNS = 8


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------

def run_one(module_path: str, env: dict) -> tuple[bool, float]:
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


# ---------------------------------------------------------------------------
# Summary loading
# ---------------------------------------------------------------------------

def load_summary(method_name: str, exp_name: str) -> dict | None:
    path = os.path.join(WORKSHOP, "story_241", exp_name, method_name, "summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _fmt_rate(num: int, den: int) -> str:
    if not den:
        return "—"
    return f"{num}/{den} ({num/den:.0%})"


def _error_distribution(summary: dict) -> dict[str, int]:
    """Count cycle-level error kinds across all runs.

    Both methods record errors at cycle granularity in cycles[*].error.
    method_brain has: plan_unparseable, workflow_error: ...
    method_brain_code has: code_unparseable, syntax_error: ..., no_solve_function,
                            runtime_error: ..., budget_exceeded: ...
    We bucket by the prefix before the first colon for grouping.
    """
    counts: dict[str, int] = {}
    for r in summary.get("results", []) or []:
        for c in r.get("cycles", []) or []:
            err = c.get("error")
            if not err:
                continue
            head = err.split(":", 1)[0].strip()
            counts[head] = counts.get(head, 0) + 1
    return counts


def _token_totals(summary: dict) -> dict[str, int]:
    by_role = summary.get("totals", {}).get("tokens_by_role", {})
    out = {"brain_in": 0, "brain_out": 0, "writer_in": 0, "writer_out": 0}
    for role, m in by_role.items():
        if role == "brain":
            out["brain_in"] += m.get("input_tokens", 0)
            out["brain_out"] += m.get("output_tokens", 0)
        elif role == "writer":
            out["writer_in"] += m.get("input_tokens", 0)
            out["writer_out"] += m.get("output_tokens", 0)
    out["total"] = sum(out.values())
    return out


def _format_plan(cycle: dict, method_name: str) -> str:
    """Render the cycle's plan as a short multi-line string for the
    strategy evolution snapshot. method_brain → workflow JSON;
    method_brain_code → Python source."""
    if method_name == "method_brain":
        wf = cycle.get("workflow_json")
        if wf is None:
            return "(unparseable / no workflow)"
        try:
            return json.dumps(wf, indent=2, ensure_ascii=False)
        except Exception:
            return str(wf)
    if method_name == "method_brain_code":
        code = cycle.get("code")
        return code or "(unparseable / no code)"
    return "(unknown method)"


def _strategy_corners(results: list[dict], theme_id: str) -> list[tuple[str, dict, dict]]:
    """For one theme, pick 4 "corner" (run, cycle) pairs to show strategy
    evolution: (run 1, cycle 1), (run 1, last), (last run, cycle 1),
    (last run, last). Returns list of (label, run_record, cycle_record).
    Returns fewer if the theme has <2 runs."""
    theme_runs = [r for r in results if r.get("theme_id") == theme_id]
    if not theme_runs:
        return []
    first_run = theme_runs[0]
    last_run = theme_runs[-1]
    corners: list[tuple[str, dict, dict]] = []

    def pick(run_rec: dict, which: str, label: str):
        cycles = run_rec.get("cycles") or []
        if not cycles:
            return
        cycle = cycles[0] if which == "first" else cycles[-1]
        corners.append((label, run_rec, cycle))

    pick(first_run, "first", f"run {first_run.get('run_idx', '?')} / cycle "
                              f"{(first_run.get('cycles') or [{}])[0].get('cycle', '?')} (FIRST run, FIRST cycle)")
    if len(first_run.get("cycles") or []) > 1:
        pick(first_run, "last", f"run {first_run.get('run_idx', '?')} / cycle "
                                 f"{(first_run.get('cycles') or [{}])[-1].get('cycle', '?')} (FIRST run, LAST cycle)")
    if last_run is not first_run:
        pick(last_run, "first", f"run {last_run.get('run_idx', '?')} / cycle "
                                 f"{(last_run.get('cycles') or [{}])[0].get('cycle', '?')} (LAST run, FIRST cycle)")
        if len(last_run.get("cycles") or []) > 1:
            pick(last_run, "last", f"run {last_run.get('run_idx', '?')} / cycle "
                                    f"{(last_run.get('cycles') or [{}])[-1].get('cycle', '?')} (LAST run, LAST cycle)")
    return corners


def build_comparison_report(brain_summary: dict, code_summary: dict) -> str:
    """Build the human-readable text comparison report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("method_brain vs method_brain_code — A/B comparison")
    lines.append("=" * 78)
    lines.append(f"rendered_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # ----- Pass rates -----
    lines.append("## Pass rates")
    lines.append("")
    for tier_name, getter in [
        ("RUN-level (pass@8 cycles, primary metric)",
            lambda o: (o["runs_hits"], o["runs_total"])),
        ("CYCLE-level (each cycle counts; n is variable due to early exit)",
            lambda o: (o["cycle_hits"], o["cycle_total"])),
        ("VALIDATED (cycle-level, filtering writer-#1 cold-start luck)",
            lambda o: (o["validated_hits"], o["cycle_total"])),
    ]:
        lines.append(f"### {tier_name}")
        b_over = overall_counts(brain_summary, "method_brain")
        c_over = overall_counts(code_summary, "method_brain_code")
        b_num, b_den = getter(b_over)
        c_num, c_den = getter(c_over)
        lines.append(f"  method_brain      : {_fmt_rate(b_num, b_den)}")
        lines.append(f"  method_brain_code : {_fmt_rate(c_num, c_den)}")
        if b_den and c_den:
            delta = (c_num / c_den) - (b_num / b_den)
            lines.append(f"  Δ (code − brain)  : {delta:+.1%}")
        lines.append("")

    # ----- Per-theme breakdown -----
    lines.append("## Per-theme breakdown (run / cycle / validated)")
    lines.append("")
    b_theme = per_theme_counts(brain_summary, "method_brain")
    c_theme = per_theme_counts(code_summary, "method_brain_code")
    all_themes = sorted(set(b_theme) | set(c_theme))
    header = f"  {'theme':<24} {'method_brain':<24} {'method_brain_code':<24}"
    lines.append(header)
    lines.append(f"  {'-' * 22:<24} {'-' * 22:<24} {'-' * 22:<24}")
    for tid in all_themes:
        b = b_theme.get(tid, {"runs_hits": 0, "runs_total": 0,
                              "cycle_hits": 0, "cycle_total": 0,
                              "validated_hits": 0})
        c = c_theme.get(tid, {"runs_hits": 0, "runs_total": 0,
                              "cycle_hits": 0, "cycle_total": 0,
                              "validated_hits": 0})
        b_str = (f"run {_fmt_rate(b['runs_hits'], b['runs_total'])}"
                 f" | cyc {_fmt_rate(b['cycle_hits'], b['cycle_total'])}"
                 f" | val {_fmt_rate(b['validated_hits'], b['cycle_total'])}")
        c_str = (f"run {_fmt_rate(c['runs_hits'], c['runs_total'])}"
                 f" | cyc {_fmt_rate(c['cycle_hits'], c['cycle_total'])}"
                 f" | val {_fmt_rate(c['validated_hits'], c['cycle_total'])}")
        lines.append(f"  {tid:<24}")
        lines.append(f"    method_brain      : {b_str}")
        lines.append(f"    method_brain_code : {c_str}")
        lines.append("")

    # ----- Error distribution -----
    lines.append("## Error distribution (cycles with non-null error)")
    lines.append("")
    b_err = _error_distribution(brain_summary)
    c_err = _error_distribution(code_summary)
    if not b_err and not c_err:
        lines.append("  (no cycle-level errors recorded)")
    else:
        all_kinds = sorted(set(b_err) | set(c_err))
        lines.append(f"  {'error_kind':<32} {'method_brain':>16} {'method_brain_code':>20}")
        for k in all_kinds:
            lines.append(f"  {k:<32} {b_err.get(k, 0):>16} {c_err.get(k, 0):>20}")
    lines.append("")

    # ----- Token cost -----
    lines.append("## Token cost (sum across all runs)")
    lines.append("")
    b_tok = _token_totals(brain_summary)
    c_tok = _token_totals(code_summary)
    lines.append(f"  {'role':<20} {'method_brain':>16} {'method_brain_code':>20} {'Δ':>14}")
    for k, label in [("brain_in", "brain in"),
                     ("brain_out", "brain out"),
                     ("writer_in", "writer in"),
                     ("writer_out", "writer out"),
                     ("total", "TOTAL")]:
        b_v = b_tok.get(k, 0)
        c_v = c_tok.get(k, 0)
        delta = c_v - b_v
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {label:<20} {b_v:>16,} {c_v:>20,} {sign}{delta:>13,}")
    lines.append("")

    # ----- Strategy evolution snapshots -----
    lines.append("## Strategy evolution snapshots")
    lines.append("")
    lines.append("Four 'corner' cycles per theme — first-run-first-cycle,")
    lines.append("first-run-last-cycle, last-run-first-cycle, last-run-last-cycle.")
    lines.append("Compare how strategy_notes (and the plan itself) evolves both")
    lines.append("WITHIN a run (cycle 1 → cycle last) and ACROSS runs (run 1 → last)")
    lines.append("for each method.")
    lines.append("")
    for tid in all_themes:
        lines.append(f"### {tid}")
        lines.append("")
        for method_name, summary in [("method_brain", brain_summary),
                                     ("method_brain_code", code_summary)]:
            lines.append(f"  --- {method_name} ---")
            corners = _strategy_corners(summary.get("results", []) or [], tid)
            if not corners:
                lines.append("    (no runs for this theme)")
                lines.append("")
                continue
            for label, run_rec, cycle in corners:
                fl = cycle.get("final_length", "?")
                hit = "Pass" if cycle.get("hit") else "Fail"
                sn = (cycle.get("strategy_notes") or "").strip() or "(empty)"
                # Wrap strategy_notes for readability
                if len(sn) > 200:
                    sn = sn[:197] + "..."
                lines.append(f"    [{label}] → length {fl} ({hit})")
                lines.append(f"      strategy_notes: {sn}")
                plan_snippet = _format_plan(cycle, method_name)
                # Indent and truncate the plan to keep the report scannable
                plan_lines = plan_snippet.splitlines()
                if len(plan_lines) > 12:
                    plan_lines = plan_lines[:11] + [f"      ... ({len(plan_snippet.splitlines()) - 11} more lines, see summary.json)"]
                for pl in plan_lines:
                    lines.append(f"      | {pl}")
                lines.append("")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=DEFAULT_EXP_NAME,
                        help=f"experiment name; outputs land at "
                             f"story_241/<exp>/method_brain[_code]/ "
                             f"(default {DEFAULT_EXP_NAME!r})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"runs per theme (default {DEFAULT_RUNS})")
    parser.add_argument("--themes", default=",".join(AVAILABLE_THEMES),
                        help="comma-separated theme ids (default: all 4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip a method if its summary.json already exists")
    parser.add_argument("--report-only", action="store_true",
                        help="skip subprocesses; just read existing summaries "
                             "and (re)generate comparison.txt + comparison.html")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY") and not args.report_only:
        print("[error] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    exp_name = args.exp
    base_dir = os.path.join(WORKSHOP, "story_241", exp_name)
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 72)
    print(f"method_brain vs method_brain_code comparison")
    print(f"  exp        : {exp_name}")
    print(f"  themes     : {args.themes}")
    print(f"  runs/theme : {args.runs}")
    print(f"  base_dir   : {base_dir}")
    print("=" * 72, flush=True)

    timings: dict[str, float] = {}
    for method_name, module_path in METHODS:
        method_dir = os.path.join(base_dir, method_name)
        method_summary = os.path.join(method_dir, "summary.json")

        if args.report_only:
            print(f"[skip] {method_name}: report-only mode")
            continue
        if args.skip_existing and os.path.exists(method_summary):
            print(f"[skip] {method_name}: summary.json already exists "
                  f"at {method_summary}")
            continue

        env = os.environ.copy()
        env["STORY_EXP_NAME"] = exp_name
        env["STORY_RUNS_PER_THEME"] = str(args.runs)
        env["STORY_THEMES"] = args.themes
        ok, elapsed = run_one(module_path, env)
        timings[method_name] = elapsed
        if not ok:
            print(f"[error] {module_path} failed; aborting", file=sys.stderr)
            sys.exit(1)

    # ----- Load both summaries -----
    print()
    print("=" * 72)
    print("Loading summaries + building comparison report")
    print("=" * 72, flush=True)
    brain_summary = load_summary("method_brain", exp_name)
    code_summary = load_summary("method_brain_code", exp_name)
    if brain_summary is None or code_summary is None:
        missing = [n for n, s in [("method_brain", brain_summary),
                                  ("method_brain_code", code_summary)] if s is None]
        print(f"[error] missing summary(ies): {missing}", file=sys.stderr)
        sys.exit(1)

    report = build_comparison_report(brain_summary, code_summary)
    print(report)
    report_path = os.path.join(base_dir, "comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[saved] {report_path}")

    # ----- Render HTML -----
    print()
    print("Rendering comparison.html ...")
    html_env = os.environ.copy()
    html_env["STORY_EXP_NAME"] = exp_name
    proc = subprocess.run(
        [sys.executable, "-m", "runners.story_task.html"],
        cwd=_PROJECT_ROOT,
        env=html_env,
    )
    if proc.returncode != 0:
        print(f"[warn] html.py exited with rc={proc.returncode}; "
              f"comparison.txt is still saved")

    if timings:
        print()
        print("Wall-clock timings:")
        for m, t in timings.items():
            print(f"  {m:<22} {t:>8.1f}s")
        print(f"  {'TOTAL':<22} {sum(timings.values()):>8.1f}s")


if __name__ == "__main__":
    main()
