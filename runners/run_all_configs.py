"""Run all 4 ablation configs sequentially on the same case list.

This is the one-shot driver for the c0/c1/c2/c3 ablation study. It:
  1. Optionally re-samples the case list (or accepts an existing one)
  2. For each config (c0_baseline, c1_judge, c2_planspec, c3_codespec):
     a. Materializes those cases into Execution/<prefix>_<config>/
     b. Runs the agent with the chosen pipeline config
  3. Prints a side-by-side summary at the end

Each config produces its own experiment dir, so dataset.html / per-case
JSON outputs can be inspected and compared independently. The case list is
shared across all 4 dirs so per-case status is directly comparable.

Failure of one config is non-fatal — the remaining configs still run, and
the summary at the end shows which configs succeeded.

Usage:
    # First time: sample the case list AND run all 4 configs
    python -m runners.run_all_configs \\
        --src new_baseline_allpass \\
        --failed 36 --passed 14 \\
        --exp-prefix exp50 \\
        --workers 4 --batch-size 4

    # Reuse an existing case list (skip re-sampling)
    python -m runners.run_all_configs \\
        --case-list exp_50_case_list.json \\
        --exp-prefix exp50

    # Pick a subset of configs (e.g. just c0 and c2)
    python -m runners.run_all_configs \\
        --case-list exp_50_case_list.json \\
        --exp-prefix exp50 \\
        --configs c0_baseline c2_planspec
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP

REPORT_FILENAME = "mbpp_exp_final_results.json"
ALL_CONFIGS = ["c0_baseline", "c1_judge", "c2_planspec", "c3_codespec"]


def _run(cmd: list[str]) -> int:
    """Run a subprocess command, stream output to console, return its exit code."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def _maybe_sample(args) -> str:
    """If --case-list was given, return it. Otherwise call sample_cases.py
    to produce one and return the new path."""
    if args.case_list:
        if not os.path.exists(args.case_list):
            print(f"[error] --case-list {args.case_list} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Reusing case list: {args.case_list}")
        return args.case_list

    if not args.src:
        print("[error] need either --case-list or --src to sample from",
              file=sys.stderr)
        sys.exit(1)

    out = args.case_list_out or f"{args.exp_prefix}_case_list.json"
    cmd = [
        sys.executable, "-m", "runners.sample_cases",
        "--src", args.src,
        "--failed", str(args.failed),
        "--passed", str(args.passed),
        "--output", out,
        "--seed", str(args.seed),
    ]
    rc = _run(cmd)
    if rc != 0:
        print(f"[error] sample_cases failed (rc={rc})", file=sys.stderr)
        sys.exit(rc)
    return out


def _setup_one(exp_name: str, case_list: str, split: str) -> int:
    cmd = [
        sys.executable, "-m", "runners.mbpp_task", "setup",
        "--exp", exp_name,
        "--case-list", case_list,
        "--split", split,
    ]
    return _run(cmd)


def _run_one_config(exp_name: str, config: str, workers: int, batch_size: int,
                    skip_existing: bool = False) -> int:
    cmd = [
        sys.executable, "-m", "runners.mbpp_task", "run",
        "--exp", exp_name,
        "--config", config,
        "--workers", str(workers),
        "--batch-size", str(batch_size),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    return _run(cmd)


def _read_report_summary(exp_name: str, root: str) -> dict | None:
    path = os.path.join(root, exp_name, REPORT_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _print_final_summary(exp_prefix: str, configs: list[str], root: str) -> None:
    print("\n" + "=" * 72)
    print(f"All-configs run summary (exp_prefix={exp_prefix})")
    print("=" * 72)
    header = f"{'config':<14} {'total':>5} {'passed':>7} {'failed':>7} {'crashed':>8} {'pass@1':>8}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        exp_name = f"{exp_prefix}_{cfg}"
        report = _read_report_summary(exp_name, root)
        if report is None:
            print(f"{cfg:<14} (no report — config did not finish)")
            continue
        t = report.get("totals", {})
        total = t.get("total", 0)
        passed = t.get("passed", 0)
        failed = t.get("failed", 0)
        crashed = t.get("crashed", 0)
        rate = (passed / total * 100) if total else 0.0
        print(f"{cfg:<14} {total:>5} {passed:>7} {failed:>7} {crashed:>8} {rate:>7.2f}%")
    print("=" * 72)
    print(f"\nIndividual reports: {root}/{exp_prefix}_<config>/{REPORT_FILENAME}")
    print(f"Individual HTML:    {root}/{exp_prefix}_<config>/dataset.html")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run multiple ablation configs sequentially on the same case list."
    )
    # case-list source (mutually-ish exclusive)
    p.add_argument("--case-list", default=None,
                   help="reuse an existing case list JSON; if omitted, --src is required")
    p.add_argument("--src", default=None,
                   help="source experiment to sample from (used if --case-list not given)")
    p.add_argument("--failed", type=int, default=36,
                   help="number of failed cases to sample (used with --src)")
    p.add_argument("--passed", type=int, default=14,
                   help="number of random passed cases to sample (used with --src)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for sampling")
    p.add_argument("--case-list-out", default=None,
                   help="where to write the new case list (default: <prefix>_case_list.json)")

    # experiment naming
    p.add_argument("--exp-prefix", required=True,
                   help="experiment dir prefix; each config goes to "
                        "Execution/<prefix>_<config>/")

    # which configs to run
    p.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                   choices=ALL_CONFIGS,
                   help="subset of configs to run (default: all 4)")

    # runner knobs (same as mbpp_task run)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--split", default="test")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cases whose working_memory.json already exists")
    p.add_argument("--root", default=WORKSHOP)

    # control
    p.add_argument("--skip-setup", action="store_true",
                   help="don't re-materialize cases (assume dirs already populated)")
    args = p.parse_args()

    started = datetime.datetime.now()
    print(f"=== run_all_configs started {started.isoformat(timespec='seconds')} ===")
    print(f"  configs:    {args.configs}")
    print(f"  prefix:     {args.exp_prefix}")
    print(f"  workers:    {args.workers}, batch_size: {args.batch_size}")
    print(f"  split:      {args.split}")

    # 1. Get / build the case list
    case_list = _maybe_sample(args)

    # 2. Run each config
    statuses: dict[str, str] = {}
    for cfg in args.configs:
        exp_name = f"{args.exp_prefix}_{cfg}"
        print(f"\n{'#' * 72}")
        print(f"# CONFIG: {cfg}  →  Execution/{exp_name}/")
        print(f"{'#' * 72}")

        if not args.skip_setup:
            rc = _setup_one(exp_name, case_list, args.split)
            if rc != 0:
                print(f"[error] setup failed for {cfg} (rc={rc}); skipping run")
                statuses[cfg] = "setup-failed"
                continue

        rc = _run_one_config(exp_name, cfg, args.workers, args.batch_size,
                             skip_existing=args.skip_existing)
        statuses[cfg] = "ok" if rc == 0 else f"run-failed(rc={rc})"

    # 3. Final summary
    _print_final_summary(args.exp_prefix, args.configs, args.root)

    finished = datetime.datetime.now()
    elapsed = (finished - started).total_seconds()
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    print(f"Per-config status: {statuses}")


if __name__ == "__main__":
    main()
