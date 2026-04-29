"""Sample a deterministic case list for ablation experiments.

Reads an existing experiment's report (e.g. new_baseline) and picks:
  - all (or N) failed cases → the "where can we improve?" set
  - K random PASSED cases → the "did we break anything safe?" control

Output: a JSON list of case_ids (e.g. ["mbpp_0019", "mbpp_0083", ...]) that
multiple ablation experiments can ingest via `mbpp_task setup --case-list ...`,
guaranteeing every config runs the same set of tasks for fair comparison.

Usage:
    python -m runners.sample_cases \\
        --src new_baseline \\
        --failed 36 --passed 14 \\
        --output exp_50_case_list.json \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP

REPORT_FILENAME = "mbpp_exp_final_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a deterministic case list from an existing exp's report."
    )
    parser.add_argument("--src", required=True,
                        help="source experiment name under Execution/ (e.g. new_baseline)")
    parser.add_argument("--failed", type=int, default=36,
                        help="how many failed cases to include (default 36; "
                             "if fewer exist, all are included)")
    parser.add_argument("--passed", type=int, default=14,
                        help="how many random passed cases to include (default 14)")
    parser.add_argument("--output", required=True,
                        help="output JSON path")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic random sampling")
    parser.add_argument("--root", default=WORKSHOP)
    args = parser.parse_args()

    report_path = os.path.join(args.root, args.src, REPORT_FILENAME)
    if not os.path.exists(report_path):
        print(f"[error] report not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    failed_ids = [r["instance"] for r in report["results"]
                  if r.get("status") in ("failed", "crashed")]
    passed_ids = [r["instance"] for r in report["results"]
                  if r.get("status") == "passed"]

    print(f"Source experiment: {args.src}")
    print(f"  total cases:  {len(report['results'])}")
    print(f"  passed:       {len(passed_ids)}")
    print(f"  failed/crashed: {len(failed_ids)}")

    rng = random.Random(args.seed)

    # Failed: take up to N (sorted for stability)
    take_failed = sorted(failed_ids)[: args.failed]
    if args.failed > len(failed_ids):
        print(f"  [warn] only {len(failed_ids)} failed cases available, "
              f"asked for {args.failed}")

    # Passed: random sample
    if args.passed > len(passed_ids):
        print(f"  [warn] only {len(passed_ids)} passed cases available, "
              f"asked for {args.passed}")
        take_passed = passed_ids
    else:
        take_passed = rng.sample(passed_ids, args.passed)
    take_passed = sorted(take_passed)

    sampled = sorted(set(take_failed + take_passed))

    payload = {
        "src": args.src,
        "seed": args.seed,
        "n_failed": len(take_failed),
        "n_passed": len(take_passed),
        "n_total": len(sampled),
        "failed_ids": take_failed,
        "passed_ids": take_passed,
        "case_ids": sampled,
    }
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSampled {len(sampled)} cases ({len(take_failed)} failed + "
          f"{len(take_passed)} passed) → {out_path}")


if __name__ == "__main__":
    main()
