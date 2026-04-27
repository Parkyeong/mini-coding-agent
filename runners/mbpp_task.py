"""Single-file MBPP runner: setup + run + report. Experiment-scoped layout.

Each experiment lives under WORKSHOP/<exp_name>/, fully self-contained:

    Execution/
      <exp_name>/
        single_case_details/             <- per-case workspaces
          mbpp_0011/
            prompt.md, solution.py, test_solution.py, memory.json
          mbpp_0012/
            ...
        mbpp_global_facts.json           <- facts shared across this run's cases
        mbpp_exp_final_results.json      <- structured pass/fail summary

Subcommands
-----------
    setup --exp NAME [--subset sanitized|full] [--split test|train|...] [--limit N]
        download MBPP and materialize N instances under <exp_name>/single_case_details/

    run   --exp NAME [--limit N]
        run agent over materialized instances, write report to <exp_name>/

    all   --exp NAME ...
        setup then run

Run from the project root:
    python -m runners.mbpp_task setup --exp baseline --limit 10
    python -m runners.mbpp_task run   --exp baseline
    python -m runners.mbpp_task all   --exp baseline --limit 10
"""
import argparse
import datetime
import glob
import json
import os
import sys

# Allow `python runners/mbpp_task.py ...` from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as config_module
from config import WORKSHOP, ENABLE_METRICS, COMMAND_TIMEOUT, MODEL
from environment import Environment
from main import run_task, build_agents
from memory import MemoryManager
from metrics import MetricsTracker


SOLUTION_STUB = '"""MBPP task — implement the function described in prompt.md."""\n'

# Filenames inside an experiment directory.
CASES_DIRNAME = "single_case_details"
FACTS_FILENAME = "mbpp_global_facts.json"
REPORT_FILENAME = "mbpp_exp_final_results.json"


def _exp_paths(exp_name: str) -> dict:
    """Resolve all paths for an experiment in one place."""
    exp_dir = os.path.join(WORKSHOP, exp_name)
    return {
        "exp_dir": exp_dir,
        "cases_dir": os.path.join(exp_dir, CASES_DIRNAME),
        "facts_file": os.path.join(exp_dir, FACTS_FILENAME),
        "report_file": os.path.join(exp_dir, REPORT_FILENAME),
    }


# ---------------------------------------------------------------------------
# setup: download dataset and materialize one workspace per problem
# ---------------------------------------------------------------------------

def materialize_instance(cases_dir: str, task_id: int, text: str,
                         test_list: list[str]) -> str:
    workspace = os.path.join(cases_dir, f"mbpp_{task_id:04d}")
    os.makedirs(workspace, exist_ok=True)

    with open(os.path.join(workspace, "solution.py"), "w", encoding="utf-8") as f:
        f.write(SOLUTION_STUB)

    test_body = "from solution import *\n\n"
    for i, t in enumerate(test_list):
        test_body += f"def test_case_{i}():\n    {t}\n\n"
    with open(os.path.join(workspace, "test_solution.py"), "w", encoding="utf-8") as f:
        f.write(test_body)

    # prompt.md: prose + asserts so the agent sees the exact required signature.
    tests_block = "\n".join(test_list)
    with open(os.path.join(workspace, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# MBPP Task {task_id}\n\n"
            f"{text}\n\n"
            f"## Your code should pass these tests:\n"
            f"```python\n{tests_block}\n```\n\n"
            f"Implement the solution in `solution.py`. "
            f"The function name and signature must exactly match the assertions above.\n"
        )

    memory = {
        "project_context": {
            "project_name": f"mbpp_{task_id:04d}",
            "workspace": workspace,
            "language": "python",
            "framework": "pytest",
            "entry_file": "solution.py",
            "test_command": "pytest -q test_solution.py",
            "test_timeout": 30,
            "updated_at": "",
        },
        "task_history": [],
        "facts": [],
    }
    with open(os.path.join(workspace, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    return workspace


def cmd_setup(args) -> None:
    paths = _exp_paths(args.exp)
    os.makedirs(paths["cases_dir"], exist_ok=True)

    # 'sanitized' is the standard eval set: hand-verified, function names in
    # prose match the asserts. 'full' has more instances but ~30% noise where
    # prose name != assert name, which costs the agent for no real reason.
    from datasets import load_dataset

    dataset_name = "google-research-datasets/mbpp"
    print(f"[exp={args.exp}] Loading {dataset_name}/{args.subset} split={args.split} ...")
    ds = load_dataset(dataset_name, args.subset, split=args.split)

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"Materializing {n} instances under {paths['cases_dir']} ...")

    # Field name diff between subsets: full uses 'text', sanitized uses 'prompt'.
    text_field = "prompt" if args.subset == "sanitized" else "text"

    for i in range(n):
        row = ds[i]
        ws = materialize_instance(
            paths["cases_dir"],
            task_id=row["task_id"],
            text=row[text_field],
            test_list=row["test_list"],
        )
        print(f"  [{i+1}/{n}] {ws}")

    print("Setup done.")


# ---------------------------------------------------------------------------
# run: drive the agent across materialized instances
# ---------------------------------------------------------------------------

def run_one(workspace: str, facts_file: str) -> dict:
    instance_name = os.path.basename(workspace)
    # Override globals so all modules see the right workspace for this instance.
    config_module.PROJECT_NAME = instance_name
    config_module.WORKSPACE = workspace

    memory = MemoryManager(
        memory_file=os.path.join(workspace, "memory.json"),
        global_facts_file=facts_file,
    )
    metrics = MetricsTracker() if ENABLE_METRICS else None
    env = Environment(workspace, command_timeout=COMMAND_TIMEOUT)

    planner, coder = build_agents(env, memory, metrics)

    with open(os.path.join(workspace, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read()

    result_text = run_task(prompt, planner, coder, memory, metrics)
    last_record = memory.data["task_history"][-1] if memory.data["task_history"] else {}
    return {
        "instance": instance_name,
        "status": last_record.get("status", "unknown"),
        "attempts": last_record.get("attempts", 0),
        "files_changed": last_record.get("files_changed", []),
        "result_text": result_text,
    }


def cmd_run(args) -> None:
    paths = _exp_paths(args.exp)

    if not os.path.isdir(paths["cases_dir"]):
        print(f"[exp={args.exp}] No cases dir at {paths['cases_dir']}. "
              f"Run `setup --exp {args.exp}` first.")
        return

    workspaces = sorted(glob.glob(os.path.join(paths["cases_dir"], "mbpp_*")))
    if args.limit:
        workspaces = workspaces[: args.limit]

    if not workspaces:
        print(f"[exp={args.exp}] No mbpp_* workspaces in {paths['cases_dir']}.")
        return

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[exp={args.exp}] Running {len(workspaces)} MBPP instances ...")
    print(f"  cases_dir : {paths['cases_dir']}")
    print(f"  facts_file: {paths['facts_file']}")
    print(f"  report    : {paths['report_file']}")

    results = []
    for ws in workspaces:
        print(f"\n========== {os.path.basename(ws)} ==========")
        try:
            results.append(run_one(ws, paths["facts_file"]))
        except Exception as e:
            results.append({
                "instance": os.path.basename(ws),
                "status": "crashed",
                "error": str(e),
            })

    finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    crashed = sum(1 for r in results if r.get("status") == "crashed")
    other = total - passed - failed - crashed

    summary = {
        "experiment": args.exp,
        "model": MODEL,
        "started_at": started_at,
        "finished_at": finished_at,
        "totals": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "crashed": crashed,
            "other": other,
        },
        "results": results,
    }

    print(f"\n=== MBPP report (exp={args.exp}) ===")
    print(f"  passed:  {passed}/{total}")
    print(f"  failed:  {failed}/{total}  (LLM solved incorrectly)")
    print(f"  crashed: {crashed}/{total}  (tool/system error)")
    if other:
        print(f"  other:   {other}/{total}")

    if crashed:
        print(f"\nCrashed instances:")
        for r in results:
            if r.get("status") == "crashed":
                print(f"  - {r['instance']}: {r.get('error', '?')}")

    os.makedirs(paths["exp_dir"], exist_ok=True)
    with open(paths["report_file"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {paths['report_file']}")


# ---------------------------------------------------------------------------
# all: setup + run
# ---------------------------------------------------------------------------

def cmd_all(args) -> None:
    cmd_setup(args)
    cmd_run(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_exp_arg(p):
    p.add_argument("--exp", required=True,
                   help="experiment name; folder name under Execution/")


def _add_setup_args(p):
    p.add_argument("--subset", default="sanitized", choices=["sanitized", "full"],
                   help="MBPP subset (default: sanitized = 427 hand-verified problems)")
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test", "prompt"])
    p.add_argument("--limit", type=int, default=10, help="0 = all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MBPP runner (setup + run, experiment-scoped)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Download MBPP and materialize instances")
    _add_exp_arg(p_setup)
    _add_setup_args(p_setup)
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="Run the agent over materialized instances")
    _add_exp_arg(p_run)
    p_run.add_argument("--limit", type=int, default=0, help="0 = all matched")
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("all", help="setup then run")
    _add_exp_arg(p_all)
    _add_setup_args(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
