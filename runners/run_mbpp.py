"""
Run the agent over previously materialized MBPP instances.

Each instance has its own per-instance memory.json (project_context +
task_history, kept for per-instance debugging) AND shares a single global
facts file (cross-instance fact accumulation, so reinforcement actually
happens across MBPP problems).
"""
import os
import json
import argparse
import glob

from config import WORKSHOP, ENABLE_METRICS, MBPP_GLOBAL_FACTS_FILE
import config as config_module

from planner import Planner
from coder import Coder
from verifier import Verifier
from memory import MemoryManager
from metrics import MetricsTracker
import tool

from main import run_task


def run_one(workspace: str) -> dict:
    # Override globals so all modules see the right workspace
    instance_name = os.path.basename(workspace)
    config_module.PROJECT_NAME = instance_name
    config_module.WORKSPACE = workspace

    # Dual-file mode: per-instance memory.json for task_history,
    # shared MBPP_GLOBAL_FACTS_FILE for facts
    memory = MemoryManager(
        memory_file=os.path.join(workspace, "memory.json"),
        global_facts_file=MBPP_GLOBAL_FACTS_FILE,
    )
    metrics = MetricsTracker() if ENABLE_METRICS else None

    planner  = Planner(metrics_tracker=metrics)
    coder    = Coder(metrics_tracker=metrics, memory=memory)
    verifier = Verifier(metrics_tracker=metrics, memory=memory)
    tool.set_memory_manager(memory)

    with open(os.path.join(workspace, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read()

    result_text = run_task(prompt, planner, coder, verifier, memory, metrics)
    last_record = memory.data["task_history"][-1] if memory.data["task_history"] else {}
    return {
        "instance": instance_name,
        "status": last_record.get("status", "unknown"),
        "attempts": last_record.get("attempts", 0),
        "files_changed": last_record.get("files_changed", []),
        "result_text": result_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="mbpp_*", help="glob under WORKSHOP")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default="mbpp_report.json")
    args = parser.parse_args()

    workspaces = sorted(glob.glob(os.path.join(WORKSHOP, args.pattern)))
    if args.limit:
        workspaces = workspaces[: args.limit]

    print(f"Running {len(workspaces)} MBPP instances ...")
    results = []
    for ws in workspaces:
        print(f"\n========== {os.path.basename(ws)} ==========")
        try:
            results.append(run_one(ws))
        except Exception as e:
            results.append({"instance": os.path.basename(ws), "status": "crashed", "error": str(e)})

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    crashed = sum(1 for r in results if r.get("status") == "crashed")
    other = total - passed - failed - crashed

    print(f"\n=== MBPP report ===")
    print(f"  passed:  {passed}/{total}")
    print(f"  failed:  {failed}/{total}  (LLM solved incorrectly — agent capability issue)")
    print(f"  crashed: {crashed}/{total}  (tool/system error — fix code, not the LLM)")
    if other:
        print(f"  other:   {other}/{total}")

    if crashed:
        print(f"\nCrashed instances (these need code fixes):")
        for r in results:
            if r.get("status") == "crashed":
                print(f"  - {r['instance']}: {r.get('error', '?')}")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()