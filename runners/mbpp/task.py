"""Single-file MBPP runner: setup + run + report. Experiment-scoped layout.

Each experiment lives under WORKSHOP/<exp_name>/, fully self-contained:

    Execution/
      <exp_name>/
        single_case_details/             <- per-case workspaces
          mbpp_0011/
            prompt.md, solution.py, test_solution.py
            long_term_memory.json        <- project_context + task_history + this case's facts
            working_memory.json          <- per-task event_log + plan + candidate_facts (written at end_task)
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
    python -m runners.mbpp.task setup --exp baseline --limit 10
    python -m runners.mbpp.task run   --exp baseline
    python -m runners.mbpp.task all   --exp baseline --limit 10
"""
import argparse
import datetime
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

# Allow `python runners/mbpp/task.py ...` from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    WORKSHOP, ENABLE_METRICS, COMMAND_TIMEOUT, MODEL,
    ENABLE_LLM_DEDUP, ROLE_CONFIGS, MAX_MEMORY_FACTS,
)
from engine import run_task, build_llm_nodes
from environment import Environment
from llm_node import LLMNode
from memory import MemoryManager, add_facts_to_pool, cap_pool
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

    long_term = {
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
    with open(os.path.join(workspace, "long_term_memory.json"), "w", encoding="utf-8") as f:
        json.dump(long_term, f, indent=2, ensure_ascii=False)

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

    text_field = "prompt" if args.subset == "sanitized" else "text"

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    rows = [ds[i] for i in range(n)]

    print(f"Materializing {n} instances under {paths['cases_dir']} ...")

    for i, row in enumerate(rows):
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

def run_one(workspace: str, seed_facts: list = None) -> dict:
    """Run the agent on one materialized case. Self-contained: no shared state
    with other cases (single-file memory mode), so this is safe to call from
    parallel workers. The runner merges per-case facts into the experiment-wide
    global pool periodically as workers finish.

    `seed_facts` is the read-only snapshot of the global pool at the moment this
    case started. Planner sees these as prior project knowledge alongside this
    case's own learnings.
    """
    instance_name = os.path.basename(workspace)

    memory = MemoryManager(
        long_term_file=os.path.join(workspace, "long_term_memory.json"),
        working_memory_file=os.path.join(workspace, "working_memory.json"),
        global_facts_file=None,
        seed_facts=seed_facts,
    )
    metrics = MetricsTracker() if ENABLE_METRICS else None
    env = Environment(
        workspace,
        command_timeout=COMMAND_TIMEOUT,
        protected_files=["test_solution.py"],
    )

    nodes = build_llm_nodes(env, memory, metrics)

    with open(os.path.join(workspace, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read()

    result_text = run_task(
        prompt,
        planner=nodes["planner"],
        coder=nodes["coder"],
        summarizer=nodes["summarizer"],
        memory=memory,
        metrics=metrics,
    )
    last_record = memory.data["task_history"][-1] if memory.data["task_history"] else {}
    learned_facts = list(memory.data.get("facts", []))
    return {
        "instance": instance_name,
        "status": last_record.get("status", "unknown"),
        "attempts": last_record.get("attempts", 0),
        "files_changed": last_record.get("files_changed", []),
        "result_text": result_text,
        "learned_facts": learned_facts,
    }


def _is_already_done(workspace: str) -> bool:
    """A case is considered done if its working_memory.json exists. The agent
    writes that file at end_task, so its presence means the case made it
    through the orchestrator without crashing the runner."""
    return os.path.exists(os.path.join(workspace, "working_memory.json"))


def _run_one_safely(workspace: str, seed_facts: list = None) -> dict:
    """Wrapper that turns exceptions into a 'crashed' result so a single bad
    case doesn't kill the whole batch."""
    instance_name = os.path.basename(workspace)
    try:
        return run_one(workspace, seed_facts=seed_facts)
    except Exception as e:
        return {
            "instance": instance_name,
            "status": "crashed",
            "error": str(e),
            "learned_facts": [],
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

    # --skip-existing: drop cases that already have working_memory.json. Lets
    # you resume a long parallel run after a crash without re-paying for
    # already-done cases.
    skipped: list[str] = []
    if args.skip_existing:
        kept = []
        for ws in workspaces:
            if _is_already_done(ws):
                skipped.append(os.path.basename(ws))
            else:
                kept.append(ws)
        workspaces = kept

    # Validate batch_size >= workers — see design note in cmd help.
    if args.batch_size < args.workers:
        print(f"[error] --batch-size ({args.batch_size}) must be >= "
              f"--workers ({args.workers}). A worker that finishes BEFORE the "
              f"buffer fills has to start the next case using the most recent "
              f"merged snapshot; if batch_size < workers, multiple in-flight "
              f"workers would have used a stale snapshot the buffer is "
              f"already overwriting.")
        sys.exit(2)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[exp={args.exp}] Running {len(workspaces)} MBPP instances "
          f"(workers={args.workers}, batch_size={args.batch_size}"
          f"{', skipped=' + str(len(skipped)) if skipped else ''}) ...")
    print(f"  cases_dir : {paths['cases_dir']}")
    print(f"  facts_file: {paths['facts_file']}")
    print(f"  report    : {paths['report_file']}")

    # Construct dedup_node once (shared across all merge calls). LLM calls go
    # through ROLE_CONFIGS["dedup"] each time. Safe to reuse a single LLMNode
    # across batches — its `messages` get reset on every find_equivalent call,
    # no leakage between rounds.
    dedup_node = None
    if ENABLE_LLM_DEDUP:
        from role_pool.dedup import PROMPT as DEDUP_PROMPT
        dedup_cfg = ROLE_CONFIGS["dedup"]
        dedup_node = LLMNode(
            system_prompt=DEDUP_PROMPT,
            role="dedup",
            max_steps=dedup_cfg["max_steps"],
            model=dedup_cfg["model"],
            temperature=dedup_cfg.get("temperature"),
            max_tokens=dedup_cfg.get("max_tokens"),
        )

    # Global facts pool: lives in main thread only. Workers read snapshots
    # at submit time, never mutate it directly. Each batch_size completed
    # cases triggers a single-threaded merge (LLM dedup + cap to 40).
    pool: list[dict] = []
    pending_merge: list[dict] = []  # buffer of completed-case results awaiting merge

    def _flush_pending_merge() -> None:
        """Merge all queued case facts into the global pool, then cap to
        MAX_MEMORY_FACTS. Called when buffer reaches batch_size (and once at end
        for any leftover). Safe to call with empty buffer (no-op)."""
        if not pending_merge:
            return
        all_facts: list[dict] = []
        for r in pending_merge:
            all_facts.extend(r.get("learned_facts", []) or [])
        before_pool = len(pool)
        before_facts = len(all_facts)
        hits = add_facts_to_pool(pool, all_facts, dedup_node) if all_facts else 0
        evicted = cap_pool(pool, MAX_MEMORY_FACTS)
        added = len(pool) - before_pool + evicted
        print(
            f"    [merge] {len(pending_merge)} case(s), {before_facts} raw facts "
            f"-> +{added} new, {hits} deduped, {evicted} evicted "
            f"| pool size = {len(pool)}",
            flush=True,
        )
        pending_merge.clear()

    results: list[dict] = []
    n = len(workspaces)
    if args.workers <= 1:
        # Sequential path — preserved for debugging / single-threaded runs.
        # Still gets seed_facts + periodic merge semantics for parity.
        for i, ws in enumerate(workspaces, 1):
            print(f"\n========== [{i}/{n}] {os.path.basename(ws)} ==========")
            res = _run_one_safely(ws, seed_facts=list(pool))
            results.append(res)
            if res.get("status") == "passed":
                pending_merge.append(res)
            if len(pending_merge) >= args.batch_size:
                _flush_pending_merge()
    else:
        # Rolling window: keep `workers` cases in flight at all times. As any
        # case completes, pull its facts into pending_merge; if buffer hits
        # batch_size, do the merge (single-threaded in main); then immediately
        # start the next case, snapshotting the (possibly updated) pool as seed.
        pending_workspaces = list(workspaces)
        in_flight: dict = {}  # future -> workspace
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            # Fill initial pipeline.
            while pending_workspaces and len(in_flight) < args.workers:
                ws = pending_workspaces.pop(0)
                fut = ex.submit(_run_one_safely, ws, list(pool))
                in_flight[fut] = ws

            done = 0
            while in_flight:
                completed, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in completed:
                    ws = in_flight.pop(fut)
                    done += 1
                    res = fut.result()
                    results.append(res)
                    status = res.get("status", "?")
                    print(f"  [{done:>3}/{n}] {os.path.basename(ws)}: {status}",
                          flush=True)

                    # Failed/crashed cases: drop their facts entirely (per project rule).
                    if status == "passed":
                        pending_merge.append(res)

                    # Batch boundary: merge accumulated facts into pool.
                    if len(pending_merge) >= args.batch_size:
                        _flush_pending_merge()

                    # Refill the worker with the next pending case (using the
                    # current pool as seed — captures any merge that just happened).
                    if pending_workspaces:
                        next_ws = pending_workspaces.pop(0)
                        new_fut = ex.submit(
                            _run_one_safely, next_ws, list(pool),
                        )
                        in_flight[new_fut] = next_ws

    # Tail merge: any leftover passed cases that didn't fill a final batch.
    _flush_pending_merge()

    finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    crashed = sum(1 for r in results if r.get("status") == "crashed")
    other = total - passed - failed - crashed

    # Sort results by instance name so output is deterministic regardless of
    # parallel completion order.
    results.sort(key=lambda r: r.get("instance", ""))

    summary = {
        "experiment": args.exp,
        "model": MODEL,
        "started_at": started_at,
        "finished_at": finished_at,
        "workers": args.workers,
        "skipped_existing": skipped,
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
    if skipped:
        print(f"  skipped (already had working_memory.json): {len(skipped)}")

    if crashed:
        print(f"\nCrashed instances:")
        for r in results:
            if r.get("status") == "crashed":
                print(f"  - {r['instance']}: {r.get('error', '?')}")

    os.makedirs(paths["exp_dir"], exist_ok=True)
    with open(paths["report_file"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {paths['report_file']}")

    # Save the in-memory global pool to disk. The pool was built incrementally
    # during the run via per-batch merges (see _flush_pending_merge above), so
    # there's no separate "merge after run" pass — just serialize the final
    # state.
    payload = {
        "facts": pool,
        "updated_at": datetime.datetime.now().isoformat(timespec='seconds'),
    }
    os.makedirs(os.path.dirname(paths["facts_file"]), exist_ok=True)
    with open(paths["facts_file"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Final global facts pool: {len(pool)} unique facts -> {paths['facts_file']}")

    # Render dataset.html alongside the report. Failure here is non-fatal — the
    # JSON outputs are the source of truth; rendering can always be retried via
    # `python -m runners.mbpp.html --exp <name>`.
    try:
        from runners.mbpp.html import render_experiment
        html_path = render_experiment(args.exp)
        print(f"HTML report saved to {html_path}")
    except Exception as e:
        print(f"[warn] HTML render failed (non-fatal): {e}")


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


def _add_run_args(p):
    p.add_argument("--workers", type=int, default=4,
                   help="number of parallel agent workers (default: 4; "
                        "set to 1 for sequential / debugging)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="merge facts into the global pool every N completed "
                        "cases (default: 4). Must be >= --workers.")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cases whose working_memory.json already exists "
                        "(use to resume after a crash)")


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
    _add_run_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("all", help="setup then run")
    _add_exp_arg(p_all)
    _add_setup_args(p_all)
    _add_run_args(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
