import argparse
import tool
from config import PROVIDER, MODEL, ENABLE_METRICS, MAX_RETRIES_PER_STEPS as MAX_RETRIES_PER_STEP, MAX_REPLANS
from planner import Planner
from coder import Coder
from verifier import Verifier
from memory import MemoryManager
from metrics import MetricsTracker



def run_task(user_prompt:str, planner:Planner, coder:Coder,
            verifier:Verifier, memory:MemoryManager,metrics:MetricsTracker=None)->str:
    """Execute the full flow for a single task: plan -> execute -> verify -> retry/replan """

    task_id = memory.generate_task_id()

    working = memory.begin_task(task_id = task_id, user_prompt = user_prompt)

    memory_context = memory.get_context_for_planner()
    failure_context = None
    total_attempts= 0
    final_plan:list[str] = []
    overall_passed = False

    print(f"\n{'='*50}")
    print(f"Task[{task_id}]:{user_prompt}")
    print(f"\n{'='*50}")



    try:
        for replan in range(MAX_REPLANS+1):
            if replan > 0:
                print(f"\n-- replan attempt {replan}/{MAX_REPLANS}--")

            # Planning
            print("\n[Phase: Planning]")
            plan_steps = planner.create_plan(
                user_task = user_prompt,
                memory_context = memory_context,
                failure_context = failure_context
            )
            final_plan = plan_steps
            working.set_plan(plan_steps)

            for index, step in enumerate(plan_steps):
                print(f"  Step{index+1}: {step}")

            # Execution
            print("\n[Phase: Execution]")
            all_passed = True
            error_history = []

            for step_idx, step_desc in enumerate(plan_steps):
                print(f"\n ---Step {step_idx+1}/{len(plan_steps)}:{step_desc}---")
                step_passed = False
                current_step = step_desc

                for attempt in range(MAX_RETRIES_PER_STEP):
                    total_attempts += 1
                    if attempt > 0:
                        print(f"Retry {attempt}/{MAX_RETRIES_PER_STEP-1}")

                    # Coder Execution
                    coder.reset_message()
                    coder_result = coder.run(current_step)
                    print(f"[Coder]{'completed' if coder_result['completed'] else 'max steps reached'}")

                    # Verifier Execution
                    verify_result = verifier.verify(
                        user_prompt = user_prompt,
                        step_description = step_desc,
                        coder_result = coder_result["text"],
                        files_changed = list(working.files_changed)
                    )

                    print(f"[Verifier]{verify_result['reason']}")

                    if verify_result["passed"]:
                        print(f"PASSED")
                        step_passed = True
                        break

                    else:
                        print(f"FAILED: {verify_result['fix_suggestion']}")
                        error_history.append({
                            "step":step_desc,
                            "attempt":attempt+1,
                            "reason":verify_result["reason"],
                            "fix_suggestion":verify_result["fix_suggestion"]
                        })

                        current_step =(
                            f"{step_desc}\n\n"
                            f"Previous attempt failed:\n"
                            f"Reason:{verify_result['reason']}\n"
                            f"Fix suggestion:{verify_result['fix_suggestion']}"
                        )

                if not step_passed:
                    all_passed = False
                    failure_context = _build_failure_context(plan_steps, step_idx, error_history)
                    break

            # After all steps in this plan attempt
            if all_passed:
                overall_passed = True
                summary = f"Completed {len(plan_steps)} steps successfully."
                _print_metrics(metrics)
                return f"Task completed successfully.\n{summary}"

        # All replans exhausted
        summary = f"Task failed after {MAX_REPLANS} replan attempts."
        _print_metrics(metrics)
        return summary

    finally:
        # End task: promote (if passed) and record history regardless
        memory.end_task(
            task_id = task_id,
            passed = overall_passed,
            plan = final_plan,
            attempts = total_attempts,
            summary = "Completed successfully. " if overall_passed else "Failed. See error history for details."
        )

def _build_failure_context(plan:list, failed_step_idx:int, error_history:list)->str:
    """Build failure context for Planner"""
    lines = [
        f"Previous plan:{plan}",
        f"Failed at step {failed_step_idx+1}:{plan[failed_step_idx]}",
        f"Attempts made:{len(error_history)}",
        "Error details:"
    ]
    for err in error_history[-3:]:
        lines.append(f"- Attempt {err['attempt']}:{err['reason']}")
        if err['fix_suggestion']:
            lines.append(f"Suggestion:{err['fix_suggestion']}")

    return "\n".join(lines)

def _print_metrics(metrics):
    if ENABLE_METRICS and metrics:
        print(f"\n -- Metrics--")
        print(metrics.summary())



def main():
    parser = argparse.ArgumentParser(description="Mini Coding Agent")
    parser.add_argument("--project", required=True, help="Project name (used as workspace folder name)")
    args = parser.parse_args()

    from config import set_project
    set_project(args.project)

    from config import WORKSPACE
    print("=" * 40)
    print(f"Mini Coding Agent")
    print(f"LLM: {PROVIDER} / {MODEL}")
    print(f"Project: {args.project}")
    print(f"Workspace: {WORKSPACE}")
    print(f"Type 'exit' to quit.")
    print("=" * 40)

    memory = MemoryManager()
    metrics = MetricsTracker() if ENABLE_METRICS else None

    planner = Planner(metrics_tracker=metrics)
    coder = Coder(metrics_tracker=metrics,memory=memory)
    verifier = Verifier(metrics_tracker=metrics,memory=memory)

    tool.set_memory_manager(memory)

    while True:
        user_input = input("\n User:").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        result = run_task(user_input, planner, coder, verifier, memory, metrics)
        print("=" * 40)
        print(f"Result: {result}")
        print("=" * 40)

if __name__ == "__main__":
    main()
