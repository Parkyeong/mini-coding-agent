"""Orchestration core: plan -> execute -> verify -> retry / replan.

This module IS the agent (the system) — it composes multiple LLMNode instances
(planner, coder), the verifier function, the test_runner, and memory into a
goal-pursuing loop. An individual LLMNode by itself isn't an "agent"; the agent
is what this file orchestrates.

This module is a library, not an entry point. Run the agent through a runner
(e.g. `python -m runners.mbpp_task run --exp <name>`), which constructs the
LLM nodes and calls run_task() for each task.
"""

from config import (
    ENABLE_METRICS,
    MAX_RETRIES_PER_STEPS as MAX_RETRIES_PER_STEP,
    MAX_REPLANS,
)
from environment import Environment
from llm_node import LLMNode
from memory import MemoryManager
from metrics import MetricsTracker
import tools as Tools
import planner as planner_role
import coder as coder_role
import verifier as verifier_role


def run_task(user_prompt: str, planner: LLMNode, coder: LLMNode,
             memory: MemoryManager, metrics: MetricsTracker = None) -> str:
    """Plan -> execute -> verify -> retry / replan, for one user task."""
    task_id = memory.generate_task_id()
    working = memory.begin_task(task_id=task_id, user_prompt=user_prompt)
    memory_context = memory.get_context_for_planner()
    failure_context = None
    total_attempts = 0
    final_plan: list[str] = []
    overall_passed = False
    error_history: list[dict] = []

    replans_used = 0

    try:
        for replan in range(MAX_REPLANS + 1):
            if replan > 0:
                replans_used = replan
                print(f"Replan #{replan}")

            plan_steps = planner_role.create_plan(
                planner,
                user_task=user_prompt,
                memory_context=memory_context,
                failure_context=failure_context,
            )
            final_plan = plan_steps
            working.set_plan(plan_steps)
            n_steps = len(plan_steps)
            print(f"Plan: {n_steps} steps")

            all_passed = True
            error_history = []

            for step_idx, step_desc in enumerate(plan_steps):
                step_passed = False
                current_step = step_desc

                for attempt in range(MAX_RETRIES_PER_STEP):
                    total_attempts += 1

                    coder_result = coder_role.run_coder(coder, current_step, memory)
                    verify_result = verifier_role.verify(memory, env=coder.env)

                    attempt_tag = "" if attempt == 0 else f" (retry {attempt})"
                    step_label = f"  step {step_idx+1}/{n_steps}"

                    if verify_result["passed"]:
                        print(f"{step_label}: PASS{attempt_tag}")
                        step_passed = True
                        break

                    will_retry = attempt < MAX_RETRIES_PER_STEP - 1
                    retry_marker = " — retry" if will_retry else ""
                    print(f"{step_label}: FAIL ({verify_result['reason']}){attempt_tag}{retry_marker}")

                    error_history.append({
                        "step": step_desc,
                        "attempt": attempt + 1,
                        "reason": verify_result["reason"],
                        "fix_suggestion": verify_result["fix_suggestion"],
                    })
                    current_step = (
                        f"{step_desc}\n\n"
                        f"Previous attempt failed:\n"
                        f"Reason:{verify_result['reason']}\n"
                        f"Fix suggestion:{verify_result['fix_suggestion']}"
                    )

                if not step_passed:
                    all_passed = False
                    failure_context = _build_failure_context(plan_steps, step_idx, error_history)
                    break

            if all_passed:
                overall_passed = True
                print(f"PASSED ({total_attempts} attempts, {replans_used} replans)")
                _print_metrics(metrics)
                return f"Task completed: {n_steps} steps."

        print(f"FAILED ({total_attempts} attempts, {replans_used} replans)")
        _print_metrics(metrics)
        return f"Task failed after {MAX_REPLANS} replan attempts."

    finally:
        memory.end_task(
            task_id=task_id,
            passed=overall_passed,
            plan=final_plan,
            attempts=total_attempts,
            summary="Completed successfully. " if overall_passed else "Failed. See error history for details.",
            error_history=error_history if not overall_passed else None,
        )


def _build_failure_context(plan: list, failed_step_idx: int, error_history: list) -> str:
    lines = [
        f"Previous plan:{plan}",
        f"Failed at step {failed_step_idx+1}:{plan[failed_step_idx]}",
        f"Attempts made:{len(error_history)}",
        "Error details:",
    ]
    for err in error_history[-3:]:
        lines.append(f"- Attempt {err['attempt']}:{err['reason']}")
        if err["fix_suggestion"]:
            lines.append(f"Suggestion:{err['fix_suggestion']}")
    return "\n".join(lines)


def _print_metrics(metrics):
    if ENABLE_METRICS and metrics:
        print(metrics.summary())


def build_llm_nodes(env: Environment, memory: MemoryManager,
                    metrics: MetricsTracker = None) -> tuple[LLMNode, LLMNode]:
    """Build the two LLM-driven role nodes (planner + coder). Verifier is a
    plain function (no LLM, just pytest) so it isn't an LLMNode and isn't
    returned here. Together with verifier and the rest of the system, these
    nodes form the agent that engine.run_task orchestrates."""
    planner = LLMNode(
        system_prompt=planner_role.PROMPT,
        role="planner",
        max_steps=1,
        metrics_tracker=metrics,
    )
    coder = LLMNode(
        system_prompt=coder_role.PROMPT,
        role="coder",
        tools=Tools.get_tools(),
        env=env,
        memory=memory,
        metrics_tracker=metrics,
    )
    return planner, coder
