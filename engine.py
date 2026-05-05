"""Orchestration core: planner -> coder -> verifier -> summarizer.

One pipeline, one task. Three nested feedback loops:

  outer  : replan (up to MAX_REPLANS+1 times)
  middle : iterate plan steps
  inner  : per-step retry (up to MAX_RETRIES_PER_STEP times)

On all-steps-pass, summarizer extracts 1-2 project-level facts. On failure,
candidate facts are dropped and end_task records the error history.
"""

from config import (
    ENABLE_METRICS,
    MAX_RETRIES_PER_STEPS as MAX_RETRIES_PER_STEP,
    MAX_REPLANS,
    ROLE_CONFIGS,
)
from environment import Environment
from llm_node import LLMNode
from memory import MemoryManager
from metrics import MetricsTracker
import tool_pool as Tools
from role_pool import planner as planner_role
from role_pool import coder as coder_role
from role_pool import verifier as verifier_role
from role_pool import summarizer as summarizer_role


# Role-name → module mapping for the LLM-driven roles built by
# build_llm_nodes. dedup is built by the runner (mbpp_task.py), not here,
# so it doesn't appear in this map. verifier is non-LLM (no PROMPT module
# attribute used here), so it doesn't appear either.
_ROLE_MODULES = {
    "planner": planner_role,
    "coder": coder_role,
    "summarizer": summarizer_role,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_task(user_prompt: str, planner: LLMNode, coder: LLMNode,
             summarizer: LLMNode, memory: MemoryManager,
             metrics: MetricsTracker = None) -> str:
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

                    coder_role.run_coder(coder, current_step, memory)
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
                _try_summarize(summarizer, memory, coder.env)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_summarize(summarizer, memory, env) -> None:
    """Call summarizer; fail-soft on any error."""
    if summarizer is None:
        return
    try:
        new_facts = summarizer_role.summarize(summarizer, memory, env=env)
        wm = memory.get_working()
        if wm is not None:
            for f in new_facts:
                wm.add_candidate_fact(f["fact"], f["category"])
        if new_facts:
            print(f"Summarizer: {len(new_facts)} fact(s) extracted")
    except Exception as e:
        print(f"[warn] summarizer failed (non-fatal): {type(e).__name__}: {e}")


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


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------

def build_llm_nodes(env: Environment, memory: MemoryManager,
                    metrics: MetricsTracker = None) -> dict:
    """Build LLMNodes for every LLM-driven role declared in config.ROLE_CONFIGS.

    Iterates the dict so adding a new role = adding a config entry + a module
    in role_pool/ + an entry in _ROLE_MODULES — no special-casing here.
    Roles with model=None (e.g. verifier) are skipped. dedup is built by the
    runner, not here.
    """
    nodes: dict = {}
    for name, module in _ROLE_MODULES.items():
        cfg = ROLE_CONFIGS.get(name)
        if cfg is None or cfg["model"] is None:
            continue
        nodes[name] = LLMNode(
            system_prompt=module.PROMPT,
            role=name,
            tools=Tools.get_tools() if cfg["uses_tools"] else None,
            max_steps=cfg["max_steps"],
            env=env if cfg["uses_tools"] else None,
            memory=memory,
            metrics_tracker=metrics,
            model=cfg["model"],
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
        )
    return nodes
