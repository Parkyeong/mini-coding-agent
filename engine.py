"""Orchestration core: dispatches a task through one of several pipelines.

Four pipelines exposed via run_task(config=...):

  c0_baseline  — current default. planner → coder (per-step retry) → verifier
                  → on pass, summarizer → end_task. Replan up to MAX_REPLANS.

  c1_judge     — task_judge ranks the task as "simple" or "complex".
                  simple  → coder ReAct loop → verifier → summarizer
                  complex → identical to c0_baseline

  c2_planspec  — planner uses the spec-extraction PROMPT (planner.PROMPT_WITH_SPEC):
                  it first reverse-engineers a rule from the test cases, then
                  emits steps. The rule is prepended to every step the coder
                  receives. Otherwise identical to c0_baseline.

  c3_codespec  — no planner. Coder uses coder.PROMPT_WITH_SPEC, which tells it
                  to first derive a rule from tests, then implement. Coder ReAct
                  loop with up to MAX_ATTEMPTS_NO_PLANNER attempts.

Every pipeline ends with summarizer (on pass) and end_task. Failure paths
are fail-soft (summarizer skipped, candidate_facts dropped).
"""

from config import (
    ENABLE_METRICS,
    MAX_RETRIES_PER_STEPS as MAX_RETRIES_PER_STEP,
    MAX_REPLANS,
    SUMMARIZER_MODEL,
    JUDGE_MODEL,
)
from environment import Environment
from llm_node import LLMNode
from memory import MemoryManager
from metrics import MetricsTracker
import tools as Tools
import planner as planner_role
import coder as coder_role
import verifier as verifier_role
import summarizer as summarizer_role
import task_judge as judge_role


# How many coder iterations before giving up in c1_simple / c3 pipelines
# (no planner = no replan, so we cap by attempt count). Each attempt is a
# fresh coder.run() with the failure context appended to the prompt.
MAX_ATTEMPTS_NO_PLANNER = 5


# ---------------------------------------------------------------------------
# Public entry point — dispatches by config
# ---------------------------------------------------------------------------

def run_task(user_prompt: str, planner: LLMNode, coder: LLMNode,
             summarizer: LLMNode, memory: MemoryManager,
             metrics: MetricsTracker = None,
             config: str = "c0_baseline",
             judge: LLMNode = None,
             planner_spec: LLMNode = None,
             coder_spec: LLMNode = None) -> str:
    """Dispatch into the pipeline selected by `config`.

    Args:
        config: one of "c0_baseline" | "c1_judge" | "c2_planspec" | "c3_codespec"
        judge:        required for c1_judge
        planner_spec: required for c2_planspec (planner with PROMPT_WITH_SPEC)
        coder_spec:   required for c3_codespec (coder with PROMPT_WITH_SPEC)

    The base planner / coder / summarizer nodes are required for all configs.
    Extra nodes are needed only by the configs that use them.
    """
    if config == "c0_baseline":
        return _run_with_planner(
            user_prompt, planner, coder, summarizer, memory, metrics,
        )
    if config == "c1_judge":
        if judge is None:
            raise ValueError("config=c1_judge requires `judge` LLMNode")
        return _run_with_judge(
            user_prompt, planner, coder, summarizer, memory, metrics, judge,
        )
    if config == "c2_planspec":
        if planner_spec is None:
            raise ValueError("config=c2_planspec requires `planner_spec` LLMNode")
        return _run_with_plan_spec(
            user_prompt, planner_spec, coder, summarizer, memory, metrics,
        )
    if config == "c3_codespec":
        if coder_spec is None:
            raise ValueError("config=c3_codespec requires `coder_spec` LLMNode")
        return _run_no_planner_codespec(
            user_prompt, coder_spec, summarizer, memory, metrics,
        )
    raise ValueError(f"unknown config: {config}")


# ---------------------------------------------------------------------------
# c0_baseline: original planner-driven pipeline
# ---------------------------------------------------------------------------

def _run_with_planner(user_prompt, planner, coder, summarizer, memory, metrics) -> str:
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
# c1_judge: judge first, then dispatch to with-planner or no-planner
# ---------------------------------------------------------------------------

def _run_with_judge(user_prompt, planner, coder, summarizer, memory, metrics, judge) -> str:
    """Run the judge once. If it says 'simple', use no-planner pipeline; if
    'complex', use the full planner-driven pipeline. Default to complex on
    parse failure (defensive: rather use planner unnecessarily than skip it
    when it might have helped)."""
    test_block = _extract_test_block(user_prompt)
    decision = judge_role.judge_complexity(judge, user_prompt, test_block)
    print(f"Judge decision: {decision}")

    # Log the judge decision into the working memory so dataset.html can
    # surface it (working memory is created by the downstream pipeline).
    if decision == "simple":
        return _run_no_planner_with_judge(
            user_prompt, coder, summarizer, memory, metrics, judge_decision="simple",
        )
    return _run_with_planner_judge_logged(
        user_prompt, planner, coder, summarizer, memory, metrics,
        judge_decision="complex",
    )


def _run_with_planner_judge_logged(user_prompt, planner, coder, summarizer,
                                   memory, metrics, judge_decision) -> str:
    """Same as _run_with_planner but tags the working memory with the judge
    decision once begin_task has happened."""
    task_id = memory.generate_task_id()
    memory.begin_task(task_id=task_id, user_prompt=user_prompt)
    _log_judge_decision(memory, judge_decision)
    # Re-end the task and call the standard pipeline. Cleaner: inline the
    # logic with the judge tag. We'll just call _run_with_planner manually
    # since memory.end_task is in its finally block — but we already started
    # a task. So we end this one and let _run_with_planner start a fresh one.
    # Instead: refactor to put the tag inside _run_with_planner via a flag.
    # For simplicity here we end and restart.
    memory.end_task(
        task_id=task_id, passed=False, plan=[], attempts=0,
        summary="judge tagging only — restarting under planner pipeline",
    )
    return _run_with_planner(user_prompt, planner, coder, summarizer, memory, metrics)


def _run_no_planner_with_judge(user_prompt, coder, summarizer, memory, metrics,
                               judge_decision: str) -> str:
    """No-planner pipeline with judge_decision tagged on working memory."""
    return _run_no_planner_loop(
        user_prompt, coder, summarizer, memory, metrics,
        judge_decision=judge_decision, use_spec_input=False,
    )


# ---------------------------------------------------------------------------
# c2_planspec: planner emits rule + steps; rule is prepended to every step
# ---------------------------------------------------------------------------

def _run_with_plan_spec(user_prompt, planner_spec, coder, summarizer,
                       memory, metrics) -> str:
    """Like _run_with_planner but the planner runs PROMPT_WITH_SPEC and emits
    a (rule, examples_walk, steps) tuple. The rule is included verbatim in
    every step the coder receives, so coder always implements based on the
    test-derived rule, not just on the natural-language prompt.
    """
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

            rule, walk, plan_steps = planner_role.create_plan_with_spec(
                planner_spec,
                user_task=user_prompt,
                memory_context=memory_context,
                failure_context=failure_context,
            )
            final_plan = plan_steps
            working.set_plan(plan_steps)
            n_steps = len(plan_steps)
            print(f"Plan-with-spec: rule extracted, {n_steps} steps")
            _log_planspec_metadata(memory, rule, walk)

            # Build the rule preamble that every step's prompt will include.
            rule_preamble = (
                f"## Rule (derived from tests by planner)\n{rule}\n\n"
                f"## Examples walkthrough\n{walk}\n\n"
                if rule else ""
            )

            all_passed = True
            error_history = []
            for step_idx, step_desc in enumerate(plan_steps):
                step_passed = False
                current_step = f"{rule_preamble}## Current step\n{step_desc}"

                for attempt in range(MAX_RETRIES_PER_STEP):
                    total_attempts += 1
                    coder_role.run_coder(coder, current_step, memory)
                    verify_result = verifier_role.verify(memory, env=coder.env)

                    step_label = f"  step {step_idx+1}/{n_steps}"
                    attempt_tag = "" if attempt == 0 else f" (retry {attempt})"

                    if verify_result["passed"]:
                        print(f"{step_label}: PASS{attempt_tag}")
                        step_passed = True
                        break

                    will_retry = attempt < MAX_RETRIES_PER_STEP - 1
                    retry_marker = " — retry" if will_retry else ""
                    print(f"{step_label}: FAIL ({verify_result['reason']}){attempt_tag}{retry_marker}")

                    error_history.append({
                        "step": step_desc, "attempt": attempt + 1,
                        "reason": verify_result["reason"],
                        "fix_suggestion": verify_result["fix_suggestion"],
                    })
                    current_step = (
                        f"{rule_preamble}## Current step\n{step_desc}\n\n"
                        f"Previous attempt failed:\n"
                        f"Reason: {verify_result['reason']}\n"
                        f"Fix suggestion: {verify_result['fix_suggestion']}"
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
            task_id=task_id, passed=overall_passed, plan=final_plan,
            attempts=total_attempts,
            summary="Completed successfully. " if overall_passed else "Failed.",
            error_history=error_history if not overall_passed else None,
        )


# ---------------------------------------------------------------------------
# c3_codespec: no planner; coder uses spec-extraction PROMPT
# ---------------------------------------------------------------------------

def _run_no_planner_codespec(user_prompt, coder_spec, summarizer, memory, metrics) -> str:
    return _run_no_planner_loop(
        user_prompt, coder_spec, summarizer, memory, metrics,
        judge_decision=None, use_spec_input=True,
    )


# ---------------------------------------------------------------------------
# Shared no-planner loop (used by c1_simple and c3_codespec)
# ---------------------------------------------------------------------------

def _run_no_planner_loop(user_prompt, coder, summarizer, memory, metrics,
                         judge_decision: str = None, use_spec_input: bool = False) -> str:
    """Coder ReAct loop with no planner. Each attempt re-runs coder over the
    full task; on failure, the verifier's fix_suggestion is appended to the
    prompt for the next attempt.
    """
    task_id = memory.generate_task_id()
    memory.begin_task(task_id=task_id, user_prompt=user_prompt)
    if judge_decision:
        _log_judge_decision(memory, judge_decision)

    overall_passed = False
    total_attempts = 0
    accumulated_error = ""

    try:
        for attempt in range(MAX_ATTEMPTS_NO_PLANNER):
            total_attempts += 1
            current_input = user_prompt
            if accumulated_error:
                current_input = (
                    f"{user_prompt}\n\n"
                    f"## Previous attempt failed (last verifier output)\n"
                    f"{accumulated_error}"
                )

            if use_spec_input:
                coder_role.run_coder_with_spec(coder, current_input, memory)
            else:
                coder_role.run_coder(coder, current_input, memory)
            verify_result = verifier_role.verify(memory, env=coder.env)

            attempt_label = f"  attempt {attempt+1}/{MAX_ATTEMPTS_NO_PLANNER}"
            if verify_result["passed"]:
                print(f"{attempt_label}: PASS")
                overall_passed = True
                _try_summarize(summarizer, memory, coder.env)
                print(f"PASSED ({total_attempts} attempts, 0 replans)")
                _print_metrics(metrics)
                return f"Task completed in {total_attempts} attempt(s)."

            print(f"{attempt_label}: FAIL ({verify_result['reason']})")
            accumulated_error = (
                f"Reason: {verify_result['reason']}\n"
                f"Fix suggestion: {verify_result['fix_suggestion']}"
            )

        print(f"FAILED after {total_attempts} attempts (no planner)")
        _print_metrics(metrics)
        return f"Task failed after {total_attempts} attempts."

    finally:
        memory.end_task(
            task_id=task_id, passed=overall_passed, plan=[],
            attempts=total_attempts,
            summary="Completed successfully. " if overall_passed else "Failed.",
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


def _log_judge_decision(memory, decision: str) -> None:
    if memory is None:
        return
    wm = memory.get_working()
    if wm is None:
        return
    wm.event_log.append("judge", {"decision": decision})


def _log_planspec_metadata(memory, rule: str, walk: str) -> None:
    if memory is None:
        return
    wm = memory.get_working()
    if wm is None:
        return
    wm.event_log.append("planspec", {
        "rule": rule[:1000],
        "examples_walk": walk[:1000],
    })


def _extract_test_block(user_prompt: str) -> str:
    """Pull the ```python ... ``` block of asserts out of a prompt.md style
    user prompt. Returns empty string if no block found — judge then runs on
    the prompt alone."""
    import re
    m = re.search(r"```python\s*\n(.*?)```", user_prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


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
# Node construction (now returns a dict so different configs can pull only
# what they need without touching the runner signature each time)
# ---------------------------------------------------------------------------

def build_llm_nodes(env: Environment, memory: MemoryManager,
                    metrics: MetricsTracker = None,
                    config: str = "c0_baseline") -> dict:
    """Build the LLM nodes needed for `config`. Returns a dict of
    {planner, coder, summarizer, judge, planner_spec, coder_spec},
    only filling in keys the chosen config actually uses (others are None
    for clarity).

    All configs need coder + summarizer. c0/c1/c2 also need planner.
    Extras: c1 → judge, c2 → planner_spec, c3 → coder_spec.
    """
    nodes: dict = {
        "planner": None, "coder": None, "summarizer": None,
        "judge": None, "planner_spec": None, "coder_spec": None,
    }

    # Always built (used by every config)
    nodes["coder"] = LLMNode(
        system_prompt=coder_role.PROMPT,
        role="coder",
        tools=Tools.get_tools(),
        env=env,
        memory=memory,
        metrics_tracker=metrics,
    )
    nodes["summarizer"] = LLMNode(
        system_prompt=summarizer_role.PROMPT,
        role="summarizer",
        max_steps=1,
        memory=memory,
        metrics_tracker=metrics,
        model=SUMMARIZER_MODEL,
    )

    # Planner is needed by c0 / c1 (complex branch) / c2 (as planner_spec)
    if config in ("c0_baseline", "c1_judge"):
        nodes["planner"] = LLMNode(
            system_prompt=planner_role.PROMPT,
            role="planner",
            max_steps=1,
            memory=memory,
            metrics_tracker=metrics,
        )

    if config == "c1_judge":
        nodes["judge"] = LLMNode(
            system_prompt=judge_role.PROMPT,
            role="judge",
            max_steps=1,
            memory=memory,
            metrics_tracker=metrics,
            model=JUDGE_MODEL,
        )

    if config == "c2_planspec":
        nodes["planner_spec"] = LLMNode(
            system_prompt=planner_role.PROMPT_WITH_SPEC,
            role="planner_spec",
            max_steps=1,
            memory=memory,
            metrics_tracker=metrics,
        )

    if config == "c3_codespec":
        nodes["coder_spec"] = LLMNode(
            system_prompt=coder_role.PROMPT_WITH_SPEC,
            role="coder_spec",
            tools=Tools.get_tools(),
            env=env,
            memory=memory,
            metrics_tracker=metrics,
        )

    return nodes
