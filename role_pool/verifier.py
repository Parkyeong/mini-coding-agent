"""Verifier role: run the project's tests, return pass/fail directly.

For benchmark evaluation (MBPP / SWE-bench style) the test result IS the
verdict — no LLM judge needed. We just look at pytest's returncode and
hand the failure output back as fix_suggestion for the next coder retry.

Every verify() call also pushes a "verify" event into the working memory's
event_log so the engine-driven verifier pytest invocations are visible in
the trace alongside coder's manual run_command pytest invocations. Without
this, the timeline only shows pytest runs that happened to pass through the
LLMNode tool loop (coder's choice) and silently misses the engine-mandated
ones (one per coder iteration).
"""

from config import VERIFIER_RUN_TESTS
from tool_pool.test_runner import run_tests


def verify(memory, env) -> dict:
    """Run the project's tests once.

    Args:
        memory:  MemoryManager — used to read project_context (test_command,
                 test_timeout). May be None.
        env:     Environment — owns the workspace path the tests run against.
                 Required (we no longer fall back to a global config var).

    Returns:
        {
          "passed":         bool,
          "reason":         short summary of the outcome,
          "fix_suggestion": failure excerpt (empty when passed),
          "test_block":     full [TestRunner] block, useful for logging,
        }
    """
    verdict, run_result = _verify_impl(memory, env)
    _log_verify_event(memory, verdict, run_result)
    return verdict


def _verify_impl(memory, env):
    """Pure verdict computation. Returns (verdict_dict, run_result_or_None).
    run_result is the TestRunResult when tests actually ran, None otherwise."""
    if not VERIFIER_RUN_TESTS:
        return ({
            "passed": False,
            "reason": "test execution disabled by config",
            "fix_suggestion": "",
            "test_block": "[TestRunner] disabled by config",
        }, None)

    cmd_hint = None
    timeout_hint = None
    if memory is not None:
        ctx = memory.data.get("project_context", {})
        cmd_hint = ctx.get("test_command") or None
        timeout_hint = ctx.get("test_timeout") or None

    result = run_tests(
        env,
        memory_hint_command=cmd_hint,
        memory_hint_timeout=timeout_hint,
    )
    test_block = result.to_prompt_block()

    if result.passed():
        return ({
            "passed": True,
            "reason": f"all tests passed ({result.command})",
            "fix_suggestion": "",
            "test_block": test_block,
        }, result)

    if not result.executed:
        return ({
            "passed": False,
            "reason": f"test execution skipped: {result.error or 'no test command'}",
            "fix_suggestion": "ensure project_context.test_command is set in memory.json",
            "test_block": test_block,
        }, result)

    if result.timed_out:
        return ({
            "passed": False,
            "reason": f"tests timed out (timeout={timeout_hint or 'default'}s)",
            "fix_suggestion": "narrow scope or increase test_timeout",
            "test_block": test_block,
        }, result)

    # tests ran but returncode != 0 — feed the failure tail back so coder can
    # see what failed on retry. stderr first because pytest writes most
    # failure info there; fall back to stdout if stderr is empty.
    failure_excerpt = (result.stderr or result.stdout or "")[-1500:]
    return ({
        "passed": False,
        "reason": f"tests failed (returncode={result.returncode})",
        "fix_suggestion": failure_excerpt,
        "test_block": test_block,
    }, result)


def _log_verify_event(memory, verdict, run_result) -> None:
    """Push a 'verify' event onto the working memory event_log so the
    engine-driven pytest invocations are discoverable in the case trace.
    Safe to call with memory=None or with no active working memory."""
    if memory is None:
        return
    wm = memory.get_working() if hasattr(memory, "get_working") else None
    if wm is None:
        return
    payload = {
        "passed": bool(verdict.get("passed", False)),
        "reason": (verdict.get("reason") or "")[:200],
        "command": getattr(run_result, "command", None) if run_result else None,
        "returncode": getattr(run_result, "returncode", None) if run_result else None,
        "timed_out": getattr(run_result, "timed_out", False) if run_result else False,
    }
    wm.event_log.append("verify", payload)
