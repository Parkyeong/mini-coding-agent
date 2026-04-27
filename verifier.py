"""Verifier role: run the project's tests, return pass/fail directly.

For benchmark evaluation (MBPP / SWE-bench style) the test result IS the
verdict — no LLM judge needed. We just look at pytest's returncode and
hand the failure output back as fix_suggestion for the next coder retry.
"""

import config
from config import VERIFIER_RUN_TESTS
from test_runner import run_tests


def verify(memory) -> dict:
    """Run the project's tests once.

    Returns:
        {
          "passed":         bool,
          "reason":         short summary of the outcome,
          "fix_suggestion": failure excerpt (empty when passed),
          "test_block":     full [TestRunner] block, useful for logging,
        }
    """
    if not VERIFIER_RUN_TESTS:
        return {
            "passed": False,
            "reason": "test execution disabled by config",
            "fix_suggestion": "",
            "test_block": "[TestRunner] disabled by config",
        }

    cmd_hint = None
    timeout_hint = None
    if memory is not None:
        ctx = memory.data.get("project_context", {})
        cmd_hint = ctx.get("test_command") or None
        timeout_hint = ctx.get("test_timeout") or None

    result = run_tests(
        workspace=config.WORKSPACE,
        memory_hint_command=cmd_hint,
        memory_hint_timeout=timeout_hint,
    )
    test_block = result.to_prompt_block()

    if result.passed():
        return {
            "passed": True,
            "reason": f"all tests passed ({result.command})",
            "fix_suggestion": "",
            "test_block": test_block,
        }

    if not result.executed:
        return {
            "passed": False,
            "reason": f"test execution skipped: {result.error or 'no test command'}",
            "fix_suggestion": "ensure project_context.test_command is set in memory.json",
            "test_block": test_block,
        }

    if result.timed_out:
        return {
            "passed": False,
            "reason": f"tests timed out (timeout={timeout_hint or 'default'}s)",
            "fix_suggestion": "narrow scope or increase test_timeout",
            "test_block": test_block,
        }

    # tests ran but returncode != 0 — feed the failure tail back so coder can
    # see what failed on retry. stderr first because pytest writes most
    # failure info there; fall back to stdout if stderr is empty.
    failure_excerpt = (result.stderr or result.stdout or "")[-1500:]
    return {
        "passed": False,
        "reason": f"tests failed (returncode={result.returncode})",
        "fix_suggestion": failure_excerpt,
        "test_block": test_block,
    }
