from agent import BaseAgent
import config
from config import VERIFIER_RUN_TESTS
from test_runner import run_tests, TestRunResult

VERIFIER_PROMPT = """
You are a code change verifier.

Your job: judge whether the step was completed correctly.

You will receive:
- the original user request
- the step that was being executed
- the coder's output, including any commands they ran and their results
- the list of files the coder actually modified during this step
- (optionally) a [TestRunner] block with the REAL output of running the project's tests

Trust the [TestRunner] block over the coder's self-report when they disagree.

Check these things:
1. Functional correctness: Do the execution results (test output, command output) show success?
2. Intent alignment: Does the change actually address what the user asked for?
3. Scope: are the modified files reasonable for the requested change?
   Flag any files modified that look unrelated to the request.
4. Side effects: anything in stderr suggesting unrelated breakage?


**You MUST respond in EXACTLY this format (no other text):
STATUS: PASSED
REASON: <one line citing concrete evidence>
FIX_SUGGESTION: None

Or if failed:
STATUS: FAILED
REASON: <one line citing concrete evidence, e.g. "pytest reported 2 failures in test_foo.py">
FIX_SUGGESTION: <one line>**"""


class Verifier(BaseAgent):
    def __init__(self, metrics_tracker=None, memory=None):
        super().__init__(
            system_prompt=VERIFIER_PROMPT,
            tools=[],
            max_steps=1,
            metrics_tracker=metrics_tracker,
            agent_role="verifier",
        )

        self.memory = memory  # MemoryManager, used to read test_command/test_timeout

    def verify(self, user_prompt: str, step_description: str, coder_result: str,
               files_changed: list[str] | None = None) -> dict:
        """
        Verify the coder's execution result.

        Args:
            user_prompt: original user prompt.
            step_description: the step being executed.
            coder_result: coder's output.
            files_changed: list of files the coder modified.

        Returns:
            dict with keys: passed, reason, fix_suggestion
        """
        files_changed = files_changed or []
        test_block, test_result = self._maybe_run_test()

        verify_input = (
            f"Original User Request: {user_prompt}\n"
            f"Step Being Executed: {step_description}\n"
            f"Files Modified by Coder: {files_changed}\n"
            f"Coder's Actions and Results: {coder_result}\n\n"
            f"Real Test Output: {test_block}\n"
        )

        self.reset_message()
        result = self.run(verify_input)
        return self._parse_verdict(result["text"])

    def _maybe_run_test(self) -> tuple[str, TestRunResult | None]:
        if not VERIFIER_RUN_TESTS:
            return "[TestRunner] disabled by config", None

        memory_hint_command = None
        memory_hint_timeout = None
        if self.memory is not None:
            context = self.memory.data.get("project_context", {})
            memory_hint_command = context.get("test_command") or None
            memory_hint_timeout = context.get("test_timeout") or None

        result = run_tests(
            workspace=config.WORKSPACE,
            memory_hint_command=memory_hint_command,
            memory_hint_timeout=memory_hint_timeout,
        )

        return result.to_prompt_block(), result

    def _parse_verdict(self, text: str) -> dict:
        """Parse the LLM's passed/failed output into a dict."""
        lines = text.strip().splitlines()
        status = "FAILED"
        reason = ""
        fix_suggestion = ""

        for line in lines:
            line = line.strip()

            if line.upper().startswith("STATUS:"):
                status = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
            elif line.upper().startswith("FIX_SUGGESTION:"):
                fix_suggestion = line.split(":", 1)[1].strip()

        return {
            "passed": status == "PASSED",
            "reason": reason,
            "fix_suggestion": fix_suggestion,
        }
