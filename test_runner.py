"""
Test runner for the verifier.

Domain logic only — test_command discovery (memory hint -> marker -> none),
result-shape (TestRunResult), output truncation. The actual subprocess call
is delegated to Environment.run_command, so when we add a DockerEnvironment
later, test_runner doesn't need to know.
"""

import os
import subprocess
from typing import Optional

from config import VERIFIER_TEST_TIMEOUT_DEFAULT, VERIFIER_OUTPUT_MAX_CHARS


class TestRunResult:
    def __init__(
        self,
        executed: bool,
        command: Optional[str],
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        timed_out: bool,
        backend: str,                   # "subprocess" | "skipped"
        detection_source: str,          # "memory" | "marker" | "none"
        error: Optional[str] = None,
    ):
        self.executed = executed
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.backend = backend
        self.detection_source = detection_source
        self.error = error

    def passed(self) -> bool:
        return self.executed and not self.timed_out and self.returncode == 0

    def to_prompt_block(self) -> str:
        "Format the test result into a string block for LLM input."
        if not self.executed:
            return f"[TestRunner] skipped (source={self.detection_source}): {self.error or 'no test command available'}"

        head = (
            f"[TestRunner] backend={self.backend} command={self.command} "
            f"source={self.detection_source} returncode={self.returncode} timed_out={self.timed_out}"
        )

        return f"{head}\n--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}\n--- end ---\n"

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "backend": self.backend,
            "detection_source": self.detection_source,
            "error": self.error,
        }


def _truncate(text: str, limit: int = VERIFIER_OUTPUT_MAX_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [truncated {len(text) - limit} chars] ...\n" + text[-half:]


# ---------------------------------------------------------------------------
# Marker-based fallback detection (Python only, per project decision)
# ---------------------------------------------------------------------------
def marker_based_test_detection(workspace: str) -> Optional[str]:
    """Last-resort detection. Returns None if nothing matches.

    Note: we deliberately do NOT scan workspace root for stray test_*.py files.
    Per project decision, dataset outputs (MBPP / SWE-bench) live in
    dedicated subfolders, so a loose test_*.py at workspace root must not
    auto-trigger pytest.
    """
    if os.path.exists(os.path.join(workspace, "pytest.ini")):
        return "pytest -q"
    if os.path.exists(os.path.join(workspace, "pyproject.toml")):
        return "pytest -q"
    if os.path.isdir(os.path.join(workspace, "tests")):
        return "pytest -q"
    return None


# ---------------------------------------------------------------------------
# High-level entry point used by Verifier
# ---------------------------------------------------------------------------

def run_tests(
    env,
    memory_hint_command: Optional[str] = None,
    memory_hint_timeout: Optional[int] = None,
) -> TestRunResult:
    """Resolve the test command (memory -> marker -> none) and run it through env.

    env is the same Environment instance the coder used; this guarantees tests
    run against the exact files coder modified, and lets us swap to a Docker
    backend later by changing only the env class.
    """
    timeout = memory_hint_timeout or VERIFIER_TEST_TIMEOUT_DEFAULT

    if memory_hint_command:
        command, source = memory_hint_command.strip(), "memory"
    else:
        detected = marker_based_test_detection(env.workspace)
        if detected:
            command, source = detected, "marker"
        else:
            return TestRunResult(
                executed=False,
                command=None,
                returncode=None,
                stdout="",
                stderr="",
                timed_out=False,
                backend="skipped",
                detection_source="none",
                error="No test command found in memory or via marker detection",
            )

    backend = getattr(env, "backend_name", "unknown")

    try:
        result = env.run_command(command, timeout=timeout)
        return TestRunResult(
            executed=True,
            command=command,
            returncode=result["returncode"],
            stdout=_truncate(result["stdout"]),
            stderr=_truncate(result["stderr"]),
            timed_out=False,
            backend=backend,
            detection_source=source,
        )
    except subprocess.TimeoutExpired as e:
        return TestRunResult(
            executed=True,
            command=command,
            returncode=None,
            stdout=_truncate(e.stdout) if e.stdout else "",
            stderr=_truncate(e.stderr) if e.stderr else "",
            timed_out=True,
            backend=backend,
            detection_source=source,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        return TestRunResult(
            executed=False,
            command=command,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=False,
            backend=backend,
            detection_source=source,
            error=f"{type(e).__name__}: {e}",
        )
