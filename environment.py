"""
Execution environment.

A stateful sandbox: each task gets one Environment that knows its workspace
and how to perform primitive operations on it (read / write / list / shell).
tool_pool/ops.py delegates to this; tool_pool/test_runner.py uses it too.

Currently only a local subprocess backend is implemented. A docker / remote
backend can later be added by swapping the run_command / read_file / write_file
implementations without touching tool_pool/ or engine.py.
"""

import os
import subprocess
from typing import Optional


class Environment:
    # Used by test_runner / verifier to label which backend ran a test.
    # Subclasses (e.g. DockerEnvironment) override this to identify themselves.
    backend_name = "subprocess"

    def __init__(self, workspace: str, command_timeout: int = 20,
                 protected_files: Optional[list[str]] = None):
        self.workspace = os.path.abspath(workspace)
        self.command_timeout = command_timeout
        # Files (matched by basename) that are read-only for the agent. Used
        # by benchmark runners to lock the official grading test file so the
        # agent cannot tamper with it during the run. Tool-level only; a
        # determined agent could still bypass via run_command shell tricks.
        self.protected_files: set[str] = set(protected_files or [])

    def safe_path(self, path: str) -> str:
        if os.path.isabs(path):
            abs_path = os.path.abspath(path)
        else:
            abs_path = os.path.abspath(os.path.join(self.workspace, path))

        if os.path.commonpath([abs_path, self.workspace]) != self.workspace:
            raise ValueError("Path is outside the workspace")
        return abs_path

    def relpath(self, path: str) -> str:
        try:
            return os.path.relpath(self.safe_path(path), self.workspace)
        except Exception:
            return path

    def read_file(self, file_path: str) -> str:
        path = self.safe_path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, file_path: str, content: str) -> None:
        path = self.safe_path(file_path)
        if os.path.basename(path) in self.protected_files:
            raise PermissionError(
                f"'{os.path.basename(path)}' is read-only (benchmark grading file). "
                f"Modify solution.py instead."
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def list_dir(self, dir_path: str = ".") -> list[dict]:
        target = self.safe_path(dir_path)
        items = sorted(os.listdir(target))
        return [
            {"name": n, "is_dir": os.path.isdir(os.path.join(target, n))}
            for n in items
        ]

    def walk(self, dir_path: str = "."):
        for root, dirs, files in os.walk(self.safe_path(dir_path)):
            yield root, dirs, files

    def run_command(self, command: str, timeout: Optional[int] = None) -> dict:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=timeout or self.command_timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
