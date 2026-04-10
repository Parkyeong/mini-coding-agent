"""
Load the MBPP dataset and materialize each instance into a standardized
workspace under WORKSHOP, with a per-instance memory.json pre-populated
with the test_command.

Dataset: google-research-datasets/mbpp, subset='sanitized' (427 hand-verified
problems). We use the 'test' split (257 problems, task_id 11~510) for
benchmark evaluation. Field naming note: sanitized uses 'prompt' for the
task description (full uses 'text'); other fields (task_id, code, test_list)
are identical across both subsets.
"""
import os
import json
import argparse
from datasets import load_dataset

from config import WORKSHOP


SOLUTION_STUB = '''"""MBPP task — implement the function described in prompt.md."""
'''


def materialize_instance(task_id: int, text: str, code: str, test_list: list[str]) -> str:
    workspace = os.path.join(WORKSHOP, f"mbpp_{task_id:04d}")
    os.makedirs(workspace, exist_ok=True)

    # solution.py: empty stub (the agent will fill it in)
    with open(os.path.join(workspace, "solution.py"), "w", encoding="utf-8") as f:
        f.write(SOLUTION_STUB)

    # test_solution.py: import from solution and run MBPP asserts
    test_body = "from solution import *\n\n"
    for i, t in enumerate(test_list):
        test_body += f"def test_case_{i}():\n    {t}\n\n"
    with open(os.path.join(workspace, "test_solution.py"), "w", encoding="utf-8") as f:
        f.write(test_body)

    # prompt.md: human task description + the test cases (so the agent
    # knows the exact function name and signature). Without showing the
    # tests, MBPP problems often have ambiguous function names (e.g.
    # "lucid numbers" in prose vs `get_ludic` in tests for task 603).
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

    # memory.json: pre-fill test_command so verifier hits the first fallback
    memory = {
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
    with open(os.path.join(workspace, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    return workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "prompt"])
    parser.add_argument("--limit", type=int, default=10, help="0 = all")
    args = parser.parse_args()

    print(f"Loading MBPP split={args.split} ...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=args.split)

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"Materializing {n} instances ...")

    for i in range(n):
        row = ds[i]
        ws = materialize_instance(
            task_id=row["task_id"],
            text=row["prompt"],          # sanitized split uses "prompt"
            code=row["code"],
            test_list=row["test_list"],
        )
        print(f"  [{i+1}/{n}] {ws}")

    print("Done.")


if __name__ == "__main__":
    main()