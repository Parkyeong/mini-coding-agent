"""Planner role: prompt + input building + output parsing.

The actual LLM call goes through an LLMNode instance configured for this
role. engine.build_llm_nodes constructs the LLMNode with PROMPT below; then
engine.run_task calls create_plan(node, ...) to get a list of steps.

This module is config + helpers, not a class. The "planner" runtime object is
the LLMNode instance built in engine.py.
"""

# Brain-facing metadata.
SUPPORTED_TASKS = ["code"]

BRAIN_DESCRIPTION = """planner (LLM worker, gpt-4o-mini):
    Args: user_task (str), memory_context (str, optional), failure_context (str, optional)
    Returns: list of 3-6 plan steps (strings)
    Notes: breaks coding tasks into actionable steps. Each step is meant for
           the coder role to execute.
"""

PROMPT = """You are a task planning expert for a coding agent.
The user will give you a coding task and optionally some project context.

Your job:
1. Break the task into 3-6 clear, actionable steps
2. Each step should be executable by a coding agent that can read/write files and run commands
3. Always include a verification step (run tests, check output, etc.)

Output format: one step per line, numbered, no explanations.

Example:
1. Read the project structure to understand the codebase layout.
2. Read the relevant source files to understand existing implementation.
3. Modify the code to implement the required changes.
4. Run tests to verify the changes work correctly."""


def build_input(user_task: str, memory_context: str = "", failure_context: str = None) -> str:
    parts = []
    if memory_context:
        parts.append(f"Project Context\n{memory_context}")
    if failure_context:
        parts.append(f"Previous Attempt Failed\n{failure_context}")
    parts.append(f"Task\n{user_task}")
    return "\n\n".join(parts)


def parse_plan(text: str) -> list[str]:
    lines = text.strip().splitlines()
    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] in ".、）)":
            line = line[2:].strip()
        elif len(line) > 3 and line[0].isdigit() and line[1].isdigit() and line[2] in ".、":
            line = line[3:].strip()
        steps.append(line)
    return steps if steps else [text.strip()]


def create_plan(node, user_task: str, memory_context: str = "",
                failure_context: str = None) -> list[str]:
    """Run the planner LLMNode once and return parsed steps."""
    node.reset_message()
    result = node.run(build_input(user_task, memory_context, failure_context))
    return parse_plan(result["text"])
