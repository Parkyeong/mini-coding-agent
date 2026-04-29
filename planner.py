"""Planner role: prompt + input building + output parsing.

The actual LLM call goes through an LLMNode instance configured for this
role. engine.build_llm_nodes constructs the LLMNode with PROMPT below; then
engine.run_task calls create_plan(node, ...) to get a list of steps.

This module is config + helpers, not a class. The "planner" runtime object is
the LLMNode instance built in engine.py.
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


# ---------------------------------------------------------------------------
# c2_planspec variant: planner also extracts a natural-language rule from
# the test cases before listing steps. Used by the c2_planspec experimental
# config in engine.run_task. The rule is then passed to coder so the coder
# implements based on the rule, not the (potentially ambiguous) prompt text.
# ---------------------------------------------------------------------------

PROMPT_WITH_SPEC = """You are a task planning expert for a coding agent.

The user will give you a coding task that includes a natural-language
description AND a few test assertions. The natural-language description
may be ambiguous, vague, or even slightly inaccurate; the test assertions
are the ground truth for the function's behavior.

Your job has TWO parts:

PART 1 — Extract a clear rule from the tests (spec extraction):
  - Read the test assertions carefully
  - Manually reason about what input → output transformation they imply
  - Write a 1-3 sentence general rule that explains all test cases
  - If the natural-language description and the tests disagree, the tests
    win — describe what the tests actually require, not what the prose says
  - Do NOT hardcode specific input/output pairs ("if input==X return Y");
    write a *generalizable* rule
  - Walk through one test case to prove your rule produces the expected
    output

PART 2 — List concrete steps for the coder:
  - 1-3 actionable steps (most simple tasks need only 1)
  - Reference the rule explicitly so coder knows what to implement
  - Always include a verification step (run tests)

Output format (in order, no extra prose around it):

RULE: <1-3 sentences describing the function's actual behavior>
EXAMPLES_WALK: <walk through one test using the rule>
STEPS:
1. ...
2. ...
"""


def parse_plan_with_spec(text: str) -> tuple[str, str, list[str]]:
    """Parse the planner_v2 output into (rule, examples_walk, steps).

    Tolerant: if any section is missing the parser falls back to plausible
    defaults so the run can still proceed.
    """
    rule = ""
    walk = ""
    steps_text = ""

    # Greedy line-walk; section starts when we see RULE: / EXAMPLES_WALK: / STEPS:
    section = None
    buf: dict[str, list[str]] = {"RULE": [], "EXAMPLES_WALK": [], "STEPS": []}
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("RULE:"):
            section = "RULE"
            rest = stripped[len("RULE:"):].strip()
            if rest:
                buf["RULE"].append(rest)
            continue
        if upper.startswith("EXAMPLES_WALK:") or upper.startswith("EXAMPLE_WALK:"):
            section = "EXAMPLES_WALK"
            rest = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if rest:
                buf["EXAMPLES_WALK"].append(rest)
            continue
        if upper.startswith("STEPS:"):
            section = "STEPS"
            rest = stripped[len("STEPS:"):].strip()
            if rest:
                buf["STEPS"].append(rest)
            continue
        if section is not None and stripped:
            buf[section].append(stripped)

    rule = " ".join(buf["RULE"]).strip()
    walk = " ".join(buf["EXAMPLES_WALK"]).strip()
    steps_text = "\n".join(buf["STEPS"]).strip()

    steps = parse_plan(steps_text) if steps_text else []
    if not steps:
        # Defensive: if STEPS section was missing, treat the whole rule as
        # a one-step plan so the run proceeds.
        steps = ["Implement the function in solution.py based on the rule."]

    return rule, walk, steps


def create_plan_with_spec(node, user_task: str, memory_context: str = "",
                          failure_context: str = None
                          ) -> tuple[str, str, list[str]]:
    """c2_planspec entry point. Returns (rule, examples_walk, steps)."""
    node.reset_message()
    result = node.run(build_input(user_task, memory_context, failure_context))
    return parse_plan_with_spec(result.get("text", ""))
