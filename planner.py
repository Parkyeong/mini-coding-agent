import llm
from metrics import MetricsTracker
from config import ENABLE_METRICS
from agent import BaseAgent

PLANNER_PROMPT = """You are a task planning expert for a coding agent.
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


class Planner(BaseAgent):
    def __init__(self, metrics_tracker=None):
        super().__init__(
            system_prompt = PLANNER_PROMPT,
            tools = [],
            max_steps =1,
            metrics_tracker = metrics_tracker,
            agent_role = "planner"

        )


    def create_plan(self, user_task:str,memory_context:str= "", failure_context:str=None)->list[str]:
        """
        Generate a plan.
        Args:
        - user_task: original user request
        - memory_context: historical context from MemoryManager
        - failure_context: if replanning, contains previous failure info
        Return: list of step ["step1", "step2",...]
        """

        prompt_parts = []
        if memory_context:
            prompt_parts.append(f"Project Context\n{memory_context}")
        if failure_context:
            prompt_parts.append(f"Previous Attempt Failed\n{failure_context}")

        prompt_parts.append(f"Task\n{user_task}")
        full_prompt= "\n\n".join(prompt_parts)

        self.reset_message()
        result = self.run(full_prompt)
        return self._parse_plan(result["text"])

    def _parse_plan(self, text:str)->list[str]:
        """Parse step list from LLM output"""
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
