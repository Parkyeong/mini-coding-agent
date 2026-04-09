import llm
import tool as Tools
from metrics import MetricsTracker
from config import ENABLE_METRICS, MAX_STEPS
from agent import BaseAgent


CODER_PROMPT = """
You are a professional coding agent.
**Your scope is strictly limited to the workspace. You cannot access anything outside of the workspace.**

Workflow:
1. Understand what needs to be done
2. Use tools to read files, understand context
3. Make the necessary changes
4. Run commands to verify your changes
5. If you discover useful facts about the project, use save_memory to record them

Rules:
- Prefer minimal change.If a local replacement is enough, do not rewrite the whole file.
- Always try to verify your changes by running relevant commands.
- In your final response: summarize what you changed, what tools you used, and verification results."""

class Coder(BaseAgent):
    def __init__(self, metrics_tracker=None, memory=None):
        super().__init__(
            system_prompt = CODER_PROMPT,
            tools = Tools.get_tools(),
            metrics_tracker = metrics_tracker,
            agent_role = "coder"
        )
        self.memory = memory

    def run(self, input_text:str) ->dict:
        if self.memory is not None:
            wm = self.memory.get_working()
            if wm is not None:
                snapshot = wm.snapshot_for_coder()
                if snapshot:
                    input_text = f"{snapshot}\n\n current step:{input_text}"

        return super().run(input_text)

