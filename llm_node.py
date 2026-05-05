"""LLM execution node — the smallest reusable building block of the agent system.

One class, used for every LLM-driven role (planner / coder). It runs the
"prompt -> LLM -> maybe tool calls -> tool results -> repeat" loop up to
max_steps times. Role-specific behaviour (system prompt, tools, input
building, output parsing) lives in the role module under role_pool/
(role_pool/planner.py, role_pool/coder.py, ...), not here. This file only
knows how to iterate.

Naming note: this is the "node" in graph-based agent terminology (cf.
LangGraph). The full *agent system* is engine.py + role_pool + memory +
tool_pool + environment together. An LLMNode by itself is not a complete
agent; it's a single reasoning/action unit.
"""

import llm
from config import MAX_STEPS, ENABLE_METRICS
from metrics import MetricsTracker


# Tool args carrying a workspace path: the loop normalises them to relative
# paths before logging, so files_changed (derived from event_log) deduplicates
# cleanly across absolute / relative variants emitted by the LLM.
_PATH_ARG_TOOLS = ("write_file", "replace_in_file", "read_file")


class LLMNode:
    def __init__(
        self,
        system_prompt: str,
        role: str = "node",
        tools: list = None,
        max_steps: int = MAX_STEPS,
        env=None,
        memory=None,
        metrics_tracker: MetricsTracker = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.system_prompt = system_prompt
        self.role = role
        self.tools = tools if tools is not None else []
        self.max_steps = max_steps
        self.env = env
        self.memory = memory
        self.metrics_tracker = metrics_tracker
        # Optional per-node overrides; None means use config defaults / API
        # defaults. All sourced from config.ROLE_CONFIGS by build_llm_nodes.
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []

    def run(self, input_text: str) -> dict:
        self.messages.append({"role": "user", "content": input_text})

        total_input_tokens = 0
        total_output_tokens = 0
        total_latency = 0
        steps_used = 0

        for step in range(self.max_steps):
            response = llm.chat(
                messages=self.messages,
                system_prompt=self.system_prompt,
                tools=self.tools,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            total_input_tokens += response.get("input_tokens", 0)
            total_output_tokens += response.get("output_tokens", 0)
            total_latency += response.get("latency", 0)
            steps_used = step + 1

            self._log_llm_call(response)

            tool_calls = response.get("tool_calls", [])
            text = response.get("text", "")

            if text:
                self._log_event("text", {"content": text})

            if not tool_calls:
                if text:
                    self.messages.append({"role": "assistant", "content": text})
                self._record_metrics(steps_used, total_input_tokens, total_output_tokens, total_latency)
                return {"text": text, "completed": True}

            self.messages.append(llm.build_assistant_message(response))

            calls_and_results = []
            for call in tool_calls:
                args_for_log = self._normalize_args_for_log(call["name"], call["args"])
                self._log_event("tool_call", {"name": call["name"], "args": args_for_log})

                result = self._execute_tool(call["name"], call["args"])

                self._log_event("tool_result", {
                    "name": call["name"],
                    "args": args_for_log,
                    "result": result,
                })
                calls_and_results.append((call, result))

            self.messages.extend(llm.build_tool_result_message(calls_and_results))

        output_text = f"LLMNode reached max step {self.max_steps} without completing."
        self._record_metrics(steps_used, total_input_tokens, total_output_tokens, total_latency)
        return {"text": output_text, "completed": False}

    def reset_message(self):
        self.messages = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        from tool_pool import execute_tool
        return execute_tool(tool_name, tool_args, env=self.env, memory=self.memory)

    def _normalize_args_for_log(self, name: str, args: dict) -> dict:
        if name not in _PATH_ARG_TOOLS or self.env is None:
            return args
        fp = args.get("file_path")
        if not fp:
            return args
        normalized = dict(args)
        normalized["file_path"] = self.env.relpath(fp)
        return normalized

    def _log_event(self, kind: str, payload: dict) -> None:
        if self.memory is None:
            return
        wm = self.memory.get_working()
        if wm is None:
            return
        wm.event_log.append(kind, payload)

    def _log_llm_call(self, response: dict) -> None:
        self._log_event("llm_call", {
            "role": self.role,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "latency": response.get("latency", 0),
        })

    def _record_metrics(self, steps: int, input_tokens: int, output_tokens: int, latency: float):
        if ENABLE_METRICS and self.metrics_tracker:
            self.metrics_tracker.record(
                step=steps,
                agent_role=self.role,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
            )
