"""The agent loop. One class, used for every role (planner / coder / verifier).

Role behaviour (system prompt, tools, max_steps, input building, output
parsing) lives in the corresponding role module (planner.py / coder.py /
verifier.py). This file only knows how to iterate: call LLM, dispatch tools,
push events into working memory, stop when the LLM has nothing else to do.
"""

import llm
from config import MAX_STEPS, ENABLE_METRICS
from metrics import MetricsTracker


# Tool args carrying a workspace path: agent loop normalises them to relative
# paths before logging, so files_changed (derived from event_log) deduplicates
# cleanly across absolute / relative variants emitted by the LLM.
_PATH_ARG_TOOLS = ("write_file", "replace_in_file", "read_file")


class Agent:
    def __init__(
        self,
        system_prompt: str,
        role: str = "agent",
        tools: list = None,
        max_steps: int = MAX_STEPS,
        env=None,
        memory=None,
        metrics_tracker: MetricsTracker = None,
    ):
        self.system_prompt = system_prompt
        self.role = role
        self.tools = tools if tools is not None else []
        self.max_steps = max_steps
        self.env = env
        self.memory = memory
        self.metrics_tracker = metrics_tracker
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

        output_text = f"agent reached max step {self.max_steps} without completing."
        self._record_metrics(steps_used, total_input_tokens, total_output_tokens, total_latency)
        return {"text": output_text, "completed": False}

    def reset_message(self):
        self.messages = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        from tools import execute_tool
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
