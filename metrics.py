from typing import List

class CallMetrics:
    def __init__(self, step:int, agent_role:str, input_tokens:int, output_tokens:int, latency:float):
        self.step = step
        self.agent_role = agent_role # planner/actor
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency = latency

class MetricsTracker:
    def __init__(self):
        self.calls: List[CallMetrics] = []

    def record(self, step:int, agent_role:str, input_tokens:int, output_tokens:int, latency:float):
        metric = CallMetrics(step, agent_role, input_tokens, output_tokens, latency)
        self.calls.append(metric)
        print(f"Step {step} | Role: {agent_role} | Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | Latency: {latency:.0f}ms")

    def summary(self):
            if not self.calls:
                return "No metrics recorded."

            total_input = sum(c.input_tokens for c in self.calls)
            total_output = sum(c.output_tokens for c in self.calls)
            total_latency = sum(c.latency for c in self.calls)
            result = [
                "== Metrics Summary ==",
                f"Total Calls: {len(self.calls)}",
                f"Total Input Tokens: {total_input}",
                f"Total Output Tokens: {total_output}",
                f"Total Latency: {total_latency:.0f}ms",
                f"Average Latency per Call: {total_latency/len(self.calls):.0f}ms"
            ]
            return "\n".join(result)
