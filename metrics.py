from typing import List


class CallMetrics:
    def __init__(self, step: int, agent_role: str, input_tokens: int,
                 output_tokens: int, latency: float):
        self.step = step
        self.agent_role = agent_role
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency = latency


class MetricsTracker:
    def __init__(self):
        self.calls: List[CallMetrics] = []

    def record(self, step: int, agent_role: str, input_tokens: int,
               output_tokens: int, latency: float):
        self.calls.append(CallMetrics(step, agent_role, input_tokens, output_tokens, latency))

    def summary(self) -> str:
        """One-line summary suitable for end-of-task output."""
        if not self.calls:
            return "metrics: no calls"

        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_latency_s = sum(c.latency for c in self.calls) / 1000.0
        return (
            f"metrics: {len(self.calls)} calls, "
            f"{total_input}/{total_output} in/out tokens, "
            f"{total_latency_s:.1f}s"
        )
