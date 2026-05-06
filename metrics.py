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

    def by_role(self) -> dict:
        """Aggregate per-role tokens / call counts. Used by story_task runners
        to compare per-component cost across baseline / Track A / Track B."""
        result: dict = {}
        for c in self.calls:
            r = result.setdefault(
                c.agent_role,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0},
            )
            r["calls"] += 1
            r["input_tokens"] += c.input_tokens
            r["output_tokens"] += c.output_tokens
            r["latency_ms"] += c.latency
        return result
