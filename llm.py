"""LLM client. Single provider: OpenRouter (OpenAI-compatible API)."""

import json
import time
from typing import Optional

from openai import OpenAI

from config import MODEL, API_KEY, BASE_URL


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy client init so importing llm.py doesn't crash when env isn't set yet."""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Export it before running, e.g. `export OPENROUTER_API_KEY=...`."
            )
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def chat(messages: list, system_prompt: str, tools: list) -> dict:
    """One LLM round-trip. Returns text, tool_calls, tokens, latency."""
    full_messages = _to_openai_messages(messages, system_prompt)
    openai_tools = _to_openai_tools(tools) if tools else None

    kwargs = {"model": MODEL, "messages": full_messages}
    if openai_tools:
        kwargs["tools"] = openai_tools

    t_start = time.perf_counter()
    response = _get_client().chat.completions.create(**kwargs)
    latency = (time.perf_counter() - t_start) * 1000

    msg = response.choices[0].message
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            # The LLM occasionally outputs malformed JSON for tool arguments
            # (especially when content is very long). Don't crash the run —
            # surface the parse error as a synthetic arg dict so the dispatcher
            # can return an error message and let the LLM retry.
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                args = {
                    "_parse_error": (
                        f"JSON decode failed at position {e.pos}: {e.msg}. "
                        f"raw arguments length={len(tc.function.arguments)}"
                    )
                }
            tool_calls.append({
                "name": tc.function.name,
                "args": args,
                "id": tc.id,
            })

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    return {
        "text": msg.content or "",
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": latency,
    }


def build_assistant_message(response: dict) -> dict:
    """Unified assistant message from chat() response."""
    msg = {"role": "assistant", "content": response.get("text", "")}
    if response.get("tool_calls"):
        msg["tool_calls"] = response["tool_calls"]
    return msg


def build_tool_result_message(call_and_results: list) -> list:
    """Unified tool result messages.

    input:  [(call_info, result), ...]
    output: [{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}, ...]
    """
    return [
        {
            "role": "tool",
            "tool_call_id": call_info["id"],
            "name": call_info["name"],
            "content": result,
        }
        for call_info, result in call_and_results
    ]


def _to_openai_messages(messages: list, system_prompt: str) -> list:
    """Convert unified messages -> OpenAI format."""
    converted = []
    if system_prompt:
        converted.append({"role": "system", "content": system_prompt})

    for m in messages:
        if m["role"] == "user":
            converted.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if "tool_calls" in m and m["tool_calls"]:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            converted.append(msg)
        elif m["role"] == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "content": m["content"],
            })
    return converted


def _to_openai_tools(tools: list) -> list:
    """Wrap tool definitions in OpenAI format if not already."""
    result = []
    for t in tools:
        if "type" in t and t["type"] == "function":
            result.append(t)
        else:
            result.append({"type": "function", "function": t})
    return result
