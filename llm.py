import json
import time
from config import PROVIDER, MODEL, API_KEY, BASE_URL
from openai import OpenAI

# Note: google-genai is imported lazily inside the Gemini class so that
# users running PROVIDER="local" / "openai" / "claude" don't need to install
# google-genai at all.


def build_assistant_message(response:dict)->dict:
    """Build a unified assistant message from chat() response"""
    message = {
        "role": "assistant",
        "content": response.get("text", "")}

    if response.get("tool_calls"):
        message["tool_calls"] = response["tool_calls"]

    return message

def build_tool_result_message(call_and_results:list[tuple[dict,str]])->list[dict]:
    """Build unified tool result message.
    input:[(call_info1, result1),(call_info2, result2),...]
    output:[{"role":"tool", "tool_call_id":"", "name":"..", "content":"..."}]"""
    message = []
    for call_info, result in call_and_results:
        message.append({
            "role":"tool",
            "tool_call_id": call_info["id"],
            "name": call_info["name"],
            "content": result
        })
    return message

#Base class
class BaseLLM:
    def chat(self, messages:list, system_prompt:str, tools:list) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


#Gemini client
class Gemini(BaseLLM):
    def __init__(self):
        # Lazy import: only load google-genai when actually using gemini.
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=API_KEY)

    def chat(self, messages:list,system_prompt:str, tools:list) -> dict:
        from google.genai import types

        gemini_messages = self._to_gemini_messages(messages)

        t_start = time.perf_counter()

        response = self.client.models.generate_content(
            model=MODEL,
            contents=gemini_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools)
        )

        t_end = time.perf_counter()
        latency = (t_end - t_start) * 1000

        tool_calls = []
        text_parts = []

        parts = response.candidates[0].content.parts
        for index, part in enumerate(parts):
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append({
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args),
                    "id": f"call_{index}"})
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)


        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_tokens_count", 0) or 0
        output_tokens = getattr(usage, "candidates_tokens_count", 0) or 0

        return {
            "text": "\n".join(text_parts).strip(),
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency
        }

    def _to_gemini_messages(self, messages:list) -> list:
        """
        Convert unified messages -> gemini format.
        key rules:
        "assistant" -> "model", "user/tool"->"user"
        assistant with tool_calls->model with function_call parts
        tool results -> user with function_response parts
        consecutive tool results must be merge into one user message
        """
        converted = []
        i = 0
        while i<len(messages):
            m = messages[i]

            if m["role"] == "user":
                converted.append({
                    "role":"user",
                    "parts":[{"text":m["content"]}]
                })
                i += 1

            elif m["role"] == "assistant":
                if "tool_calls" in m and m["tool_calls"]:
                    parts = []
                    if m.get("content"):
                        parts.append({"text": m["content"]})
                    for tc in m["tool_calls"]:
                        parts.append({
                            "function_call":{
                                "name": tc["name"],
                                "args": json.dumps(tc["args"])
                            }
                        })
                    converted.append({
                        "role":"model",
                        "parts": parts
                    })

                else:
                    converted.append({
                        "role":"model",
                        "parts":[{"text":m.get("content", "")}]
                    })
                i += 1

            elif m["role"] == "tool":
                parts = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    parts.append({
                        "function_response":{
                            "name": messages[i]["name"],
                            "response": messages[i]["content"]
                        }
                    })
                    i += 1
                converted.append({"role":"user","parts": parts})
            else:
                i += 1
        return converted



class OpenAICompatible(BaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

    def chat(self, messages: list, system_prompt: str, tools: list) -> dict:
        full_messages = self._to_openai_messages(messages, system_prompt)
        openai_tools = self._to_openai_tools(tools) if tools else None

        t_start = time.perf_counter()

        kwargs = {
            "model": MODEL,
            "messages": full_messages,
        }
        # `enable_thinking` is a Qwen-specific extension supported by local
        # inference servers (vLLM/SGLang). Official OpenAI API rejects
        # unknown arguments with 400, so only pass it for local provider.
        if PROVIDER == "local":
            kwargs["extra_body"] = {"enable_thinking": False}
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t_start)*1000

        message = response.choices[0].message
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments)
                tool_calls.append({
                    "name": tc.function.name,
                    "args": args,
                    "id": tc.id,
                })
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        return {
            "text": message.content or "",
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency
        }

    def _to_openai_messages(self, messages: list, system_prompt: str) -> list:
        """Convert unified messages -> OpenAI format."""
        converted = []
        if system_prompt:
            converted.append({"role":"system", "content":system_prompt})

        for m in messages:
            if m["role"] == "user":
                converted.append({"role":"user", "content":m["content"]})
            elif m["role"] == "assistant":
                msg = {"role":"assistant", "content":m.get("content") or ""}
                if "tool_calls" in m and m["tool_calls"]:
                    msg["tool_calls"] = [{
                        "id":tc['id'],
                        "type":"function",
                        "function":{
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    } for tc in m["tool_calls"]]
                converted.append(msg)
            elif m["role"] == "tool":
                converted.append({
                    "role":"tool",
                    "tool_call_id":m['tool_call_id'],
                    "content": m["content"],
                })

        return converted


    def _to_openai_tools(self, tools:list)->list:
        """Wrap tool definitions in OpenAI format"""
        result = []
        for t in tools:
            if "type" in t and t["type"] == "function":
                result.append(t)
            else:
                result.append({
                    'type':'function',
                    "function":t
                })
        return result



def get_llm_client()->BaseLLM:
    if PROVIDER == "gemini":
        return Gemini()
    elif PROVIDER in ["openai", "claude", "local"]:
        return OpenAICompatible()
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")


_client = get_llm_client()

def chat(messages:list, system_prompt:str, tools:list) -> dict:
    return _client.chat(messages, system_prompt, tools)
