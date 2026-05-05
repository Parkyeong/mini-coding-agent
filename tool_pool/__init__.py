"""tool_pool — pool of callable utilities used by the agent system.

Two coexisting protocols, kept separate by usage convention (not by directory):

  LLM-facing tools (re-exported here, dispatched via execute_tool):
    ops.py — read_file / write_file / list_dir / search_in_files /
             replace_in_file / run_command
    These return strings to the LLM and never write to memory; the agent loop
    pushes events into WorkingMemory.event_log after each tool call.

  Python-facing helpers (NOT re-exported; imported directly by callers):
    test_runner.py — test command discovery + result wrapper, used by verifier.
                     Import as: `from tool_pool.test_runner import run_tests`.

The split is intentional: LLM tools speak JSON in/string out; Python helpers
return rich objects. Putting both in one directory means "all callable
utilities live here", but `__init__.py` only exposes the LLM protocol so the
LLM dispatcher never sees Python helpers.
"""

from tool_pool.ops import (
    read_file,
    write_file,
    list_dir,
    search_in_files,
    replace_in_file,
    run_command,
)


TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the content of a file given its path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to be read."}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file given its path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to be written."},
                "content": {"type": "string", "description": "The content to be written to the file."},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the files in a directory given its path.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "The path to the directory to be listed."}
            },
            "required": ["dir_path"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to be executed."}
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_in_files",
        "description": "Search for a keyword in all files within a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "The path to the directory to search in."},
                "keyword": {"type": "string", "description": "The keyword to search for."},
            },
            "required": ["dir_path", "keyword"],
        },
    },
    {
        "name": "replace_in_file",
        "description": (
            "Replace one specific occurrence of old_text with new_text in a file. "
            "Use this for small targeted edits inside an existing file. "
            "RULES: (1) old_text MUST appear EXACTLY ONCE in the file — include "
            "enough surrounding context to make it unique. (2) old_text cannot be "
            "empty (use write_file to create new files). (3) Make sure old_text and "
            "new_text are syntactically self-contained — if old_text ends with `:` "
            "then new_text must also end with `:`, otherwise you will end up with "
            "double colons or missing colons. (4) If you need to rewrite an entire "
            "file or create a new one, use write_file instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to edit."},
                "old_text": {"type": "string", "description": "The exact text to be replaced. Must appear exactly once in the file."},
                "new_text": {"type": "string", "description": "The replacement text. Must be syntactically consistent with old_text (matching colons, parentheses, etc.)."},
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    },
]


# Tools that need an Environment to execute, indexed by name.
_ENV_TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_command": run_command,
    "search_in_files": search_in_files,
    "replace_in_file": replace_in_file,
}


def execute_tool(name: str, args: dict, *, env=None, memory=None) -> str:
    """Dispatch a tool call.

    Args:
        name: tool name as defined in TOOL_DEFINITIONS.
        args: parsed JSON args from the LLM.
        env: Environment instance (required for fs/shell tools).
        memory: kept for backward-compat; currently unused (no memory tools).
    """
    # llm.py wraps JSON parse errors as {"_parse_error": "..."} so the LLM
    # gets a clear message and can retry with valid arguments.
    if isinstance(args, dict) and "_parse_error" in args:
        return (
            f"Error: tool arguments could not be parsed as JSON. "
            f"{args['_parse_error']}. "
            f"This usually happens when the content field is very long or contains "
            f"unescaped quotes/newlines. Try a shorter or properly escaped payload."
        )

    try:
        if name in _ENV_TOOLS:
            if env is None:
                return f"Error calling {name}: no environment configured for this agent."
            return _ENV_TOOLS[name](env, **args)

        return f"Tool '{name}' not found."

    except TypeError as e:
        return (
            f"Error calling {name}: {e}. "
            f"Check the tool's parameter names and types in the tool definition."
        )
    except Exception as e:
        return f"Error calling {name}: {type(e).__name__}: {e}"


def get_tools():
    """Return tool definitions in OpenAI/OpenRouter format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOL_DEFINITIONS
    ]
