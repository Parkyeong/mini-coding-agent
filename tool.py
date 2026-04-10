import os
import subprocess
import config
from config import PROVIDER, COMMAND_TIMEOUT
_memory_manager = None

def safe_path(path:str) -> str:
    "ensure the path is within the workspace"
    if os.path.isabs(path):
        abs_path = os.path.abspath(path)
    else:
        abs_path = os.path.abspath(os.path.join(config.WORKSPACE, path))
    
    workspace_abs = os.path.abspath(config.WORKSPACE)
    
    if os.path.commonpath([abs_path, workspace_abs]) != workspace_abs:
        raise ValueError("Path is outside the workspace")
    
    return abs_path




def read_file(file_path):
    "read file and return content"
    try:
        path = safe_path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            if file_path:
                result = f.read()
                _record_observation("read_file", f"{file_path}: {result[:200]}")
                return result
            else:
                return None
    except FileNotFoundError:
        return "File not found"
    
    except Exception as e:
        return f"An error occurred: {e}"



    
def write_file(file_path, content):
    "write content to file"
    try:
        path = safe_path(file_path) 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
            _record_file_change(file_path)
        return f"Content written successfully to {path}"   
    except Exception as e:
        return f"An error occurred: {e}"
  


def list_dir(dir_path:str = ".") ->str:
    "list files in directory"
    try:
        target = safe_path(dir_path)
        items = os.listdir(target)
        items.sort()

        result = []
        for item in items:
            full_path = os.path.join(target, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR]{item}")
            else:
                result.append(f"[FILE]{item}")

        output = "\n".join(result) if result else "Directory is empty"
        _record_observation("list_dir", f"{dir_path}: {output[:200]}")
        
        return output
    except Exception as e:
        return f"An error occurred: {e}"
    
def run_command(command:str) -> str:
    "run shell command and return output"
    try:
        result = subprocess.run(command, shell=True,cwd=config.WORKSPACE, capture_output=True, text=True,timeout=COMMAND_TIMEOUT)

        output = []
        output.append(f"returncode:{result.returncode}")
        if result.stdout:
            output.append(f"stdout:{result.stdout}")
        if result.stderr:
            output.append(f"stderr:{result.stderr}")
        
        return "\n".join(output) if output else "Command executed successfully with no output"
    except Exception as e:
        return f"An error occurred: {e}"


def search_in_files(keyword:str, dir_path:str = ".") -> str:
    "when only know the keyword but not the specific file, search in all files and return the file name and line number"
    try:
        results = []
        for root, dirs, files in os.walk(safe_path(dir_path)):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except Exception:
                    continue
                for i, line in enumerate(lines):
                    if keyword in line:
                        relative_path = os.path.relpath(file_path, config.WORKSPACE)
                        results.append(f"{relative_path}:{i+1}:{line.strip()}")
        if not results:
            return f"No matches found for '{keyword}'"
        return "\n".join(results)
    except Exception as e:
        return f"An error occurred: {e}"

def replace_in_file(file_path:str, old_text:str, new_text:str) -> str:
    "replace text in file, instead of write whole content"
    try:
        # Guard: empty old_text is dangerous because Python's str.replace("", x)
        # inserts x between every two characters, which has destroyed
        # test_solution.py in past benchmark runs (mbpp_0602 grew to 2281 lines).
        # If the LLM wants to write a file from scratch, it should use write_file.
        if old_text == "":
            return (
                "Error: old_text cannot be empty. "
                "Use write_file to create or fully overwrite a file. "
                "Use replace_in_file only for targeted edits inside an existing file."
            )

        path = safe_path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_text not in content:
            return f"'{old_text}' not found in file."

        # Guard: also catch ambiguous matches (multiple occurrences) to avoid
        # accidentally rewriting all of them when LLM only wanted one.
        occurrences = content.count(old_text)
        if occurrences > 1:
            return (
                f"Error: '{old_text}' appears {occurrences} times in {path}. "
                f"replace_in_file requires a unique match. "
                f"Provide more surrounding context in old_text to make it unique, "
                f"or use write_file to rewrite the whole file."
            )

        updated = content.replace(old_text, new_text)
        with open(path,'w', encoding='utf-8') as f:
            f.write(updated)
            _record_file_change(file_path)

        return f"'{old_text}' replaced with '{new_text}' in {path} successfully"
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"An error occurred: {e}"

# 通用工具定义（provider 无关）
TOOL_DEFINITIONS = [
    {"name": "read_file",
     "description": "Read the content of a file given its path.",
     "parameters": {
         "type": "object",
         "properties": {
             "file_path": {"type": "string", "description": "The path to the file to be read."}},
         "required": ["file_path"]}
    },
    {"name": "write_file",
     "description": "Write content to a file given its path.",
     "parameters": {
         "type": "object",
         "properties": {
             "file_path": {"type": "string", "description": "The path to the file to be written."},
             "content": {"type": "string", "description": "The content to be written to the file."}},
         "required": ["file_path", "content"]}
    },
    {"name": "list_dir",
     "description": "List the files in a directory given its path.",
     "parameters": {
         "type": "object",
         "properties": {
             "dir_path": {"type": "string", "description": "The path to the directory to be listed."}},
         "required": ["dir_path"]}
    },
    {"name": "run_command",
     "description": "Run a shell command and return its output.",
     "parameters": {
         "type": "object",
         "properties": {
             "command": {"type": "string", "description": "The shell command to be executed."}},
         "required": ["command"]}
    },
    {"name": "search_in_files",
     "description": "Search for a keyword in all files within a directory.",
     "parameters": {
         "type": "object",
         "properties": {
             "dir_path": {"type": "string", "description": "The path to the directory to search in."},
             "keyword": {"type": "string", "description": "The keyword to search for."}},
         "required": ["dir_path", "keyword"]}
    },
    {"name": "replace_in_file",
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
             "new_text": {"type": "string", "description": "The replacement text. Must be syntactically consistent with old_text (matching colons, parentheses, etc.)."}},
         "required": ["file_path", "old_text", "new_text"]}
    },
    {
        "name": "save_memory",
        "description": (
        "Record a single, generalizable lesson you've learned during this task. "
        "IMPORTANT: do NOT dump raw file contents or task-specific details. "
        "Compress your observation into ONE sentence that will help future tasks "
        "in similar projects (e.g. 'this project uses pytest with -q' or "
        "'auth tokens are stored in env var AUTH_TOKEN'). "
        "The fact only becomes permanent if the current task passes verification."
    ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description":  "ONE sentence, already summarized and generalized by you."},
                "category": {"type": "string", "description": "Short tag like 'testing', 'config', 'architecture', 'convention'."}},
            "required": ["fact", "category"]}
    }
]


def _build_gemini_tools():
    from google.genai import types
    return [types.Tool(function_declarations=TOOL_DEFINITIONS)]


def _build_openai_tools():
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOL_DEFINITIONS
    ]


def get_tools():
    if PROVIDER == "gemini":
        return _build_gemini_tools()
    else:
        return _build_openai_tools()


def execute_tool(tool_name, input_data):
    "execute tool by name with input data"
    tool_mapping = {
        "read_file": read_file,
        "write_file": write_file,
        "list_dir": list_dir,
        "run_command": run_command,
        "search_in_files": search_in_files,
        "replace_in_file": replace_in_file,
        "save_memory": save_memory
    }

    if tool_name not in tool_mapping:
        return f"Tool '{tool_name}' not found."

    # llm.py wraps JSON parse errors as {"_parse_error": "..."} in input_data
    # so the LLM gets a clear error and can retry with valid arguments.
    if isinstance(input_data, dict) and "_parse_error" in input_data:
        return (
            f"Error: tool arguments could not be parsed as JSON. "
            f"{input_data['_parse_error']}. "
            f"This usually happens when the content field is very long or contains "
            f"unescaped quotes/newlines. Try a shorter or properly escaped payload."
        )

    # Wrap the actual call so that bad LLM input (wrong arg name, wrong type,
    # malformed payload, etc.) becomes a string error returned to the LLM
    # instead of crashing the whole agent run. The LLM can then read the error
    # and self-correct on the next step.
    try:
        return tool_mapping[tool_name](**input_data)
    except TypeError as e:
        # Wrong / unexpected keyword arguments. Tell the LLM what the tool
        # actually accepts so it can fix its next call.
        return (
            f"Error calling {tool_name}: {e}. "
            f"Check the tool's parameter names and types in the tool definition."
        )
    except Exception as e:
        return f"Error calling {tool_name}: {type(e).__name__}: {e}"


def set_memory_manager(manager):
    """Called by main.py at startup to inject MemoryManaager instance"""
    global _memory_manager
    _memory_manager = manager

def save_memory(fact:str, category:str)->str:
    if _memory_manager is None:
        return "Memory manager not initialized."

    wm = _memory_manager.get_working()
    if wm is None:
        return "No active task to associate memory with."

    wm.add_candidate_fact(fact, category)
    return f"Recorded candidate fact: [{category}] {fact} (will be saved if task passes verification)."


def _record_file_change(file_path: str) -> None:
    if _memory_manager is None:
        return
    wm = _memory_manager.get_working()
    if wm is None:
        return
    try:
        rel = os.path.relpath(safe_path(file_path), config.WORKSPACE)
    except Exception:
        rel = file_path
    wm.add_file_changed(rel)

def _record_observation(kind: str, content: str) -> None:
    if _memory_manager is None:
        return
    wm = _memory_manager.get_working()
    if wm is None:
        return
    wm.add_observation(kind=kind, content=content)