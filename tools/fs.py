"""Filesystem tools. Pure: only depend on Environment, no memory writes."""

import os


def read_file(env, file_path: str) -> str:
    try:
        return env.read_file(file_path)
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"An error occurred: {e}"


def write_file(env, file_path: str, content: str) -> str:
    try:
        env.write_file(file_path, content)
        return f"Content written successfully to {env.safe_path(file_path)}"
    except PermissionError as e:
        return (
            f"Refused: {e} This file is locked by the benchmark and cannot be "
            f"written, replaced, or deleted. Do not retry — modify solution.py instead."
        )
    except Exception as e:
        return f"An error occurred: {e}"


def list_dir(env, dir_path: str = ".") -> str:
    try:
        items = env.list_dir(dir_path)
    except Exception as e:
        return f"An error occurred: {e}"

    if not items:
        return "Directory is empty"

    lines = []
    for it in items:
        lines.append(f"[DIR]{it['name']}" if it["is_dir"] else f"[FILE]{it['name']}")
    return "\n".join(lines)


def search_in_files(env, keyword: str, dir_path: str = ".") -> str:
    try:
        results = []
        for root, _dirs, files in env.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception:
                    continue
                for i, line in enumerate(lines):
                    if keyword in line:
                        rel = os.path.relpath(file_path, env.workspace)
                        results.append(f"{rel}:{i + 1}:{line.strip()}")
        if not results:
            return f"No matches found for '{keyword}'"
        return "\n".join(results)
    except Exception as e:
        return f"An error occurred: {e}"


def replace_in_file(env, file_path: str, old_text: str, new_text: str) -> str:
    # Guard: empty old_text destroys files (Python's str.replace("", x) inserts x
    # between every two characters). Past benchmark runs hit this — never allow it.
    if old_text == "":
        return (
            "Error: old_text cannot be empty. "
            "Use write_file to create or fully overwrite a file. "
            "Use replace_in_file only for targeted edits inside an existing file."
        )

    try:
        content = env.read_file(file_path)
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"An error occurred: {e}"

    if old_text not in content:
        return f"'{old_text}' not found in file."

    occurrences = content.count(old_text)
    if occurrences > 1:
        return (
            f"Error: '{old_text}' appears {occurrences} times in {file_path}. "
            f"replace_in_file requires a unique match. "
            f"Provide more surrounding context in old_text to make it unique, "
            f"or use write_file to rewrite the whole file."
        )

    updated = content.replace(old_text, new_text)
    try:
        env.write_file(file_path, updated)
    except PermissionError as e:
        return (
            f"Refused: {e} This file is locked by the benchmark and cannot be "
            f"written, replaced, or deleted. Do not retry — modify solution.py instead."
        )
    except Exception as e:
        return f"An error occurred: {e}"

    return f"'{old_text}' replaced with '{new_text}' in {env.safe_path(file_path)} successfully"
