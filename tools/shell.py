"""Shell tool. Delegates to Environment."""

import subprocess


def run_command(env, command: str) -> str:
    try:
        result = env.run_command(command)
    except subprocess.TimeoutExpired as e:
        return f"Command timed out after {e.timeout}s"
    except Exception as e:
        return f"An error occurred: {e}"

    parts = [f"returncode:{result['returncode']}"]
    if result["stdout"]:
        parts.append(f"stdout:{result['stdout']}")
    if result["stderr"]:
        parts.append(f"stderr:{result['stderr']}")
    return "\n".join(parts) if parts else "Command executed successfully with no output"
