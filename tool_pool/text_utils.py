"""Text utility tools — pure Python functions, no LLM, no env dependency.

These are dispatchable utilities for orchestrating roles. They live in
tool_pool/ alongside ops.py but are NOT exposed via tool_pool/__init__.py's
LLM dispatch table — callers import them directly:

    from tool_pool.text_utils import length_checker

Brain (via role_pool/brain.py) discovers brain-facing tools here by reading
the BRAIN_TOOLS dict at the bottom of this module.
"""


def length_checker(text: str, target: int = None) -> dict:
    """Return exact character count of text. Optionally compare to a target.

    Args:
        text: text to measure
        target: optional target length; if given, return diff info

    Returns:
        With target=None:
            {"length": <int>}
        With target=<int>:
            {"length": <int>, "target": <int>, "diff": <int>,
             "hit": <bool>, "delta_text": <str>}
            where delta_text is "exactly on target" / "too long by N" / "too short by N"
    """
    actual = len(text)
    if target is None:
        return {"length": actual}
    diff = actual - target
    return {
        "length": actual,
        "target": target,
        "diff": diff,
        "hit": diff == 0,
        "delta_text": (
            "exactly on target" if diff == 0
            else f"too long by {diff}" if diff > 0
            else f"too short by {-diff}"
        ),
    }


# ---------------------------------------------------------------------------
# Brain-facing tool registry
# ---------------------------------------------------------------------------
# Each entry maps a tool name (the dispatch key brain would use in its plan)
# to metadata: supported_tasks (which task types brain may see this tool in),
# description (the docstring brain reads), and callable (the actual function).
# role_pool/brain.py iterates this to build brain's tool menu.
BRAIN_TOOLS = {
    "length_checker": {
        "supported_tasks": ["story"],
        "description": """length_checker (Python function, no LLM):
    Args: text (str, required), target (int, optional)
    Returns: dict with length / diff / hit fields
    Notes: returns exact character count via Python len().""",
        "callable": length_checker,
    },
}
