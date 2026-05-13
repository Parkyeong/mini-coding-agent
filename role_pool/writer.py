"""Writer role: flash-fiction creative writing under a strict length constraint.

The actual LLM call goes through an LLMNode instance configured for this
role. Used by the story_task runners (baseline / engine_fixed / engine_brain).
The worker model is locked to gpt-4o-mini in ROLE_CONFIGS — orchestrating
roles (brain) may use stronger models, but the prose itself is always
4o-mini for fair cross-method comparison.
"""

# ---------------------------------------------------------------------------
# Brain-facing metadata
# ---------------------------------------------------------------------------
# Used by role_pool/brain.py's get_visible_roles() to filter the pool menu
# shown to brain by task_type. The brain doesn't see roles whose task type
# doesn't match the current task.
SUPPORTED_TASKS = ["story"]

BRAIN_DESCRIPTION = """writer (LLM worker, gpt-4o-mini):
    Args: theme (str, required), guidance (str, optional)
    Returns: draft story text
    Notes: tends to over/undershoot length targets by 5-30 characters.
"""

PROMPT = """You are a flash fiction writer.

Task: Write an English story of EXACTLY 241 characters.

CRITICAL LENGTH CONSTRAINT:
- The output must be EXACTLY 241 characters — not 240, not 242.
- Every character counts: letters, spaces, punctuation, line breaks.
- Before finalizing, count your characters and adjust until it is exactly 241.

Other requirements:
- Complete story arc (setup, conflict, resolution).
- Output ONLY the story body — no title, no preamble, no explanation, no
  surrounding quotes, no prefix like "Here is a story:".
"""


def build_input(theme: str, guidance: str = "", feedback: str = "",
                previous_attempt: str = "") -> str:
    """Build the user-message text fed into the writer LLMNode.

    `guidance`: optional hint set by an orchestrator (e.g. brain in
        method_fixed / method_brain). Examples: "aim for 230 chars, be
        concise". Empty in baseline (no orchestrator).
    `feedback`: length-error message from the previous attempt, e.g.
        "Previous attempt was 256 characters, too long by 15. Adjust to
        exactly 241." Empty on the first attempt.
    `previous_attempt`: the FULL TEXT of the previous attempt's story.
        When non-empty, writer is asked to revise this draft to meet the
        length target rather than write a fresh story from scratch. This
        gives precise character-level control (trim/expand by N words).
        Empty on the first attempt.
    """
    parts = [f"Theme: {theme}"]
    if guidance:
        parts.append("")
        parts.append(f"Guidance: {guidance}")
    if previous_attempt:
        parts.append("")
        parts.append("Your previous attempt (revise this directly to hit the length target):")
        parts.append(f'"""{previous_attempt}"""')
    if feedback:
        parts.append("")
        parts.append(feedback)
    parts.append("")
    if previous_attempt:
        parts.append("Rewrite the story to meet the requirements.")
    else:
        parts.append("Write the story now.")
    return "\n".join(parts)


def run_writer(node, theme: str, guidance: str = "", feedback: str = "",
               previous_attempt: str = "") -> str:
    """Run the writer LLMNode once and return the story text (stripped).

    Caller is responsible for resetting / recreating the node between attempts
    if a fresh context is wanted. Caller is also responsible for passing in
    feedback built from the previous attempt's length error, (optionally)
    guidance from an upstream orchestrator, and (optionally) the previous
    attempt's full text so writer can revise it directly.
    """
    node.reset_message()
    result = node.run(build_input(theme, guidance, feedback, previous_attempt))
    return (result.get("text") or "").strip()


def build_length_feedback(actual_len: int, target_len: int) -> str:
    """Construct the feedback string for the next attempt when the previous
    attempt missed the length target."""
    diff = target_len - actual_len
    if diff > 0:
        delta = f"too short by {diff}"
    else:
        delta = f"too long by {-diff}"
    return (
        f"Previous attempt was {actual_len} characters, {delta}. "
        f"Adjust to exactly {target_len}."
    )
