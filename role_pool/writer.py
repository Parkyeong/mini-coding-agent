"""Writer role: flash-fiction creative writing under a strict length constraint.

The actual LLM call goes through an LLMNode instance configured for this
role. Used by the story_task runners (baseline / Track A / Track B). The
worker model is locked to gpt-4o-mini in ROLE_CONFIGS — orchestrating roles
(brain in Track A) may use stronger models, but the prose itself is always
4o-mini for fair cross-method comparison.
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


def build_input(theme: str, feedback: str = "") -> str:
    """Build the user-message text fed into the writer LLMNode.

    `feedback` is the length-error message from the previous attempt, e.g.
    "Previous attempt was 256 characters, too long by 15. Adjust to exactly
    241." Empty on the first attempt.
    """
    parts = [f"Theme: {theme}"]
    if feedback:
        parts.append("")
        parts.append(feedback)
    parts.append("")
    parts.append("Write the story now.")
    return "\n".join(parts)


def run_writer(node, theme: str, feedback: str = "") -> str:
    """Run the writer LLMNode once and return the story text (stripped).

    Caller is responsible for resetting / recreating the node between attempts
    if a fresh context is wanted. Caller is also responsible for passing in
    feedback built from the previous attempt's length error.
    """
    node.reset_message()
    result = node.run(build_input(theme, feedback))
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
