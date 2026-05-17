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
    Args:
      theme (str, required)
      guidance (str, optional)         — your strategic instruction
      previous_attempt (str, optional) — full text of the prior draft

    TWO MODES (you choose by what you pass as previous_attempt):

    ─ FRESH DRAFT mode  (previous_attempt = "")
      Writer drafts from theme + your guidance. No host auto-direction.
      Use when you want a brand-new attempt unanchored to prior drafts.

    ─ MINIMAL EDIT mode  (previous_attempt = "<prior draft>")
      The HOST automatically computes diff = 241 − len(previous_attempt)
      and INJECTS a precise direction into writer's user message:

        "This is N characters — exactly M too long/short for the 241
         target. Output the SAME story with exactly M characters
         DELETED/APPENDED from the end (or tightened in place).
         Do NOT rewrite. Do NOT change the plot."

      Your `guidance` LAYERS ON TOP of this auto-direction. Examples:

      {
        "aligned_with_host": [
          "tighten the final clause",
          "remove an adverb or punctuation",
          "prefer cutting filler words near the end",
          "preserve plot and tone"
        ],
        "conflicting_with_host": [
          {"guidance": "rewrite the opening",
           "why": "host says do NOT rewrite"},
          {"guidance": "edit the middle sentence only",
           "why": "host says edit at the end"},
          {"guidance": "change the protagonist's name",
           "why": "host says only trim/append"},
          {"guidance": "compress phrases throughout",
           "why": "host narrows scope to the tail"}
        ]
      }

      If you actually need a different action (fresh start, structural
      change), use FRESH DRAFT mode instead (pass "" for
      previous_attempt). Don't try to override host inside MINIMAL EDIT.

    Notes:
      - $var only substitutes whole-string. "$last.diff chars" embedded
        inside a longer string stays LITERAL — useless. The host already
        injects the exact number in its auto-direction, so you don't
        need to.
      - Writer is gpt-4o-mini: minimal-edit accuracy is high when
        |diff| ≤ 15. For |diff| > 30, prefer FRESH DRAFT mode (cheaper
        than dragging a wrong draft toward the target step by step).
    Returns: story text
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
                previous_attempt: str = "", target_len: int = 241) -> str:
    """Build the user-message text fed into the writer LLMNode.

    `guidance`: optional hint set by an orchestrator (e.g. brain in
        method_fixed / method_brain). Examples: "aim for 230 chars, be
        concise". Empty in baseline (no orchestrator).
    `feedback`: length-error message from the previous attempt, e.g.
        "Previous attempt was 256 characters, too long by 15. Adjust to
        exactly 241." Empty on the first attempt.
    `previous_attempt`: the FULL TEXT of the previous attempt's story.
        When non-empty, writer is asked to MINIMALLY EDIT this draft —
        keep the existing prose intact and only trim / extend by the exact
        surplus or deficit reported in `feedback`. Do NOT rewrite from
        scratch. This converges faster than full regeneration because the
        model only has to touch the overflowing/missing tail.
        Empty on the first attempt.
    """
    parts = [f"Theme: {theme}"]
    if guidance:
        parts.append("")
        parts.append(f"Guidance: {guidance}")

    if previous_attempt:
        # Host-side computed direction. Symmetric "trim OR append" instructions
        # confused gpt-4o-mini (it tended to take the append branch even when
        # over-length, drifting longer every retry). So we pick the branch
        # ourselves and only show the writer the one it needs.
        cur_len = len(previous_attempt)
        diff = target_len - cur_len   # positive ⇒ too short, negative ⇒ too long

        parts.append("")
        parts.append("Your previous attempt (between the ---BEGIN--- / ---END--- markers below):")
        parts.append("---BEGIN---")
        parts.append(previous_attempt)
        parts.append("---END---")
        parts.append("")

        if diff < 0:
            n = -diff
            parts.append(
                f"This is {cur_len} characters — exactly {n} too long for the "
                f"{target_len}-character target."
            )
            parts.append(
                f"Output the SAME story with exactly {n} characters DELETED "
                f"from the end (or tightened in place). Do NOT add any new "
                f"text. Do NOT rewrite. Do NOT change the plot or the earlier "
                f"sentences. Only remove characters from the tail until the "
                f"total is exactly {target_len}."
            )
        elif diff > 0:
            n = diff
            parts.append(
                f"This is {cur_len} characters — exactly {n} too short for the "
                f"{target_len}-character target."
            )
            parts.append(
                f"Output the SAME story with exactly {n} more characters "
                f"APPENDED to the end. Do NOT rewrite. Do NOT change the plot "
                f"or any earlier sentence. Only extend the ending until the "
                f"total is exactly {target_len}."
            )
        else:
            parts.append(
                f"This is already exactly {target_len} characters. Output it "
                f"unchanged."
            )

        parts.append("")
        parts.append("Output ONLY the resulting full story (no quotes, no "
                     "explanation, no preamble).")
    else:
        if feedback:
            parts.append("")
            parts.append(feedback)
        parts.append("")
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
