"""Textplanner role: read previous draft + verifier result, output concrete
edit instructions for the writer.

Used by method_fixed in a fixed textplanner → writer → verifier loop. Unlike
brain (which outputs a plan or workflow once per round), textplanner runs
EVERY iteration:

  - Mode 1 (no previous draft): give writer a brief outline / direction
    (opening / conflict / ending — 1-3 short bullets, exact target length).
  - Mode 2 (previous draft + length feedback): output the SMALLEST POSSIBLE
    edit that closes the gap — point to a specific phrase to delete (too
    long) or a specific clause to append (too short). Do NOT rewrite.

Visibility:
  textplanner is intentionally NOT exposed to method_brain's brain — no
  SUPPORTED_TASKS attribute, and not in role_pool/brain.py's _ROLE_MODULES.
  This keeps the experimental contrast clean: method_brain's brain designs
  its own workflow with writer + length_checker; method_fixed plumbs a
  fixed textplanner-driven loop. To expose later: add SUPPORTED_TASKS=
  ["story"] here, add to _ROLE_MODULES, and add a textplanner dispatch
  branch in method_brain.py's DSL executor.
"""

from __future__ import annotations


PROMPT = """You are a text-editing planner for English flash-fiction at a
strict character-length target. You do NOT write the story yourself — you
tell the writer exactly what to do.

Two modes:

1. INITIAL OUTLINE (no previous draft yet):
   Give writer a brief direction — 1-3 short bullets covering:
     - opening / setting
     - middle / conflict or twist
     - ending / tone
   State the exact target length so writer aims for it.

2. EDIT INSTRUCTIONS (previous draft + length feedback given):
   The draft missed the length target. Output the SMALLEST POSSIBLE edit
   that closes the gap. Be SPECIFIC:
     - If too long: point to a phrase or clause to delete. Estimate the
       character savings (e.g. "delete 'in the pouring rain' to save ~18
       chars").
     - If too short: specify what to APPEND (a short clause or coda at the
       end). Estimate the gain (e.g. "add a 5-word coda after 'goodbye',
       roughly 25 chars").
   Do NOT rewrite the whole story. Do NOT change the plot. One or two
   targeted edits, not ten.

Output format: plain text — a few short bullets. No JSON, no preamble like
"Here are the edits:", no markdown headers. Just the bullets.
"""


def build_input(theme: str, target_len: int = 241,
                previous_attempt: str = "",
                verifier_result: dict | None = None) -> str:
    """User-message text for textplanner.

    First iteration (previous_attempt empty) → mode 1 (outline).
    Later iterations (previous_attempt + verifier_result) → mode 2 (edits).
    """
    if not previous_attempt:
        return (
            f"Theme: {theme}\n"
            f"Target length: exactly {target_len} characters.\n\n"
            f"This is iteration 1 — no draft yet. Give the writer an "
            f"INITIAL OUTLINE (mode 1)."
        )

    cur_len = len(previous_attempt)
    delta_text = (verifier_result or {}).get("delta_text") or ""
    diff = cur_len - target_len   # >0 too long, <0 too short
    return (
        f"Theme: {theme}\n"
        f"Target length: exactly {target_len} characters.\n\n"
        f"Writer's previous attempt ({cur_len} chars, {delta_text}):\n"
        f"---BEGIN---\n{previous_attempt}\n---END---\n\n"
        f"Give EDIT INSTRUCTIONS (mode 2) — the smallest possible edit "
        f"that closes the {abs(diff)}-character gap."
    )
