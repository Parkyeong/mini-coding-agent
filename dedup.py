"""Semantic fact-dedup judge.

The string-normalize pass ('lowercase + collapse whitespace') only catches
exact-after-normalization duplicates. The real duplicates in the global facts
pool are *paraphrases* — same meaning, different wording — which slip through.

This module provides a tiny LLM-driven binary judge: given a NEW fact and a
list of EXISTING facts, decide whether the new one is equivalent to any of
them. We run it at fact-merge time (after a parallel run finishes) so the
expense is amortized over many parallel agent calls and only adds ~30-45 min
to a 2.5h experiment.

The judge is just an LLMNode configured with this PROMPT and (typically) a
cheaper model (config.DEDUP_MODEL = gpt-4.1-mini). Used by
memory.merge_facts_into_global when ENABLE_LLM_DEDUP is on.
"""

import re

PROMPT = """You decide whether two facts about a software project mean the same thing.

You will see ONE new fact and a numbered list of existing facts. Decide whether
the new fact is equivalent to any existing one — same actionable knowledge,
even if the wording is different.

EQUIVALENT (same actionable knowledge, just different wording):
  "this project uses pytest with -q flag"
  "tests are run quietly via pytest -q"

  "function tracks seen characters with a set"
  "uses a set to record which characters have appeared"

NOT EQUIVALENT (different scope / different topic / different specificity):
  "the project uses pytest"
  "pytest is run with --disable-warnings"
  (second is more specific than first)

  "use math.pi for radian conversion"
  "use list comprehension for filtering"
  (different topics)

Respond with EXACTLY ONE integer and nothing else:
  - 0  if the new fact is novel (not equivalent to any existing fact)
  - N  the 1-indexed number of the equivalent existing fact (1, 2, 3, ...)
"""


def build_dedup_input(new_fact: str, existing: list[str]) -> str:
    lines = [f'NEW fact:\n  "{new_fact}"', "", "EXISTING facts:"]
    for i, f in enumerate(existing, 1):
        lines.append(f'  {i}. "{f}"')
    return "\n".join(lines)


def find_equivalent(node, new_fact: str, existing: list[str]) -> int:
    """Ask the dedup LLMNode whether `new_fact` is equivalent to any in `existing`.

    Returns:
      0  → novel, add as new
      N  → equivalent to existing[N-1], reinforce instead of adding

    On parse failure (model returns garbage / empty), falls back to 0 (novel)
    — slight over-add is much better than wrongly merging non-duplicates.
    """
    if not existing:
        return 0
    if not (new_fact or "").strip():
        return 0

    node.reset_message()
    result = node.run(build_dedup_input(new_fact, existing))
    text = (result.get("text", "") or "").strip()

    m = re.search(r"-?\d+", text)
    if not m:
        return 0
    n = int(m.group())
    if 0 <= n <= len(existing):
        return n
    # Out-of-range index → treat as novel (defensive).
    return 0
