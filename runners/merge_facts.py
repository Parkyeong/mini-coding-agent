"""Standalone fact merger.

Re-runs `memory.merge_facts_into_global` over an existing experiment without
touching anything else. Use cases:

  - Try LLM dedup on an experiment that was already merged with string-only
    (or no) dedup, without re-running the agent over 257 cases.
  - Compare dedup methods side-by-side: write to a different output file via
    `--output`, then diff against the experiment's main facts file.
  - Iterate on the dedup PROMPT in `dedup.py` and re-merge cheaply.

The runner's `cmd_run` already calls this same merge function at the end of
every experiment; this script is for the after-the-fact / comparison use case.

Usage:
    # Re-merge baseline with LLM dedup, overwriting its global facts file
    python -m runners.merge_facts --exp baseline

    # Same, but write to a side file so you can diff against the original
    python -m runners.merge_facts --exp baseline --output /tmp/baseline_relabel.json

    # Skip the LLM (no dedup at all — every fact added as novel)
    python -m runners.merge_facts --exp baseline --no-dedup

    # Override the dedup model
    python -m runners.merge_facts --exp baseline --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP, DEDUP_MODEL
from llm_node import LLMNode
from memory import merge_facts_into_global

CASES_DIRNAME = "single_case_details"
DEFAULT_OUTPUT_FILENAME = "mbpp_global_facts.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-merge per-case facts into a global pool, with optional LLM dedup."
    )
    parser.add_argument("--exp", required=True,
                        help="experiment name under Execution/")
    parser.add_argument("--root", default=WORKSHOP)
    parser.add_argument("--output",
                        help=f"output path (default: Execution/<exp>/{DEFAULT_OUTPUT_FILENAME})")
    parser.add_argument("--no-dedup", action="store_true",
                        help="skip LLM dedup entirely; every fact added as novel "
                             "(useful for baseline comparisons)")
    parser.add_argument("--model", default=DEDUP_MODEL,
                        help=f"override dedup model (default: {DEDUP_MODEL})")
    args = parser.parse_args()

    exp_dir = os.path.join(args.root, args.exp)
    cases_dir = os.path.join(exp_dir, CASES_DIRNAME)
    if not os.path.isdir(cases_dir):
        print(f"[error] no cases dir at {cases_dir}", file=sys.stderr)
        sys.exit(1)

    output = args.output or os.path.join(exp_dir, DEFAULT_OUTPUT_FILENAME)

    case_dirs = sorted(glob.glob(os.path.join(cases_dir, "mbpp_*")))
    local_files = [os.path.join(d, "long_term_memory.json") for d in case_dirs]
    print(f"Found {len(local_files)} per-case fact files under {cases_dir}")

    dedup_node = None
    if not args.no_dedup:
        from dedup import PROMPT as DEDUP_PROMPT
        dedup_node = LLMNode(
            system_prompt=DEDUP_PROMPT,
            role="dedup",
            max_steps=1,
            model=args.model,
        )
        print(f"Using LLM dedup with model={args.model}. "
              f"This can take ~30-45 minutes for ~1000 candidate facts.")
    else:
        print("Skipping LLM dedup (--no-dedup); every fact added as novel.")

    info = merge_facts_into_global(local_files, output, dedup_node=dedup_node)
    s = info["stats"]
    print()
    print(f"=== Merge complete ===")
    print(f"  raw fact entries:   {s['raw']:>5}  (sum across all cases)")
    print(f"  unique after merge: {s['merged']:>5}")
    print(f"  reinforced (dedup): {s['dedup_hits']:>5}  (deduped via {s['dedup_method']})")
    print(f"  sources:            {s['sources']:>5}  case files read")
    print(f"  output:             {output}")


if __name__ == "__main__":
    main()
