"""role_pool — pool of cognitive roles used by the agent system.

Each role is a small module with a PROMPT plus helper functions; the runtime
"role object" is an LLMNode instance built by engine.build_llm_nodes that
wraps the PROMPT. Verifier is the lone exception — it has no LLM, just
pytest plumbing — but lives here too because the orchestrator treats it like
any other role in the pipeline.

Current roles:
    coder.py      — implement code, runs the LLM + tool loop
    planner.py    — break a task into 3-6 actionable steps
    verifier.py   — run pytest, return pass/fail (no LLM)
    summarizer.py — extract 1-2 project-level lessons after a passed task
    dedup.py      — judge fact equivalence at batch-merge time

Adding a new role: create a new module here following the planner.py
pattern (PROMPT + a small entry function), then wire it into engine.py.
"""
