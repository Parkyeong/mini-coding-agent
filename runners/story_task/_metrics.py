"""Shared accuracy helpers across methods + run_all + html.

Three accuracy axes:

  RUN-LEVEL (pass@N): does the run end with a Pass? Denominator = number of
  runs. The conventional "did the agent succeed" metric.

  CYCLE-LEVEL: each main-loop iteration is its own attempt. Denominator =
  total cycles actually executed across all runs (variable — early-exit on
  Pass means later cycles aren't attempted). Numerator = cycles whose own
  outcome was a Pass.

  VALIDATED-LEVEL: cycles that Pass via the strategy (not via writer's
  cold-start luck on the first attempt). Same denominator as cycle-level;
  numerator = (Pass AND strategy_validated). For baseline this equals
  cycle-level (no internal strategy distinction). For method_fixed /
  method_brain, validated = NOT (Pass AND first_attempt_was_hit). This
  isolates strategy contribution from 4o-mini's per-attempt base luck rate.

Different methods structure their main loop differently, so cycle extraction
is method-specific:

  - baseline       : each (writer, length_checker) pair in trajectory = one
                     cycle. Single-attempt cycles → all are trivially
                     "validated" (no strategy to distinguish).
  - method_fixed   : each cycle in `r["cycles"]` has `attempts` (≤3
                     writer-verify) + an end-of-cycle textplanner. cycle
                     Pass = any attempt hit. validated = Pass came on
                     attempts[1+] (not attempts[0]). Plus trailing attempt.
  - method_brain   : each cycle in `r["cycles"]` has a `hit` field and an
                     explicit `strategy_validated` field set by run_one().
"""

from __future__ import annotations


def per_run_cycle_outcomes(r: dict, method_name: str) -> list[dict]:
    """Extract per-cycle outcomes (one entry per main-loop iteration)
    for a single run record.

    Returns list of {"hit": bool, "length": int, "validated": bool} in
    execution order. `validated` distinguishes strategy-driven Pass from
    cold-start luck (see module docstring).
    """
    out: list[dict] = []

    if method_name == "baseline":
        # Each writer-verify retry IS one cycle. No multi-step strategy
        # within a cycle, so "validated" is trivially True (no luck-vs-
        # strategy distinction at the cycle level).
        for step in r.get("trajectory", []):
            if step.get("role") != "length_checker":
                continue
            o = step.get("output") or {}
            if not isinstance(o, dict) or "length" not in o:
                continue
            out.append({
                "hit": bool(o.get("hit")),
                "length": o.get("length", 0),
                "validated": True,
            })
        return out

    if method_name == "method_fixed":
        for c in r.get("cycles", []):
            attempts = c.get("attempts", []) or []
            if not attempts:
                continue
            hit_in_cycle = any(a.get("hit") for a in attempts)
            first_was_hit = bool(attempts[0].get("hit"))
            if hit_in_cycle:
                a = next(a for a in attempts if a.get("hit"))
                out.append({
                    "hit": True,
                    "length": a.get("length", 0),
                    # Pass came from strategy iff it wasn't the very first
                    # attempt (textplanner advice + minimal-edit kicked in).
                    "validated": not first_was_hit,
                })
            else:
                last = attempts[-1]
                out.append({
                    "hit": False,
                    "length": last.get("length", 0),
                    # The full cycle's writer-verifies ran (3 of them, no
                    # early exit). Strategy was exercised even if it didn't
                    # win, so validated=True for the purpose of "did the
                    # multi-step process get a chance to run".
                    "validated": True,
                })
        trailing = r.get("trailing_attempt")
        if trailing is not None:
            out.append({
                "hit": bool(trailing.get("hit")),
                "length": trailing.get("length", 0),
                # Trailing writer-verify is itself the consumption step of
                # cycle-8 textplanner advice. Always counts as validated.
                "validated": True,
            })
        return out

    if (method_name == "method_brain"
            or method_name == "method_brain_code"
            or method_name.startswith("brain_ablation")):
        # method_brain, method_brain_code, and brain_ablation_X share the
        # same cycle record schema (each writes the strategy_validated
        # field per cycle). Treat them identically here.
        for c in r.get("cycles", []):
            out.append({
                "hit": bool(c.get("hit")),
                "length": c.get("final_length", 0),
                "validated": bool(c.get("strategy_validated", True)),
            })
        return out

    return out


def per_theme_counts(summary: dict, method_name: str) -> dict:
    """Aggregate run / cycle / validated counts per theme.

    Returns: {
      theme_id: {
        "runs_hits": int, "runs_total": int,
        "cycle_hits": int, "cycle_total": int,
        "validated_hits": int,   # cycles that BOTH hit AND were strategy-driven
        # validated_total uses cycle_total as denominator (shows "what fraction
        # of all cycle attempts hit via strategy, not cold-start luck").
      }
    }
    """
    by_theme: dict = {}
    for r in summary.get("results", []) or []:
        tid = r.get("theme_id")
        if not tid:
            continue
        bucket = by_theme.setdefault(
            tid,
            {"runs_hits": 0, "runs_total": 0,
             "cycle_hits": 0, "cycle_total": 0,
             "validated_hits": 0},
        )
        bucket["runs_total"] += 1
        if r.get("hit"):
            bucket["runs_hits"] += 1
        for c in per_run_cycle_outcomes(r, method_name):
            bucket["cycle_total"] += 1
            if c["hit"]:
                bucket["cycle_hits"] += 1
                if c.get("validated", True):
                    bucket["validated_hits"] += 1
    return by_theme


def overall_counts(summary: dict, method_name: str) -> dict:
    """Aggregate run / cycle / validated counts across all themes."""
    runs_hits = runs_total = cycle_hits = cycle_total = validated_hits = 0
    for r in summary.get("results", []) or []:
        runs_total += 1
        if r.get("hit"):
            runs_hits += 1
        for c in per_run_cycle_outcomes(r, method_name):
            cycle_total += 1
            if c["hit"]:
                cycle_hits += 1
                if c.get("validated", True):
                    validated_hits += 1
    return {
        "runs_hits": runs_hits,
        "runs_total": runs_total,
        "cycle_hits": cycle_hits,
        "cycle_total": cycle_total,
        "validated_hits": validated_hits,
    }
