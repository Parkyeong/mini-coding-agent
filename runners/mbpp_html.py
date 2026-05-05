"""Render an experiment's results into a self-contained dataset.html.

Reads everything under Execution/<exp>/ (the report, per-case working/long-term
memory, prompt/solution/test files, the global facts pool, and an optional
notes.md) and emits Execution/<exp>/dataset.html. Self-contained: inlined CSS,
opens directly in a browser. Independent of docs.html.

Usage:
    python -m runners.mbpp_html --exp first_test
    python -m runners.mbpp_html --exp first_test --output report.html

Library entry point used by mbpp_task.py at the end of `run`:
    render_experiment(exp_name)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import re
import sys
from collections import Counter
from statistics import median

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP

CASES_DIRNAME = "single_case_details"
FACTS_FILENAME = "mbpp_global_facts.json"
REPORT_FILENAME = "mbpp_exp_final_results.json"
NOTES_FILENAME = "notes.md"
OUTPUT_FILENAME = "dataset.html"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _safe_load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_read_text(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def _extract_case_stats(case_dir: str) -> dict:
    """Pull all per-case numbers out of working_memory.json + sibling files."""
    wm = _safe_load_json(os.path.join(case_dir, "working_memory.json"),
                         {"event_log": [], "candidate_facts": [], "plan": []})
    lt = _safe_load_json(os.path.join(case_dir, "long_term_memory.json"),
                         {"facts": []})
    prompt = _safe_read_text(os.path.join(case_dir, "prompt.md"))
    solution = _safe_read_text(os.path.join(case_dir, "solution.py"))
    test_solution = _safe_read_text(os.path.join(case_dir, "test_solution.py"))

    events = wm.get("event_log", [])
    role_counter = Counter()
    tool_counter = Counter()
    failed_tools = Counter()
    in_tok = out_tok = latency = llm_cnt = 0
    pytest_runs = pytest_pass = pytest_fail = 0
    coder_pytest = verifier_pytest = 0
    pytest_timeline: list[dict] = []  # one entry per pytest run, with source label
    plan_event_indices: list[int] = []
    first_pass_event_idx: int | None = None

    for i, ev in enumerate(events):
        kind = ev.get("kind", "?")
        payload = ev.get("payload", {})
        if kind == "llm_call":
            llm_cnt += 1
            role_counter[payload.get("role", "?")] += 1
            in_tok += payload.get("input_tokens", 0) or 0
            out_tok += payload.get("output_tokens", 0) or 0
            latency += payload.get("latency", 0) or 0
            if payload.get("role") == "planner":
                plan_event_indices.append(i)
        elif kind == "tool_call":
            tool_counter[payload.get("name", "?")] += 1
        elif kind == "tool_result":
            name = payload.get("name", "?")
            args = payload.get("args", {}) or {}
            result = payload.get("result", "")
            cmd = args.get("command", "") if isinstance(args, dict) else ""
            if name == "run_command" and "pytest" in (cmd or ""):
                pytest_runs += 1
                coder_pytest += 1
                rc = None
                if isinstance(result, str):
                    m = re.search(r"returncode:(-?\d+)", result)
                    if m:
                        rc = int(m.group(1))
                err_kind = ""
                if isinstance(result, str):
                    if "SyntaxError" in result:
                        err_kind = "SyntaxError"
                    elif "IndentationError" in result:
                        err_kind = "IndentationError"
                    elif "AssertionError" in result:
                        err_kind = "AssertionError"
                    elif rc not in (0, None):
                        err_kind = f"rc={rc}"
                if rc == 0:
                    pytest_pass += 1
                    if first_pass_event_idx is None:
                        first_pass_event_idx = i
                else:
                    pytest_fail += 1
                pytest_timeline.append({
                    "source": "coder",
                    "event_idx": i,
                    "rc": rc,
                    "err": err_kind,
                    "cmd": cmd,
                })
            if isinstance(result, dict) and result.get("ok") is False:
                failed_tools[name] += 1
        elif kind == "verify":
            # Engine-driven pytest call (one per coder iteration). Logged by
            # verifier.py — invisible to the LLM tool loop, so historically
            # absent from the timeline. Shown here labeled [verifier] so users
            # can tell it apart from coder's own run_command pytest calls.
            pytest_runs += 1
            verifier_pytest += 1
            rc = payload.get("returncode")
            passed_flag = bool(payload.get("passed"))
            timed_out = bool(payload.get("timed_out"))
            err_kind = ""
            if timed_out:
                err_kind = "timeout"
            elif rc is not None and rc != 0:
                # Try to recover error class from reason string.
                reason = payload.get("reason", "") or ""
                if "AssertionError" in reason:
                    err_kind = "AssertionError"
                elif "SyntaxError" in reason:
                    err_kind = "SyntaxError"
                elif "IndentationError" in reason:
                    err_kind = "IndentationError"
                else:
                    err_kind = f"rc={rc}"
            if passed_flag:
                pytest_pass += 1
                if first_pass_event_idx is None:
                    first_pass_event_idx = i
            else:
                pytest_fail += 1
            pytest_timeline.append({
                "source": "verifier",
                "event_idx": i,
                "rc": rc,
                "err": err_kind,
                "cmd": payload.get("command", ""),
            })

    return {
        "case_id": os.path.basename(case_dir),
        "events_total": len(events),
        "llm_calls": llm_cnt,
        "planner_calls": role_counter.get("planner", 0),
        "coder_calls": role_counter.get("coder", 0),
        "verifier_calls": role_counter.get("verifier", 0),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms": latency,
        "tools": dict(tool_counter),
        "tools_total": sum(tool_counter.values()),
        "failed_tools": dict(failed_tools),
        "pytest_runs": pytest_runs,
        "pytest_pass": pytest_pass,
        "pytest_fail": pytest_fail,
        "pytest_timeline": pytest_timeline,
        "coder_pytest_runs": coder_pytest,
        "verifier_pytest_runs": verifier_pytest,
        "plan_iterations": role_counter.get("planner", 0),
        "first_pass_event_idx": first_pass_event_idx,
        "candidate_facts": wm.get("candidate_facts", []),
        "long_term_facts": lt.get("facts", []),
        "plan_steps": wm.get("plan", []),
        "prompt": prompt,
        "solution": solution,
        "test_solution": test_solution,
    }


def collect_experiment(exp_name: str, root: str = WORKSHOP) -> dict:
    """Walk a single experiment directory and produce all data needed to render."""
    exp_dir = os.path.join(root, exp_name)
    cases_dir = os.path.join(exp_dir, CASES_DIRNAME)

    report = _safe_load_json(os.path.join(exp_dir, REPORT_FILENAME), {})
    facts_blob = _safe_load_json(os.path.join(exp_dir, FACTS_FILENAME), {"facts": []})
    notes_md = _safe_read_text(os.path.join(exp_dir, NOTES_FILENAME))

    case_dirs = sorted(
        d for d in (
            os.path.join(cases_dir, name) for name in os.listdir(cases_dir)
        ) if os.path.isdir(d)
    ) if os.path.isdir(cases_dir) else []
    case_stats = [_extract_case_stats(cd) for cd in case_dirs]

    # Cross-case totals
    totals = {
        "llm_calls": sum(c["llm_calls"] for c in case_stats),
        "planner_calls": sum(c["planner_calls"] for c in case_stats),
        "coder_calls": sum(c["coder_calls"] for c in case_stats),
        "verifier_calls": sum(c["verifier_calls"] for c in case_stats),
        "input_tokens": sum(c["input_tokens"] for c in case_stats),
        "output_tokens": sum(c["output_tokens"] for c in case_stats),
        "latency_ms": sum(c["latency_ms"] for c in case_stats),
        "tools": Counter(),
        "pytest_runs": sum(c["pytest_runs"] for c in case_stats),
        "pytest_pass": sum(c["pytest_pass"] for c in case_stats),
        "pytest_fail": sum(c["pytest_fail"] for c in case_stats),
        "candidate_facts_total": sum(len(c["candidate_facts"]) for c in case_stats),
        "long_term_facts_total": sum(len(c["long_term_facts"]) for c in case_stats),
    }
    for c in case_stats:
        for t, n in c["tools"].items():
            totals["tools"][t] += n

    return {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "report": report,
        "facts": facts_blob.get("facts", []),
        "notes_md": notes_md,
        "cases": case_stats,
        "totals": totals,
    }


# ---------------------------------------------------------------------------
# Anomaly detection (purely data-driven, no editorial)
# ---------------------------------------------------------------------------

def _detect_anomalies(cases: list[dict]) -> dict[str, set[str]]:
    """Return {case_id: set_of_anomaly_tags} for cards that should auto-open."""
    if not cases:
        return {}
    flagged: dict[str, set[str]] = {c["case_id"]: set() for c in cases}

    inputs = [c["input_tokens"] for c in cases]
    in_med = max(median(inputs), 1)
    for c in cases:
        if c["input_tokens"] > 2 * in_med and c["input_tokens"] >= 30_000:
            flagged[c["case_id"]].add("token-hot")
        if c["pytest_fail"] >= 3:
            flagged[c["case_id"]].add("many-pytest-fails")
        if c["plan_iterations"] >= 2:
            flagged[c["case_id"]].add("replanned")
        sm = c["tools"].get("save_memory", 0)
        if sm >= 8:
            flagged[c["case_id"]].add("save-memory-spam")
    return flagged


def _data_observations(data: dict) -> list[str]:
    """Plain-English bullets the script can fairly claim from the JSON."""
    obs: list[str] = []
    cases = data["cases"]
    if not cases:
        return obs

    # 1. Pass rate
    report = data["report"]
    totals = report.get("totals", {})
    if totals:
        obs.append(
            f"pass rate: {totals.get('passed', 0)} / {totals.get('total', 0)} "
            f"(failed={totals.get('failed', 0)}, crashed={totals.get('crashed', 0)})"
        )

    # 2. Token outliers
    by_in = sorted(cases, key=lambda c: -c["input_tokens"])[:3]
    if by_in:
        parts = ", ".join(
            f"{c['case_id']}={c['input_tokens']/1000:.1f}k" for c in by_in
        )
        obs.append(f"top-3 input tokens: {parts}")

    # 3. Pytest failures
    by_fail = [c for c in cases if c["pytest_fail"] > 0]
    if by_fail:
        by_fail.sort(key=lambda c: -c["pytest_fail"])
        parts = ", ".join(
            f"{c['case_id']}={c['pytest_fail']}/{c['pytest_runs']}" for c in by_fail[:5]
        )
        obs.append(
            f"cases with pytest failures: {len(by_fail)}/{len(cases)} "
            f"(top: {parts})"
        )
    else:
        obs.append("zero pytest failures across all cases")

    # 4. Plan iterations
    replanned = [c for c in cases if c["plan_iterations"] >= 2]
    if replanned:
        parts = ", ".join(
            f"{c['case_id']}({c['plan_iterations']}x)" for c in replanned
        )
        obs.append(f"verify→replan triggered in: {parts}")

    # 5. save_memory volume
    sm_calls = data["totals"]["tools"].get("save_memory", 0)
    facts_n = len(data["facts"])
    if sm_calls:
        obs.append(
            f"save_memory called {sm_calls}× across cases; "
            f"global pool ended with {facts_n} fact(s)"
        )

    # 6. global pool promotion gap — count cases whose long-term facts contribute
    # ZERO entries to the global pool (their fact strings appear nowhere). That's
    # a stronger signal than just the count mismatch (which can be legitimate
    # cross-case dedupe).
    global_fact_texts = {f.get("fact", "") for f in data["facts"]}
    missing_cases: list[tuple[str, int]] = []
    for c in cases:
        lt_facts = c.get("long_term_facts", [])
        if not lt_facts:
            continue
        in_global = sum(1 for f in lt_facts if f.get("fact", "") in global_fact_texts)
        if in_global == 0:
            missing_cases.append((c["case_id"], len(lt_facts)))
    if missing_cases:
        parts = ", ".join(f"{cid}({n})" for cid, n in missing_cases)
        total_missing = sum(n for _, n in missing_cases)
        obs.append(
            f"⚠ {len(missing_cases)} case(s) wrote long-term facts that never "
            f"reached the global pool ({total_missing} fact(s) total): {parts}"
        )

    # 7. Reinforce stats
    if data["facts"]:
        reinforced = [f for f in data["facts"] if f.get("reinforce_count", 0) > 0]
        if reinforced:
            obs.append(
                f"{len(reinforced)} / {len(data['facts'])} facts have been reinforced "
                f"≥ once across cases"
            )
        else:
            obs.append(
                f"all {len(data['facts'])} facts have reinforce_count = 0 "
                f"(no cross-case fact match yet)"
            )

    return obs


# ---------------------------------------------------------------------------
# Trivial markdown renderer for notes.md
# ---------------------------------------------------------------------------

def _render_markdown(md: str) -> str:
    """Tiny safe markdown — headings, bold, italics, inline code, lists, paragraphs."""
    if not md.strip():
        return ""
    lines = md.split("\n")
    out: list[str] = []
    in_ul = False
    in_code = False
    code_buf: list[str] = []

    def close_list():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    for line in lines:
        if line.startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
                code_buf = []
                in_code = False
            else:
                close_list()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        if not line.strip():
            close_list()
            continue
        m_h = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m_h:
            close_list()
            level = len(m_h.group(1))
            text = _inline_md(m_h.group(2))
            out.append(f"<h{min(level+2, 6)}>{text}</h{min(level+2, 6)}>")
            continue
        m_li = re.match(r"^[-*]\s+(.+)$", line)
        if m_li:
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline_md(m_li.group(1))}</li>")
            continue
        close_list()
        out.append(f"<p>{_inline_md(line)}</p>")
    close_list()
    if in_code and code_buf:
        out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
    return "\n".join(out)


def _inline_md(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    return s


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #0f1117; --panel: #1a1d27; --panel-2: #232734;
  --border: #2d3344; --border-light: #3a4054;
  --text: #e8eaed; --text-dim: #9aa0a6;
  --accent: #8ab4f8; --accent-2: #c58af9;
  --green: #81c995; --red: #f28b82; --orange: #fdd663; --pink: #f29ed4;
  --code-bg: #0b0d13;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.6; font-size: 14.5px;
}
.topbar {
  position: sticky; top: 0; z-index: 100;
  background: var(--panel); border-bottom: 1px solid var(--border);
  padding: 14px 32px; display: flex; align-items: center; gap: 24px;
}
.topbar .brand {
  font-size: 18px; font-weight: 700;
  background: linear-gradient(135deg, #8ab4f8 0%, #c58af9 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.topbar .meta { color: var(--text-dim); font-size: 13px; margin-left: auto; }
.content { max-width: 1200px; margin: 0 auto; padding: 36px 40px 80px; }
h1 { font-size: 30px; margin: 0 0 8px; color: var(--accent); }
h2 { font-size: 22px; margin: 40px 0 12px; color: var(--accent-2);
     border-left: 4px solid var(--accent-2); padding-left: 12px; }
h3 { font-size: 17px; margin: 24px 0 10px; color: var(--text); }
h4 { margin: 18px 0 8px; color: var(--accent); font-size: 15px; }
.lead { color: var(--text-dim); font-size: 14.5px; margin: 0 0 24px; }

code {
  font-family: "JetBrains Mono","Fira Code",Consolas,monospace;
  background: var(--code-bg); color: var(--accent);
  padding: 2px 6px; border-radius: 3px; font-size: 12.5px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px 16px; overflow-x: auto;
  font-size: 12.5px; line-height: 1.55; white-space: pre;
}
pre code { background: transparent; color: var(--text); padding: 0; }

.layer-stack { display: flex; flex-direction: column; gap: 12px; margin: 16px 0; }
.layer-row { display: grid; gap: 12px; }
.layer-row.cols-3 { grid-template-columns: repeat(3, 1fr); }
.layer-box {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}
.layer-box .layer-name { font-weight: 700; color: var(--accent); font-size: 16px; }
.layer-box .layer-sub { color: var(--text-dim); font-size: 12.5px; margin-top: 4px; }

.note-box {
  background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: 6px; padding: 12px 16px; margin: 14px 0; font-size: 13.5px;
}
.note-box.good { border-left-color: var(--green); }
.note-box.warn { border-left-color: var(--orange); }
.note-box.bad  { border-left-color: var(--red); }

table {
  width: 100%; border-collapse: collapse; margin: 12px 0;
  font-size: 13px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; overflow: hidden;
}
th, td {
  padding: 8px 12px; text-align: left;
  border-bottom: 1px solid var(--border); vertical-align: top;
}
th { background: var(--panel-2); color: var(--accent); font-weight: 600; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(138, 180, 248, 0.04); }

.pill {
  display: inline-block; background: var(--panel-2);
  border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px;
  font-family: "JetBrains Mono",Consolas,monospace; font-size: 11.5px;
  color: var(--accent); margin: 1px 3px 1px 0;
}
.pill.green { color: var(--green); }
.pill.orange { color: var(--orange); }
.pill.red { color: var(--red); }
.pill.pink { color: var(--pink); }

.step {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  margin: 12px 0; overflow: hidden; transition: border-color .15s ease;
}
.step:hover { border-color: var(--border-light); }
.step.flagged { border-left: 4px solid var(--orange); }
.step-header {
  padding: 14px 20px; cursor: pointer;
  display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
}
.step-header .title { font-size: 15.5px; font-weight: 600; color: var(--text); margin: 0; }
.step-header .summary { color: var(--text-dim); font-size: 12.5px; margin: 6px 0 0; }
.step-header .toggle {
  flex-shrink: 0; font-size: 11px; color: var(--text-dim);
  background: var(--panel-2); border: 1px solid var(--border);
  padding: 4px 10px; border-radius: 6px; white-space: nowrap; user-select: none;
}
.step.open .step-header .toggle::before { content: 'collapse'; }
.step:not(.open) .step-header .toggle::before { content: 'expand'; }
.step-body {
  display: none; padding: 0 20px 18px; border-top: 1px dashed var(--border);
}
.step.open .step-body { display: block; }

.tag {
  display: inline-block; padding: 1px 7px; border-radius: 3px;
  font-size: 11px; font-weight: 600; margin-right: 4px;
}
.tag.token-hot           { background: rgba(253,214,99,.15); color: var(--orange); }
.tag.many-pytest-fails   { background: rgba(242,139,130,.15); color: var(--red); }
.tag.replanned           { background: rgba(197,138,249,.15); color: var(--accent-2); }
.tag.save-memory-spam    { background: rgba(242,158,212,.15); color: var(--pink); }

.muted { color: var(--text-dim); }

.btn-bar {
  display: flex; gap: 8px; align-items: center;
  margin: 8px 0 16px;
}
.btn {
  background: transparent; border: 1px solid var(--border);
  color: var(--text-dim); padding: 6px 14px; border-radius: 6px;
  font-size: 12.5px; font-weight: 600; cursor: pointer;
  font-family: inherit; transition: all .15s ease;
}
.btn:hover { color: var(--accent); border-color: var(--accent); }
.btn:active { transform: translateY(1px); }

.kv-grid { display: grid; grid-template-columns: 140px 1fr; gap: 4px 14px;
           font-size: 13px; margin-top: 6px; }
.kv-grid .k { color: var(--accent); font-weight: 600; }
.fact-row td { font-family: "JetBrains Mono",Consolas,monospace; font-size: 12px; }
.tl-row { font-family: "JetBrains Mono",Consolas,monospace; font-size: 12px;
          color: var(--text); }
.tl-pass { color: var(--green); }
.tl-fail { color: var(--red); }
"""


def _format_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _esc(s: str) -> str:
    return html.escape(s if s is not None else "")


def _tag_label(tag: str) -> str:
    return {
        "token-hot": "🔥 token hot",
        "many-pytest-fails": "✗ pytest fails",
        "replanned": "↻ replanned",
        "save-memory-spam": "✎ save_memory spam",
    }.get(tag, tag)


def _render_overview_cards(data: dict) -> str:
    report = data["report"]
    totals = data["totals"]
    rep_t = report.get("totals", {})
    n_cases = len(data["cases"])
    pass_n = rep_t.get("passed", 0)
    total_n = rep_t.get("total", n_cases)
    pass_rate = (pass_n / total_n * 100) if total_n else 0.0
    started = report.get("started_at", "?")
    finished = report.get("finished_at", "?")
    model = report.get("model", "?")

    # Tools as a compact line
    tool_str = " · ".join(
        f"{t}:{n}" for t, n in sorted(totals["tools"].items(), key=lambda x: -x[1])
    ) or "(none)"

    return f"""
    <div class="layer-stack">
      <div class="layer-row cols-3">
        <div class="layer-box">
          <div class="layer-name">{pass_n} / {total_n} passed</div>
          <div class="layer-sub">pass@1 = {pass_rate:.1f}%. failed={rep_t.get('failed', 0)}, crashed={rep_t.get('crashed', 0)}</div>
        </div>
        <div class="layer-box">
          <div class="layer-name">{_format_int(totals['input_tokens'])} / {_format_int(totals['output_tokens'])} tokens</div>
          <div class="layer-sub">total input / output across all LLM calls</div>
        </div>
        <div class="layer-box">
          <div class="layer-name">{totals['latency_ms']/1000:.1f}s LLM latency</div>
          <div class="layer-sub">cumulative; avg {(totals['latency_ms']/max(totals['llm_calls'],1)):.0f} ms per LLM call</div>
        </div>
      </div>
      <div class="layer-row cols-3">
        <div class="layer-box">
          <div class="layer-name">{totals['llm_calls']} LLM calls</div>
          <div class="layer-sub">planner={totals['planner_calls']} · coder={totals['coder_calls']} · verifier={totals['verifier_calls']}</div>
        </div>
        <div class="layer-box">
          <div class="layer-name">{sum(totals['tools'].values())} tool calls</div>
          <div class="layer-sub">{_esc(tool_str)}</div>
        </div>
        <div class="layer-box">
          <div class="layer-name">{len(data['facts'])} global facts</div>
          <div class="layer-sub">candidate(per-case)={totals['candidate_facts_total']} · long-term(per-case)={totals['long_term_facts_total']}</div>
        </div>
      </div>
    </div>
    <p class="muted">model = <code>{_esc(model)}</code> · started {_esc(started)} · finished {_esc(finished)}</p>
    """


def _render_summary_table(data: dict, flagged: dict[str, set[str]]) -> str:
    rows = []
    rep_lookup = {r.get("instance"): r for r in data["report"].get("results", [])}
    for c in data["cases"]:
        cid = c["case_id"]
        rep = rep_lookup.get(cid, {})
        status = rep.get("status", "?")
        status_pill = {
            "passed":  '<span class="pill green">passed</span>',
            "failed":  '<span class="pill red">failed</span>',
            "crashed": '<span class="pill red">crashed</span>',
        }.get(status, f'<span class="pill orange">{_esc(status)}</span>')
        tags = " ".join(
            f'<span class="tag {t}">{_tag_label(t)}</span>'
            for t in sorted(flagged.get(cid, set()))
        )
        rows.append(f"""
          <tr>
            <td><a href="#case-{_esc(cid)}"><b>{_esc(cid)}</b></a> {tags}</td>
            <td>{status_pill}</td>
            <td>{c['plan_iterations']}</td>
            <td>{c['llm_calls']}</td>
            <td>{c['input_tokens']/1000:.1f}k</td>
            <td>{c['output_tokens']/1000:.1f}k</td>
            <td>{c['latency_ms']/1000:.1f}s</td>
            <td>{c['pytest_pass']} / {c['pytest_runs']}</td>
            <td>{c['tools'].get('save_memory', 0)}</td>
            <td>{c['tools_total']}</td>
          </tr>""")
    return f"""
    <table>
      <thead>
        <tr>
          <th>case</th><th>status</th><th>plan</th><th>LLM</th>
          <th>input</th><th>output</th><th>latency</th>
          <th>pytest pass/total</th><th>save_memory</th><th>tools</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def _render_pytest_timeline(timeline: list[dict]) -> str:
    if not timeline:
        return '<p class="muted">no pytest runs recorded.</p>'
    rows = []
    for i, t in enumerate(timeline):
        rc = t["rc"]
        cls = "tl-pass" if rc == 0 else "tl-fail"
        sym = "✓" if rc == 0 else "✗"
        err = f" — {_esc(t['err'])}" if t["err"] else ""
        rc_str = f"rc={rc}" if rc is not None else "rc=?"
        source = t.get("source", "coder")
        # Visual labels so coder's manual pytest calls and engine-driven
        # verifier runs are easy to tell apart.
        label = "[verifier]" if source == "verifier" else "[coder]   "
        rows.append(
            f'<div class="tl-row {cls}">#{i:<2} {label} {sym} '
            f'event={t["event_idx"]:<4} {rc_str}{err}</div>'
        )
    return "<pre>" + "\n".join(rows) + "</pre>"


def _render_case_card(c: dict, flagged_tags: set[str], status: str) -> str:
    cid = c["case_id"]
    open_cls = " open" if flagged_tags else ""
    flag_cls = " flagged" if flagged_tags else ""
    tag_html = " ".join(
        f'<span class="tag {t}">{_tag_label(t)}</span>'
        for t in sorted(flagged_tags)
    )

    # Tool stats table
    tool_rows = []
    for tool, n in sorted(c["tools"].items(), key=lambda x: -x[1]):
        tool_rows.append(
            f"<tr><td><code>{_esc(tool)}</code></td><td>{n}</td></tr>"
        )
    tool_table = (
        "<table><thead><tr><th>tool</th><th>count</th></tr></thead><tbody>"
        + "".join(tool_rows) + "</tbody></table>"
    ) if tool_rows else '<p class="muted">no tool calls.</p>'

    # Candidate facts (within this case)
    fact_rows = []
    for f in c["candidate_facts"]:
        fact_rows.append(
            f'<tr class="fact-row"><td><span class="pill">{_esc(f.get("category", "?"))}</span></td>'
            f'<td>{_esc(f.get("fact", ""))}</td></tr>'
        )
    fact_block = (
        f"<table><thead><tr><th>category</th><th>fact</th></tr></thead>"
        f"<tbody>{''.join(fact_rows)}</tbody></table>"
        if fact_rows else '<p class="muted">no candidate facts.</p>'
    )

    status_pill = {
        "passed":  '<span class="pill green">passed</span>',
        "failed":  '<span class="pill red">failed</span>',
        "crashed": '<span class="pill red">crashed</span>',
    }.get(status, f'<span class="pill orange">{_esc(status)}</span>')

    summary = (
        f"{status_pill} · {c['plan_iterations']} plan · {c['llm_calls']} LLM · "
        f"{c['input_tokens']/1000:.1f}k / {c['output_tokens']/1000:.1f}k tok · "
        f"{c['latency_ms']/1000:.1f}s · pytest {c['pytest_pass']}/{c['pytest_runs']} · "
        f"save_memory={c['tools'].get('save_memory', 0)}"
    )

    return f"""
    <div class="step{open_cls}{flag_cls}" id="case-{_esc(cid)}">
      <div class="step-header">
        <div>
          <p class="title">{_esc(cid)} {tag_html}</p>
          <p class="summary">{summary}</p>
        </div>
        <div class="toggle"></div>
      </div>
      <div class="step-body">
        <h4>Prompt</h4>
        <pre><code>{_esc(c['prompt'].strip())}</code></pre>

        <h4>solution.py (final)</h4>
        <pre><code>{_esc(c['solution'].strip())}</code></pre>

        <h4>test_solution.py (final)</h4>
        <pre><code>{_esc(c['test_solution'].strip())}</code></pre>

        <h4>tool calls</h4>
        {tool_table}

        <h4>pytest timeline ({c['pytest_runs']} runs · {c['pytest_pass']} pass / {c['pytest_fail']} fail · coder={c.get('coder_pytest_runs', 0)}, verifier={c.get('verifier_pytest_runs', 0)})</h4>
        {_render_pytest_timeline(c['pytest_timeline'])}

        <h4>candidate facts written by this case ({len(c['candidate_facts'])})</h4>
        {fact_block}
      </div>
    </div>
    """


def _render_facts_table(facts: list[dict]) -> str:
    if not facts:
        return '<p class="muted">no facts in global pool.</p>'
    sorted_facts = sorted(
        facts,
        key=lambda f: (
            -(f.get("reinforce_count", 0) or 0),
            -(f.get("confidence", 0) or 0),
            f.get("created_at", ""),
        ),
    )
    rows = []
    for i, f in enumerate(sorted_facts, start=1):
        reinforced_by = f.get("reinforced_by", []) or []
        reinforced_by_str = ", ".join(reinforced_by) if reinforced_by else "—"
        rows.append(f"""
          <tr>
            <td>{i}</td>
            <td><span class="pill">{_esc(f.get('category', '?'))}</span></td>
            <td>{_esc(f.get('fact', ''))}</td>
            <td>{f.get('reinforce_count', 0)}</td>
            <td>{f.get('confidence', 0):.2f}</td>
            <td>{len(reinforced_by)}</td>
            <td><code>{_esc(reinforced_by_str)}</code></td>
            <td><code>{_esc(f.get('source_task_id', '?'))}</code></td>
          </tr>
        """)
    cat_counter = Counter(f.get("category", "?") for f in facts)
    cat_lines = " · ".join(f"{c}={n}" for c, n in cat_counter.most_common())

    return f"""
    <p class="muted">{len(facts)} fact(s); by category: {_esc(cat_lines)}. Sorted by reinforce_count desc, then confidence desc.</p>
    <table>
      <thead>
        <tr>
          <th>#</th><th>category</th><th>fact</th>
          <th>reinforce_count</th><th>confidence</th>
          <th>#tasks</th><th>reinforced_by</th><th>source</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def _render_observations(obs: list[str]) -> str:
    if not obs:
        return ""
    items = "".join(f"<li>{_esc(o)}</li>" for o in obs)
    return f'<div class="note-box"><b>Auto-derived observations:</b><ul>{items}</ul></div>'


def render_html(data: dict) -> str:
    flagged = _detect_anomalies(data["cases"])
    rep_lookup = {r.get("instance"): r for r in data["report"].get("results", [])}
    obs = _data_observations(data)

    case_cards_html = "\n".join(
        _render_case_card(
            c,
            flagged.get(c["case_id"], set()),
            rep_lookup.get(c["case_id"], {}).get("status", "?"),
        )
        for c in data["cases"]
    )

    notes_html = _render_markdown(data["notes_md"])
    notes_section = (
        f'<h2>Notes (from notes.md)</h2><div class="note-box">{notes_html}</div>'
        if notes_html else ""
    )

    generated_at = _dt.datetime.now().isoformat(timespec="seconds")
    title = f"{data['exp_name']} — experiment results"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="topbar">
  <div class="brand">mini coding agent</div>
  <div class="meta">experiment: <b>{_esc(data['exp_name'])}</b> · generated {_esc(generated_at)}</div>
</div>

<div class="content">
  <h1>Experiment: {_esc(data['exp_name'])}</h1>
  <p class="lead">
    Auto-generated from <code>Execution/{_esc(data['exp_name'])}/</code>.
    Source files: mbpp_exp_final_results.json, mbpp_global_facts.json, and per-case working_memory.json / long_term_memory.json.
    Re-run <code>python -m runners.mbpp_html --exp {_esc(data['exp_name'])}</code> to refresh.
  </p>

  <h2>Overview</h2>
  {_render_overview_cards(data)}

  {_render_observations(obs)}

  {notes_section}

  <h2>Per-case summary ({len(data['cases'])} cases)</h2>
  {_render_summary_table(data, flagged)}

  <h2>Per-case detail</h2>
  <p class="muted">Cards flagged as anomalies are auto-expanded; click any header to toggle.</p>
  <div class="btn-bar">
    <button class="btn" id="btn-expand-all">expand all</button>
    <button class="btn" id="btn-collapse-all">collapse all</button>
    <button class="btn" id="btn-reset-default">reset to default</button>
  </div>
  {case_cards_html}

  <h2>Global facts pool ({len(data['facts'])})</h2>
  {_render_facts_table(data['facts'])}

</div>

<script>
  // Capture each card's initial open/collapsed state so "reset to default"
  // can restore the anomaly-driven layout without a page reload.
  const steps = document.querySelectorAll('.step');
  const defaultOpen = new Set();
  steps.forEach(s => {{
    if (s.classList.contains('open')) defaultOpen.add(s.id);
  }});

  document.querySelectorAll('.step-header').forEach(h => {{
    h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
  }});

  document.getElementById('btn-expand-all').addEventListener('click', () => {{
    steps.forEach(s => s.classList.add('open'));
  }});
  document.getElementById('btn-collapse-all').addEventListener('click', () => {{
    steps.forEach(s => s.classList.remove('open'));
  }});
  document.getElementById('btn-reset-default').addEventListener('click', () => {{
    steps.forEach(s => {{
      s.classList.toggle('open', defaultOpen.has(s.id));
    }});
  }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def render_experiment(exp_name: str, root: str = WORKSHOP,
                      output_filename: str = OUTPUT_FILENAME) -> str:
    """Build dataset.html for an experiment. Returns the absolute output path."""
    data = collect_experiment(exp_name, root=root)
    html_text = render_html(data)
    out_path = os.path.join(data["exp_dir"], output_filename)
    os.makedirs(data["exp_dir"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an experiment's results into a self-contained HTML."
    )
    parser.add_argument("--exp", required=True, help="experiment name under Execution/")
    parser.add_argument("--root", default=WORKSHOP,
                        help=f"root directory holding experiments (default: {WORKSHOP})")
    parser.add_argument("--output", default=OUTPUT_FILENAME,
                        help=f"output filename inside the exp dir (default: {OUTPUT_FILENAME})")
    args = parser.parse_args()
    out = render_experiment(args.exp, root=args.root, output_filename=args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
