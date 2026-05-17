"""html.py — render 3-method comparison HTML for the story task.

Reads the 3 summary.json files (one per method) and produces a self-contained
HTML report at WORKSHOP/story_241/comparison.html.

Structure:
  1. Overview table: hit rate / tokens / per-role breakdown for each method
  2. Per-theme grid: 4 themes × 3 methods, hit/miss per run
  3. Per-run drill-down: 16 (theme × run) blocks, each with 3 methods stacked
     as <details>, each method showing its complete trajectory (every step's
     full input/output).

Missing methods (no summary.json yet) are rendered as "(not run yet)" — so
running just one or two methods still produces a usable HTML.

Usage (standalone):
  python -m runners.story_task.html

This script is also auto-called at the end of each method's main(), so the
HTML stays in sync after every method run.
"""

from __future__ import annotations

import html as _html
import json
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import WORKSHOP


# Default methods for the standard 3-method comparison. Anything found in
# BASE_DIR (e.g. brain_ablation_a/b/c/d) is appended at render time.
DEFAULT_METHODS = ["baseline", "method_fixed", "method_brain"]
METHODS = list(DEFAULT_METHODS)   # mutated by main() after BASE_DIR is set

# If STORY_EXP_NAME is set, base path becomes story_241/<exp_name>/, matching
# the per-method output dirs. Otherwise defaults to story_241/.
_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
BASE_DIR = (
    os.path.join(WORKSHOP, "story_241", _EXP_NAME) if _EXP_NAME
    else os.path.join(WORKSHOP, "story_241")
)
OUTPUT_FILENAME = "comparison.html"


def _discover_methods(base_dir: str) -> list[str]:
    """Discover method subdirectories with summary.json under base_dir.

    Ordering: any of the 3 default methods that are present come first (in
    fixed order), then any other discovered methods (e.g. brain_ablation_a,
    brain_ablation_b, ...) sorted alphabetically.
    """
    if not os.path.isdir(base_dir):
        return list(DEFAULT_METHODS)
    found = []
    for name in sorted(os.listdir(base_dir)):
        if os.path.isfile(os.path.join(base_dir, name, "summary.json")):
            found.append(name)
    if not found:
        return list(DEFAULT_METHODS)
    ordered = [m for m in DEFAULT_METHODS if m in found]
    extras = sorted(m for m in found if m not in DEFAULT_METHODS)
    return ordered + extras


def _is_brain_method(method_name: str) -> bool:
    """Methods rendered with the brain-style view (per-cycle workflow tree,
    luck-Pass badge, validated accuracy column). Covers method_brain and any
    brain_ablation_X variant produced by brain_ablation.py."""
    return method_name == "method_brain" or method_name.startswith("brain_ablation")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_summaries() -> dict:
    """Load each method's summary.json. Missing files → None entry."""
    summaries: dict = {}
    for m in METHODS:
        path = os.path.join(BASE_DIR, m, "summary.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    summaries[m] = json.load(f)
            except Exception as e:
                summaries[m] = {"_error": f"failed to load: {e}"}
        else:
            summaries[m] = None
    return summaries


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def esc(s) -> str:
    """HTML-escape any value (cast to str first)."""
    return _html.escape(str(s) if s is not None else "")


def _pre(text) -> str:
    """Render text inside a <pre> block, escaping HTML, preserving whitespace."""
    return f"<pre>{esc(text)}</pre>"


def _json_pre(obj) -> str:
    """Render a JSON-serialisable value as pretty-printed <pre>."""
    if obj is None:
        return "<em>None</em>"
    if isinstance(obj, str):
        return _pre(obj)
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        text = str(obj)
    return _pre(text)


def _outcome_badge(hit: bool, length: int, target: int) -> str:
    if hit:
        return f'<span class="badge badge-hit">Pass ({length}/{target})</span>'
    return f'<span class="badge badge-miss">Fail ({length}/{target})</span>'


def _length_class(length: int, target: int = 241) -> str:
    """Return CSS class for a length badge based on diff from target."""
    diff = abs(length - target)
    if diff == 0:
        return "len-hit"
    if diff <= 10:
        return "len-near"
    return "len-far"


def _render_length_badge(length: int, target: int = 241, title: str = "") -> str:
    cls = _length_class(length, target)
    title_attr = f' title="{esc(title)}"' if title else ""
    return f'<span class="len-badge {cls}"{title_attr}>{length}</span>'


def _extract_attempt_lengths(trajectory: list) -> list[dict]:
    """Pull each writer→length_checker pair from a trajectory.

    Returns list of {"length": int, "diff": int, "hit": bool, "attempt": str}
    in execution order. `attempt` is a human-readable label (e.g. "R1.2",
    "attempt 3", "writer #5") inferred from the length_checker step's purpose
    string.
    """
    out: list[dict] = []
    for step in trajectory:
        if step.get("role") != "length_checker":
            continue
        outp = step.get("output") or {}
        if not isinstance(outp, dict) or "length" not in outp:
            continue
        out.append({
            "length": outp.get("length", 0),
            "diff": outp.get("diff", 0),
            "hit": outp.get("hit", False),
            "purpose": step.get("purpose", ""),
        })
    return out


def _render_attempts_strip(attempts: list, target: int = 241) -> str:
    """Render a row of length badges representing every writer attempt."""
    if not attempts:
        return ""
    parts = []
    for i, a in enumerate(attempts):
        title = f"{a.get('purpose', '')}: len={a['length']}, diff={a['diff']:+d}"
        parts.append(_render_length_badge(a["length"], target=target, title=title))
        if i < len(attempts) - 1:
            parts.append('<span class="attempts-arrow">→</span>')
    return f'<span class="attempts-strip">{"".join(parts)}</span>'


# ---------------------------------------------------------------------------
# Section 1: Overview table
# ---------------------------------------------------------------------------

def render_overview(summaries: dict) -> str:
    from runners.story_task._metrics import overall_counts
    rows = []
    rows.append("""
    <table class="overview">
      <thead>
        <tr>
          <th>Method</th>
          <th>Pass rate</th>
          <th>Total tokens (in / out)</th>
          <th>Per-role tokens</th>
          <th>Started → Finished</th>
        </tr>
      </thead>
      <tbody>
    """)
    for m in METHODS:
        s = summaries.get(m)
        if s is None:
            rows.append(f"""
              <tr>
                <td><strong>{esc(m)}</strong></td>
                <td colspan="4"><em>(not run yet)</em></td>
              </tr>
            """)
            continue
        if "_error" in s:
            rows.append(f"""
              <tr>
                <td><strong>{esc(m)}</strong></td>
                <td colspan="4"><em>error: {esc(s["_error"])}</em></td>
              </tr>
            """)
            continue

        t = s.get("totals", {})
        in_tok = t.get("tokens_input", 0)
        out_tok = t.get("tokens_output", 0)

        oc = overall_counts(s, m)

        def _fmt(num: int, den: int) -> str:
            return (f"{num} / {den} ({num/den:.0%})"
                    if den else "(no data)")

        # Stacked rate cell: run + cyc always; val for method_brain only.
        rate_lines = [
            f"<div><span class='rate-label'>run:</span> "
            f"{_fmt(oc['runs_hits'], oc['runs_total'])}</div>",
            f"<div><span class='rate-label'>cyc:</span> "
            f"{_fmt(oc['cycle_hits'], oc['cycle_total'])}</div>",
        ]
        if _is_brain_method(m):
            rate_lines.append(
                f"<div><span class='rate-label'>val:</span> "
                f"{_fmt(oc['validated_hits'], oc['cycle_total'])}</div>"
            )
        rate_cell = "".join(rate_lines)

        per_role_html = []
        for role_name, m_meta in (t.get("tokens_by_role", {}) or {}).items():
            per_role_html.append(
                f"<div><code>{esc(role_name)}</code>: "
                f"{m_meta['calls']} calls, "
                f"in={m_meta['input_tokens']}, "
                f"out={m_meta['output_tokens']}</div>"
            )
        per_role_str = "".join(per_role_html) or "<em>(none)</em>"

        started = s.get("started_at", "?")
        finished = s.get("finished_at", "?")

        rows.append(f"""
          <tr>
            <td><strong>{esc(m)}</strong></td>
            <td>{rate_cell}</td>
            <td>{in_tok:,} / {out_tok:,}</td>
            <td>{per_role_str}</td>
            <td>{esc(started)}<br>{esc(finished)}</td>
          </tr>
        """)
    rows.append("</tbody></table>")
    return "".join(rows)


# ---------------------------------------------------------------------------
# Section 2: Per-theme hit grid
# ---------------------------------------------------------------------------

def render_theme_grid(summaries: dict) -> str:
    # Collect themes from whichever summary has them
    themes: list[tuple[str, str]] = []
    seen: set = set()
    for m in METHODS:
        s = summaries.get(m)
        if not s or "_error" in s:
            continue
        for t in s.get("config", {}).get("themes", []):
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                themes.append((tid, t.get("desc", "")))

    if not themes:
        return "<p><em>No themes found in any summary.</em></p>"

    # Build a lookup: method → theme_id → list of results sorted by run_idx
    by_method_theme: dict = {}
    for m in METHODS:
        s = summaries.get(m)
        if not s or "_error" in s:
            continue
        by_method_theme[m] = {}
        for r in s.get("results", []):
            tid = r.get("theme_id")
            by_method_theme[m].setdefault(tid, []).append(r)
        for tid in by_method_theme[m]:
            by_method_theme[m][tid].sort(key=lambda x: x.get("run_idx", 0))

    cells = ['<table class="theme-grid"><thead><tr><th>Theme</th>']
    for m in METHODS:
        cells.append(f"<th>{esc(m)}</th>")
    cells.append("</tr></thead><tbody>")
    for tid, tdesc in themes:
        cells.append(f"<tr><td><strong>{esc(tid)}</strong><br><small>{esc(tdesc)}</small></td>")
        for m in METHODS:
            if m not in by_method_theme:
                cells.append("<td><em>—</em></td>")
                continue
            runs = by_method_theme[m].get(tid, [])
            # Render each run as a small badge
            badges = []
            hit_count = 0
            for r in runs:
                hit = r.get("hit", False)
                length = r.get("final_length", 0)
                if hit:
                    hit_count += 1
                    badges.append(f'<span class="mini-badge mini-hit" title="run {r.get("run_idx")}">{length}</span>')
                else:
                    badges.append(f'<span class="mini-badge mini-miss" title="run {r.get("run_idx")}">{length}</span>')
            badge_str = " ".join(badges) if badges else "<em>(no runs)</em>"
            cells.append(
                f"<td><div class='hit-summary'>{hit_count}/{len(runs)} pass</div>"
                f"<div class='badge-strip'>{badge_str}</div></td>"
            )
        cells.append("</tr>")
    cells.append("</tbody></table>")
    return "".join(cells)


# ---------------------------------------------------------------------------
# Section 3: Per-run drill-down
# ---------------------------------------------------------------------------

def _format_workflow_args(args) -> str:
    """Compactly format a call node's args dict for tree display."""
    if not isinstance(args, dict) or not args:
        return ""
    pairs = []
    for k, v in args.items():
        if isinstance(v, str):
            # Truncate long string values inline
            v_disp = v if len(v) <= 60 else v[:57] + "..."
            pairs.append(f"{k}={v_disp!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            pairs.append(f"{k}={v}")
        else:
            import json as _j
            v_str = _j.dumps(v, ensure_ascii=False)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            pairs.append(f"{k}={v_str}")
    return ", ".join(pairs)


def render_workflow_tree(node) -> str:
    """Render a workflow DSL node (and recursively its children) as nested HTML.

    Used for method_brain to visualize what brain designed. Compact and
    indented; long arg values are truncated."""
    if not isinstance(node, dict):
        return f'<div class="wf-node"><em>{esc(repr(node))}</em></div>'

    t = node.get("type", "?")

    # Build the node header (type + key attrs)
    header_bits = [f'<span class="wf-type wf-type-{t}">{esc(t)}</span>']
    if t == "loop":
        if "max_iter" in node:
            header_bits.append(f'<span class="wf-attr">max_iter={node["max_iter"]}</span>')
        if "until" in node:
            header_bits.append(f'<span class="wf-attr">until: <code>{esc(node["until"])}</code></span>')
    elif t == "if":
        header_bits.append(f'<span class="wf-attr">condition: <code>{esc(node.get("condition", ""))}</code></span>')
    elif t == "call":
        target = node.get("role") or node.get("tool") or "?"
        kind = "role" if "role" in node else ("tool" if "tool" in node else "?")
        header_bits.append(f'<span class="wf-attr">{kind}=<code>{esc(target)}</code></span>')
        args_str = _format_workflow_args(node.get("args", {}))
        if args_str:
            header_bits.append(f'<span class="wf-args">args: <code>{esc(args_str)}</code></span>')
        if "save_as" in node:
            header_bits.append(f'<span class="wf-save">→ <code>${esc(node["save_as"])}</code></span>')
    elif t == "return":
        if "value" in node:
            header_bits.append(f'<span class="wf-attr">value: <code>{esc(node["value"])}</code></span>')
    elif t == "sequence":
        n = len(node.get("steps", []))
        header_bits.append(f'<span class="wf-attr">{n} step{"s" if n != 1 else ""}</span>')

    header = " &middot; ".join(header_bits)

    # Recurse into children based on node type
    children: list = []   # list of (label, child_node)
    if t == "sequence":
        for i, s in enumerate(node.get("steps", []) or []):
            children.append((f"step {i+1}", s))
    elif t == "loop":
        body = node.get("body")
        if body is not None:
            children.append(("body", body))
    elif t == "if":
        if "then" in node:
            children.append(("then", node["then"]))
        if "else" in node:
            children.append(("else", node["else"]))
    # call / return are leaves

    if children:
        child_html = []
        for label, child in children:
            child_html.append(
                f'<div class="wf-child"><span class="wf-label">{esc(label)}:</span>'
                f'{render_workflow_tree(child)}</div>'
            )
        return f'<div class="wf-node">{header}<div class="wf-children">{"".join(child_html)}</div></div>'
    return f'<div class="wf-node wf-leaf">{header}</div>'


def _step_length_info(step, target_len: int) -> str:
    """Return inline length/diff badge HTML for a step's summary, if applicable.
    Tokens are NOT shown per step — focus is length progression."""
    role = step.get("role")
    output = step.get("output")
    if role == "length_checker" and isinstance(output, dict) and "length" in output:
        length = output.get("length", 0)
        diff = output.get("diff", 0)
        hit = output.get("hit", False)
        diff_str = "diff=0" if hit else f"diff={diff:+d}"
        status = "Pass" if hit else "Fail"
        cls = _length_class(length, target_len)
        return (
            f' <span class="step-len {cls}">'
            f'len={length} · {diff_str} · {status}</span>'
        )
    if role == "writer" and isinstance(output, str):
        return f' <span class="step-len-out">output: {len(output)} chars</span>'
    return ""


def render_trajectory(trajectory: list, target_len: int) -> str:
    """Render a flat trajectory (list of step dicts) as nested collapsible HTML."""
    if not trajectory:
        return "<p><em>(no trajectory recorded)</em></p>"
    parts = ['<div class="trajectory">']
    for step in trajectory:
        role = step.get("role", "?")
        purpose = step.get("purpose", "")
        role_class = f"role-{role}"
        inp = step.get("input")
        out = step.get("output")
        len_info = _step_length_info(step, target_len)

        parts.append(f"""
          <details class="step {role_class}">
            <summary>
              <span class="step-num">#{step.get("step", "?")}</span>
              <span class="step-role">[{esc(role)}]</span>
              <span class="step-purpose">{esc(purpose)}</span>
              {len_info}
            </summary>
            <div class="step-body">
              <div class="step-section">
                <div class="section-label">input</div>
                {_json_pre(inp)}
              </div>
              <div class="step-section">
                <div class="section-label">output</div>
                {_json_pre(out)}
              </div>
            </div>
          </details>
        """)
    parts.append("</div>")
    return "".join(parts)


def _render_memory_block(memory: list, target_len: int) -> str:
    """Render the cross-run memory that brain saw for this run.

    Each entry has two cycle snapshots:
      {run_idx, cycles_used, hit, final_length,
       last_cycle: {cycle, final_length, hit, strategy_notes, workflow_json},
       nearest_cycle: {same fields}}

    Backwards-compat: older summaries had a flat shape (workflow_json /
    strategy_notes at the top level); render those by falling back.
    """
    if not memory:
        return ""

    def _snapshot_html(label: str, snap: dict | None) -> str:
        if not snap:
            return ""
        cyc = snap.get("cycle", "?")
        fl = snap.get("final_length", 0)
        passed = snap.get("hit", False)
        cls = _length_class(fl, target_len)
        badge = (f'<span class="step-len {cls}">cycle {cyc} · len={fl} · '
                 f'{"Pass" if passed else "Fail"}</span>')
        sn = (snap.get("strategy_notes") or "").strip()
        sn_html = (f'<div class="mem-strategy"><em>{esc(sn)}</em></div>'
                   if sn else '')
        wf = snap.get("workflow_json")
        wf_html = (
            f'<details class="cycle-workflow"><summary>{esc(label)} workflow</summary>'
            f'<div class="workflow-tree">{render_workflow_tree(wf)}</div>'
            f'</details>'
        ) if wf else ''
        return (
            f'<div class="mem-snapshot">'
            f'<div class="mem-snapshot-label">{esc(label)}</div>'
            f'{badge}{sn_html}{wf_html}'
            f'</div>'
        )

    rows = []
    for m in memory:
        run_idx = m.get("run_idx", "?")
        cycles_used = m.get("cycles_used")
        hit = m.get("hit", False)
        length = m.get("final_length", 0)
        outcome_badge = _outcome_badge(hit, length, target_len)
        cycles_note = (f' <span class="small">({cycles_used} cycles used)</span>'
                       if cycles_used else '')

        # New shape with nested snapshots; fall back to flat shape if missing.
        last_c = m.get("last_cycle")
        near_c = m.get("nearest_cycle")
        had_validated = m.get("had_validated_strategy", True)

        if not had_validated:
            # Run had no strategy-validated cycle. Pass (if any) was luck.
            snapshots_html = (
                '<div class="mem-snapshot mem-snapshot-luck">'
                '<em>No validated cycle in this run — Pass (if any) was '
                'first-writer cold-start luck; no strategy to learn from.</em>'
                '</div>'
            )
        elif last_c or near_c:
            snapshots_html = _snapshot_html("LAST cycle (validated)", last_c)
            if near_c and last_c and near_c.get("cycle") != last_c.get("cycle"):
                snapshots_html += _snapshot_html("NEAREST cycle (validated)", near_c)
            elif near_c and last_c and near_c.get("cycle") == last_c.get("cycle"):
                snapshots_html += ('<div class="mem-snapshot mem-snapshot-same">'
                                   'NEAREST cycle is the same as LAST cycle.'
                                   '</div>')
        else:
            # Legacy flat shape
            flat = {
                "cycle": "?",
                "final_length": length,
                "hit": hit,
                "strategy_notes": m.get("strategy_notes", ""),
                "workflow_json": m.get("workflow_json"),
            }
            snapshots_html = _snapshot_html("last-cycle (legacy)", flat)

        rows.append(
            f'<div class="mem-row">'
            f'<span class="mem-run">run {esc(run_idx)}</span> '
            f'{outcome_badge}{cycles_note}'
            f'{snapshots_html}'
            f'</div>'
        )
    return (
        '<details class="extra mem-extra" open>'
        f'<summary>brain saw memory: {len(memory)} previous run(s) '
        '(last + nearest cycle)</summary>'
        f'<div class="mem-list">{"".join(rows)}</div>'
        '</details>'
    )


def render_method_run(method: str, run: dict, target_len: int,
                      method_config: dict | None = None) -> str:
    """Render one (method, run) block — outcome + trajectory + extras."""
    hit = run.get("hit", False)
    length = run.get("final_length", 0)
    badge = _outcome_badge(hit, length, target_len)
    writer_calls = run.get("writer_calls_used", "?")
    err = run.get("error")
    err_html = f'<div class="err">error: {esc(err)}</div>' if err else ""

    # Method-specific high-level fields
    extras = []
    # method_fixed: per-cycle view. One cycle = (writer-verify × 3, textplanner,
    # writer-verify × 1). Show pre-textplanner attempts → textplanner advice →
    # post-textplanner attempt(s) — full trajectory is also expandable below.
    if method == "method_fixed":
        cycles = run.get("cycles", [])
        trailing = run.get("trailing_attempt")
        cfg = method_config or {}
        wpc = cfg.get("writers_per_cycle", 3)

        def _attempt_badge(att):
            L = att.get("length", 0); d = att.get("diff", 0)
            hit_a = att.get("hit", False)
            cls = _length_class(L, target_len)
            label = "Pass" if hit_a else "Fail"
            return (f'<span class="step-len {cls}">len={L} · '
                    f'diff={d:+d} · {label}</span>')

        if cycles or trailing:
            extras.append('<details class="extra" open><summary>per-cycle textplanner coaching</summary>')
            for c in cycles:
                cycle_idx = c.get("cycle", "?")
                attempts = c.get("attempts", [])
                advice = (c.get("textplanner_advice") or "").strip()

                attempts_html = " ".join(_attempt_badge(a) for a in attempts) \
                    or "<em>(none)</em>"

                if advice:
                    advice_html = (
                        f"<div class='cycle-phase'>"
                        f"<span class='cycle-label'>textplanner advice "
                        f"(used by next writer call):</span>"
                        f"<pre class='iter-advice'>{esc(advice)}</pre>"
                        f"</div>"
                    )
                else:
                    advice_html = (
                        f"<div class='cycle-phase'>"
                        f"<span class='cycle-label'>textplanner:</span> "
                        f"<em>(did not run — earlier Pass)</em>"
                        f"</div>"
                    )

                extras.append(
                    f"<div class='cycle-row'>"
                    f"<strong>cycle {cycle_idx}</strong>"
                    f"<div class='cycle-phase'>"
                    f"<span class='cycle-label'>writer-verify × {len(attempts)} "
                    f"(target {wpc}):</span> {attempts_html}"
                    f"</div>"
                    f"{advice_html}"
                    f"</div>"
                )
            if trailing is not None:
                extras.append(
                    f"<div class='cycle-row trailing-row'>"
                    f"<strong>trailing writer-verify</strong> "
                    f"<span class='small'>(consumes the last textplanner's advice)</span>"
                    f"<div class='cycle-phase'>{_attempt_badge(trailing)}</div>"
                    f"</div>"
                )
            extras.append("</details>")
    # method_brain (and brain_ablation_* variants) — cross-run memory +
    # per-cycle workflow tree + strategy_notes + luck-Pass badge.
    if _is_brain_method(method):
        mem = run.get("memory_seen") or []
        if mem:
            extras.append(_render_memory_block(mem, target_len))

        cycles = run.get("cycles") or []
        if cycles:
            extras.append(
                '<details class="extra" open><summary>per-cycle brain workflows '
                f'({len(cycles)} cycle{"s" if len(cycles) != 1 else ""} executed)'
                '</summary>'
            )
            for c in cycles:
                ci = c.get("cycle", "?")
                wf = c.get("workflow_json")
                sn = (c.get("strategy_notes") or "").strip()
                fl = c.get("final_length", 0)
                hit_c = c.get("hit", False)
                wcalls = c.get("writer_calls_used_in_cycle", 0)
                # Default True for legacy data (no strategy_validated field).
                validated = c.get("strategy_validated", True)
                err_c = c.get("error")

                cls = _length_class(fl, target_len)
                label = "Pass" if hit_c else "Fail"
                badge = (f'<span class="step-len {cls}">len={fl} · '
                         f'{label}</span>')

                # Flag luck-Pass cycles so user can scan for false positives:
                # Pass + only 1 writer call = first-writer cold-start luck,
                # strategy never had a chance to run.
                if hit_c and not validated:
                    validation_badge = (
                        '<span class="luck-badge" title="Pass came from '
                        'writer #1 cold-start luck — the planned strategy '
                        'did not run">luck-Pass</span>'
                    )
                else:
                    validation_badge = ""

                err_html_c = (f"<div class='err'>error: {esc(err_c)}</div>"
                              if err_c else "")
                sn_html = (f"<div class='strategy-notes'>strategy_notes: "
                           f"<em>{esc(sn)}</em></div>" if sn else "")
                wf_html = (
                    f"<details class='cycle-workflow'><summary>workflow tree</summary>"
                    f"<div class='workflow-tree'>{render_workflow_tree(wf)}</div>"
                    f"</details>"
                ) if wf else (
                    "<div class='small'><em>(no workflow — unparseable)</em></div>"
                )

                extras.append(
                    f"<div class='brain-cycle-row'>"
                    f"<strong>cycle {ci}</strong> {badge} {validation_badge} "
                    f"<span class='small'>writer×{wcalls}</span>"
                    f"{err_html_c}"
                    f"{sn_html}"
                    f"{wf_html}"
                    f"</div>"
                )
            extras.append('</details>')

    # Attempts strip (visible without expanding): each writer→verify attempt's
    # length as a colored badge in execution order. This is the main "see
    # length progression at a glance" UX win.
    attempts = _extract_attempt_lengths(run.get("trajectory", []))
    strip_html = _render_attempts_strip(attempts, target=target_len)

    summary_html = (
        f"<strong>{esc(method)}</strong> &middot; {badge}"
        f"{strip_html}"
        f" &middot; <span class='small'>writer×{writer_calls}</span>"
    )

    body = (
        err_html
        + "".join(extras)
        + render_trajectory(run.get("trajectory", []), target_len)
    )

    return f"""
      <details class="method-run" data-method="{esc(method)}">
        <summary>{summary_html}</summary>
        <div class="method-body">{body}</div>
      </details>
    """


def render_drilldown(summaries: dict) -> str:
    """Per (theme, run) blocks, each containing 3 method blocks."""
    # Collect themes
    themes: list[tuple[str, str]] = []
    seen: set = set()
    for m in METHODS:
        s = summaries.get(m)
        if not s or "_error" in s:
            continue
        for t in s.get("config", {}).get("themes", []):
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                themes.append((tid, t.get("desc", "")))

    # Build lookup: (method, theme_id, run_idx) → run dict
    by_method_theme_run: dict = {}
    runs_per_theme = 1
    for m in METHODS:
        s = summaries.get(m)
        if not s or "_error" in s:
            continue
        for r in s.get("results", []):
            key = (m, r.get("theme_id"), r.get("run_idx"))
            by_method_theme_run[key] = r
        rpt = s.get("config", {}).get("runs_per_theme", 1)
        runs_per_theme = max(runs_per_theme, rpt)

    target_len = 241
    for s in summaries.values():
        if s and "_error" not in s:
            target_len = s.get("config", {}).get("target_len", 241)
            break

    parts = []
    for tid, tdesc in themes:
        parts.append(f"""
          <details class="theme-block" open>
            <summary><span class="theme-id">{esc(tid)}</span>
              <span class="theme-desc">{esc(tdesc)}</span></summary>
        """)
        for run_idx in range(1, runs_per_theme + 1):
            parts.append(f"""
              <details class="run-block">
                <summary>run {run_idx}</summary>
                <div class="run-methods">
            """)
            for m in METHODS:
                run = by_method_theme_run.get((m, tid, run_idx))
                if run is None:
                    parts.append(
                        f"<div class='method-missing'><strong>{esc(m)}</strong>: "
                        f"<em>(not run yet)</em></div>"
                    )
                else:
                    m_cfg = (summaries.get(m) or {}).get("config", {})
                    parts.append(render_method_run(m, run, target_len, m_cfg))
            parts.append("</div></details>")
        parts.append("</details>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1400px; margin: 24px auto; padding: 0 24px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  h3 { margin-top: 24px; color: #444; }
  table { border-collapse: collapse; margin: 12px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left;
           vertical-align: top; }
  th { background: #f5f5f5; }
  .overview td { font-size: 13px; }
  .rate-label { display: inline-block; min-width: 28px; color: #6c757d;
                font-family: ui-monospace, monospace; font-size: 11px;
                font-weight: 600; }
  .theme-grid td { min-width: 180px; }
  .hit-summary { font-weight: bold; }
  .badge-strip { margin-top: 6px; }
  .mini-badge { display: inline-block; padding: 2px 6px; margin: 1px;
                font-size: 11px; border-radius: 3px; font-family: monospace; }
  .mini-hit { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
  .mini-miss { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
           font-size: 12px; font-family: monospace; font-weight: bold; }
  .badge-hit { background: #28a745; color: white; }
  .badge-miss { background: #dc3545; color: white; }
  .small { font-size: 12px; color: #666; }
  .err { color: #c62828; background: #fff3f3; padding: 6px; margin: 8px 0;
         border-left: 3px solid #c62828; }
  details { margin: 6px 0; }
  summary { cursor: pointer; padding: 4px 0; }
  summary:hover { background: #f0f0f0; }
  .theme-block > summary { font-size: 16px; padding: 8px;
                            background: #e9ecef; }
  .theme-id { font-weight: bold; }
  .theme-desc { color: #666; margin-left: 8px; }
  .run-block { margin-left: 16px; padding: 4px 8px;
               background: #f8f9fa; border-left: 3px solid #6c757d; }
  .run-block > summary { font-weight: bold; }
  .run-methods { padding-left: 16px; }
  .method-run { margin: 4px 0; padding: 4px 8px; border-left: 2px solid #007bff; }
  .method-run[data-method="method_fixed"] { border-left-color: #28a745; }
  .method-run[data-method="method_brain"] { border-left-color: #6f42c1; }
  .method-body { padding: 8px 0 8px 16px; }
  .method-missing { padding: 4px 8px; color: #888; }
  .strategy-notes { background: #fff3cd; padding: 6px; margin: 4px 0;
                    border-left: 3px solid #ffc107; font-size: 13px; }
  .extra { margin: 8px 0; padding: 4px 8px; background: #f1f3f5; }
  .round { margin: 2px 0; padding: 2px 4px; }
  .trajectory { margin-top: 8px; }
  .step { margin: 2px 0; padding: 2px 8px; border-left: 2px solid #ced4da; }
  /* Brain "design cycle N workflow" steps stand out so you can scan where
     each cycle's planning happens: thick purple left bar + lavender fill. */
  .step.role-brain { border-left: 5px solid #6f42c1; background: #f3eefb;
                     padding: 6px 8px; margin: 8px 0; border-radius: 4px; }
  .step.role-brain > summary { font-weight: bold; color: #4a2a9c; }
  .step.role-textplanner { border-left-color: #e83e8c; }
  .step.role-writer { border-left-color: #007bff; }
  .step.role-length_checker { border-left-color: #28a745; }
  .step.role-system { border-left-color: #dc3545; background: #fff3f3; }
  .step > summary { font-family: monospace; font-size: 13px; }
  .step-num { color: #999; }
  .step-role { color: #6f42c1; font-weight: bold; margin: 0 4px; }
  .step-purpose { color: #444; }
  .tok { color: #888; font-size: 11px; margin-left: 8px; }
  .step-body { padding: 6px 0 6px 12px; }
  .step-section { margin: 4px 0; }
  .section-label { font-size: 11px; font-weight: bold; color: #666;
                   text-transform: uppercase; }
  pre { background: #f8f9fa; padding: 8px; margin: 4px 0;
        border: 1px solid #e9ecef; border-radius: 3px;
        font-family: ui-monospace, monospace; font-size: 12px;
        white-space: pre-wrap; word-wrap: break-word; max-height: 400px;
        overflow-y: auto; }
  code { font-family: ui-monospace, monospace; background: #f1f3f5;
         padding: 1px 4px; border-radius: 2px; font-size: 12px; }
  /* Length badges (3-tier color palette by diff from target) */
  .len-badge { display: inline-block; min-width: 32px; padding: 2px 6px;
               margin: 0 2px; font-size: 11px; font-family: ui-monospace, monospace;
               border-radius: 3px; text-align: center; font-weight: 600; }
  .len-hit  { background: #28a745; color: white;  border: 1px solid #1e7e34; }
  .len-near { background: #fd7e14; color: white;  border: 1px solid #d96907; }
  .len-far  { background: #dc3545; color: white;  border: 1px solid #b21f2d; }
  .attempts-strip { display: inline-flex; gap: 2px; flex-wrap: wrap;
                    margin-left: 8px; vertical-align: middle; }
  .attempts-arrow { color: #aaa; font-size: 11px; margin: 0 1px; }
  .step-len { font-family: ui-monospace, monospace; font-size: 11px;
              margin-left: 8px; padding: 1px 6px; border-radius: 3px;
              background: #f1f3f5; color: #222; font-weight: 600; }
  /* Override colored bg from .len-* on inline step-len: keep dark text on
     pale tinted backgrounds so length+diff stay readable in trajectory. */
  .step-len.len-hit  { background: #d4edda; color: #155724;
                       border: 1px solid #c3e6cb; }
  .step-len.len-near { background: #ffe8d1; color: #8a3e00;
                       border: 1px solid #fcd5a8; }
  .step-len.len-far  { background: #f8d7da; color: #721c24;
                       border: 1px solid #f5c6cb; }
  .step-len-out { color: #777; font-size: 11px; margin-left: 8px; }
  /* Workflow tree (method_brain) */
  .workflow-tree { font-family: ui-monospace, monospace; font-size: 12px;
                   background: #fefefe; padding: 8px; border-radius: 4px;
                   border: 1px solid #e9ecef; margin: 8px 0; }
  .wf-node { padding: 3px 0; }
  .wf-children { padding-left: 16px; border-left: 1px dashed #ced4da;
                 margin-left: 4px; margin-top: 2px; }
  .wf-child { padding: 2px 0; }
  .wf-label { color: #999; margin-right: 6px; font-style: italic; }
  .wf-type { font-weight: bold; color: #444; }
  .wf-type-loop { color: #6f42c1; }
  .wf-type-sequence { color: #0056b3; }
  .wf-type-if { color: #ff8c00; }
  .wf-type-call { color: #28a745; }
  .wf-type-return { color: #dc3545; }
  .wf-attr { color: #555; }
  .wf-args { color: #666; }
  .wf-save { color: #888; }
  .wf-leaf { background: #f8f9fa; padding: 4px 6px; border-radius: 3px;
             border: 1px solid #e9ecef; display: inline-block; margin: 1px 0; }
  /* Per-cycle textplanner coaching block (method_fixed) */
  .cycle-row { padding: 8px 10px; margin: 6px 0; background: white;
               border: 1px solid #f1e0eb; border-radius: 4px; }
  .cycle-row > strong { color: #b03070; }
  .cycle-row.trailing-row { background: #fff8f0; border-color: #f5c98a; }
  .cycle-row.trailing-row > strong { color: #b35900; }
  .cycle-phase { margin: 6px 0 2px 0; }
  .cycle-label { font-size: 11px; color: #6c757d; font-weight: 600;
                 text-transform: uppercase; margin-right: 6px; }
  .iter-advice { margin-top: 4px; padding: 6px 8px; background: #fff5fa;
                 border-left: 3px solid #e83e8c; font-size: 12px;
                 white-space: pre-wrap; word-wrap: break-word;
                 max-height: 240px; overflow-y: auto; }
  /* Cross-run memory block (method_brain) */
  .mem-extra { background: #e7f5ff; border-left: 3px solid #1c7ed6; }
  .mem-list { padding: 4px 0; }
  .mem-row { padding: 6px 8px; margin: 4px 0; background: white;
             border: 1px solid #c5e4fa; border-radius: 3px; }
  .mem-run { font-weight: bold; color: #1c7ed6; margin-right: 6px; }
  .mem-strategy { margin-top: 4px; font-size: 12px; color: #444;
                  font-family: ui-monospace, monospace; padding-left: 8px; }
  .mem-empty { color: #999; font-style: italic; }
  .mem-snapshot { margin-top: 6px; padding: 6px 8px; background: #f8fbff;
                  border: 1px solid #d8e8f5; border-radius: 3px; }
  .mem-snapshot-label { font-size: 11px; font-weight: 700; color: #1c7ed6;
                        text-transform: uppercase; margin-bottom: 3px; }
  .mem-snapshot-same { font-size: 12px; color: #888; font-style: italic;
                       padding: 4px 8px; }
  .mem-snapshot-luck { font-size: 12px; color: #8a6a3e; padding: 6px 8px;
                       background: #fff8e6; border: 1px solid #f5d989;
                       border-radius: 3px; margin-top: 6px; }
  .luck-badge { display: inline-block; padding: 1px 6px; margin: 0 4px;
                font-size: 11px; font-weight: 600; color: #8a6a3e;
                background: #fff3cd; border: 1px solid #f5d989;
                border-radius: 3px; font-family: ui-monospace, monospace; }
  /* Per-cycle brain workflow rows (method_brain) */
  .brain-cycle-row { padding: 8px 10px; margin: 6px 0; background: white;
                     border: 1px solid #e2d9f3; border-radius: 4px; }
  .brain-cycle-row > strong { color: #6f42c1; }
  .cycle-workflow { margin-top: 6px; }
  .cycle-workflow > summary { font-size: 12px; color: #6f42c1; }
"""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_html(summaries: dict) -> str:
    rendered_at = datetime.now().isoformat(timespec="seconds")
    run_methods = [m for m in METHODS if summaries.get(m) and "_error" not in summaries[m]]
    missing = [m for m in METHODS if not summaries.get(m) or "_error" in (summaries.get(m) or {})]

    status_line = (
        f"Methods loaded: <code>{', '.join(run_methods) or '(none)'}</code>"
        + (f" &middot; Missing: <code>{', '.join(missing)}</code>" if missing else "")
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Story Task — 3-method Comparison</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>Story Task — Comparison: baseline vs method_fixed vs method_brain</h1>
  <p class="small">Rendered {esc(rendered_at)}. {status_line}</p>

  <h2>Overview</h2>
  {render_overview(summaries)}

  <h2>Per-theme hit grid</h2>
  <p class="small">Each cell shows hit count + per-run length badges
  (<span class="mini-badge mini-hit">241</span> green = Pass,
  <span class="mini-badge mini-miss">256</span> red = Fail).</p>
  {render_theme_grid(summaries)}

  <h2>Per-run drill-down</h2>
  <p class="small">Themes are open by default. Within each run, the 3 methods
  are stacked. Click a method to see its full trajectory (every step's input
  &amp; output captured). Sub-steps are collapsed by default.</p>
  {render_drilldown(summaries)}

</body>
</html>
"""


def main(output_path: str | None = None) -> str:
    """Render the comparison HTML. Returns the path written.

    Re-reads STORY_EXP_NAME at call time (so auto-render from runner scripts
    sees the current experiment dir) and rediscovers methods present under
    BASE_DIR (the standard 3 + any brain_ablation_* variants found).

    output_path defaults to <BASE_DIR>/comparison.html.
    """
    global _EXP_NAME, BASE_DIR, METHODS
    _EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
    BASE_DIR = (os.path.join(WORKSHOP, "story_241", _EXP_NAME) if _EXP_NAME
                else os.path.join(WORKSHOP, "story_241"))
    METHODS = _discover_methods(BASE_DIR)

    summaries = load_summaries()
    html_doc = render_html(summaries)

    out = output_path or os.path.join(BASE_DIR, OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"HTML saved: {out}  (methods: {', '.join(METHODS) or 'none'})")
    return out


if __name__ == "__main__":
    main()
