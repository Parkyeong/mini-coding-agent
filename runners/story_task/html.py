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


METHODS = ["baseline", "method_fixed", "method_brain"]

# If STORY_EXP_NAME is set, base path becomes story_241/<exp_name>/, matching
# the per-method output dirs. Otherwise defaults to story_241/.
_EXP_NAME = os.environ.get("STORY_EXP_NAME", "").strip()
BASE_DIR = (
    os.path.join(WORKSHOP, "story_241", _EXP_NAME) if _EXP_NAME
    else os.path.join(WORKSHOP, "story_241")
)
OUTPUT_FILENAME = "comparison.html"


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
        return f'<span class="badge badge-hit">HIT ({length}/{target})</span>'
    return f'<span class="badge badge-miss">MISS ({length}/{target})</span>'


# ---------------------------------------------------------------------------
# Section 1: Overview table
# ---------------------------------------------------------------------------

def render_overview(summaries: dict) -> str:
    rows = []
    rows.append("""
    <table class="overview">
      <thead>
        <tr>
          <th>Method</th>
          <th>Hit rate</th>
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
        hits = t.get("hits", 0)
        runs = t.get("runs", 0)
        rate = t.get("hit_rate", 0)
        in_tok = t.get("tokens_input", 0)
        out_tok = t.get("tokens_output", 0)

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
            <td>{hits} / {runs} ({rate:.0%})</td>
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
                f"<td><div class='hit-summary'>{hit_count}/{len(runs)} hit</div>"
                f"<div class='badge-strip'>{badge_str}</div></td>"
            )
        cells.append("</tr>")
    cells.append("</tbody></table>")
    return "".join(cells)


# ---------------------------------------------------------------------------
# Section 3: Per-run drill-down
# ---------------------------------------------------------------------------

def render_trajectory(trajectory: list, target_len: int) -> str:
    """Render a flat trajectory (list of step dicts) as nested collapsible HTML."""
    if not trajectory:
        return "<p><em>(no trajectory recorded)</em></p>"
    parts = ['<div class="trajectory">']
    for step in trajectory:
        role = step.get("role", "?")
        purpose = step.get("purpose", "")
        tokens = step.get("tokens", {}) or {}
        tok_str = ""
        if tokens.get("in", 0) > 0 or tokens.get("out", 0) > 0:
            tok_str = f' <span class="tok">tokens: in={tokens.get("in", 0)}, out={tokens.get("out", 0)}</span>'
        role_class = f"role-{role}"

        # Special render for length_checker (output is dict — show inline summary)
        inp = step.get("input")
        out = step.get("output")

        parts.append(f"""
          <details class="step {role_class}">
            <summary>
              <span class="step-num">#{step.get("step", "?")}</span>
              <span class="step-role">[{esc(role)}]</span>
              <span class="step-purpose">{esc(purpose)}</span>
              {tok_str}
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


def render_method_run(method: str, run: dict, target_len: int) -> str:
    """Render one (method, run) block — outcome + trajectory + extras."""
    hit = run.get("hit", False)
    length = run.get("final_length", 0)
    badge = _outcome_badge(hit, length, target_len)
    writer_calls = run.get("writer_calls_used", "?")
    err = run.get("error")
    err_html = f'<div class="err">error: {esc(err)}</div>' if err else ""

    # Method-specific high-level fields
    extras = []
    # method_fixed: rounds + strategy_notes
    if method == "method_fixed":
        rounds = run.get("rounds", [])
        if rounds:
            extras.append('<details class="extra"><summary>brain rounds (high-level)</summary>')
            for ri, rnd in enumerate(rounds, 1):
                extras.append(
                    f"<div class='round'><strong>round {ri}</strong>: "
                    f"hit={rnd.get('hit')}, final_length={rnd.get('final_length')}, "
                    f"guidance: <em>{esc(rnd.get('guidance_used', ''))}</em></div>"
                )
                sn = rnd.get("strategy_notes")
                if sn:
                    extras.append(f"<div class='strategy-notes'>strategy_notes: <em>{esc(sn)}</em></div>")
            extras.append("</details>")
    # method_brain: workflow JSON + strategy_notes
    if method == "method_brain":
        wf = run.get("workflow")
        sn = run.get("strategy_notes")
        if sn:
            extras.append(f'<div class="strategy-notes">strategy_notes: <em>{esc(sn)}</em></div>')
        if wf:
            extras.append('<details class="extra"><summary>brain-designed workflow (JSON)</summary>')
            extras.append(_json_pre(wf))
            extras.append("</details>")

    by_role = run.get("tokens_by_role", {}) or {}
    role_tok_strs = []
    for rn, m_meta in by_role.items():
        role_tok_strs.append(
            f"<code>{esc(rn)}</code> {m_meta['calls']}× "
            f"({m_meta['input_tokens']}/{m_meta['output_tokens']})"
        )
    tok_summary = " &middot; ".join(role_tok_strs) if role_tok_strs else "(no calls)"

    summary_html = (
        f"<strong>{esc(method)}</strong> &middot; {badge} &middot; "
        f"<span class='small'>writer_calls={writer_calls}</span> &middot; "
        f"<span class='small'>{tok_summary}</span>"
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
                    parts.append(render_method_run(m, run, target_len))
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
  .step.role-brain { border-left-color: #6f42c1; }
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
  (<span class="mini-badge mini-hit">241</span> green = HIT,
  <span class="mini-badge mini-miss">256</span> red = MISS).</p>
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

    output_path defaults to WORKSHOP/story_241/comparison.html.
    """
    summaries = load_summaries()
    html_doc = render_html(summaries)

    out = output_path or os.path.join(BASE_DIR, OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"HTML saved: {out}")
    return out


if __name__ == "__main__":
    main()
