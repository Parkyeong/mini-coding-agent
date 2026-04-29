"""
Two-layer memory, three persisted files per benchmark instance:

- Working memory: created per task, lives only during a task. Holds the
  task-level event_log (every tool call/result during this task), the plan,
  and candidate facts. observations / files_changed are *derived* from the
  event_log, not stored directly. At end_task() its full state is dumped to
  working_memory.json (overwritten on next task), then the in-memory object
  is discarded.

- Long-term memory (MemoryManager): project_context + task_history (without
  event_log — that lives in working_memory.json) + this case's facts.

- Global facts (cross-case, optional): facts pooled across all instances of
  an experiment, with confidence reinforcement when the same fact is saved
  by multiple cases.
"""

import os
import json
from datetime import datetime
from typing import Optional

from config import (
    MAX_MEMORY_TASKS,
    MAX_MEMORY_FACTS,
    MAX_WORKING_OBSERVATIONS,
    WORKING_OBSERVATION_MAX_CHARS,
    FACT_INITIAL_CONFIDENCE,
    FACT_REINFORCE_DELTA,
    FACT_MAX_CONFIDENCE,
    FACT_GRACE_PERIOD_TASKS,
)


# ---------------------------------------------------------------------------
# Event Log — first-class event stream for one task
# ---------------------------------------------------------------------------

class EventLog:
    """Append-only log of mechanical events produced by the agent loop.

    Event kinds (current):
      - llm_call:    {role, input_tokens, output_tokens, latency}
      - text:        {content}
      - tool_call:   {name, args}
      - tool_result: {name, args, result}

    WorkingMemory derives observations / files_changed from this stream;
    metrics and debug logs can subscribe later without changing producers.
    """

    def __init__(self):
        self.events: list[dict] = []

    def append(self, kind: str, payload: dict) -> None:
        self.events.append({
            "kind": kind,
            "payload": payload,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })

    def filter(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

# Tool names that count as "looked at the workspace" (observation events).
_OBSERVATION_TOOLS = ("read_file", "list_dir")

# Tool names that mutate workspace files (file_changed events).
_FILE_MUTATION_TOOLS = ("write_file", "replace_in_file")


class WorkingMemory:
    def __init__(self, task_id: str, user_prompt: str):
        self.task_id = task_id
        self.user_prompt = user_prompt
        self.event_log = EventLog()
        self.plan: list[str] = []
        self.candidate_facts: list[dict] = []

    def set_plan(self, plan: list[str]) -> None:
        self.plan = list(plan)

    def add_candidate_fact(self, fact: str, category: str) -> None:
        norm = _normalize_fact(fact)
        if any(_normalize_fact(f["fact"]) == norm for f in self.candidate_facts):
            return
        self.candidate_facts.append({"fact": fact, "category": category})

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def files_changed(self) -> set[str]:
        paths = set()
        for e in self.event_log.filter("tool_result"):
            payload = e["payload"]
            if payload["name"] not in _FILE_MUTATION_TOOLS:
                continue
            fp = payload.get("args", {}).get("file_path")
            if fp:
                paths.add(fp)
        return paths

    def observations(self) -> list[dict]:
        obs = []
        for e in self.event_log.filter("tool_result"):
            payload = e["payload"]
            name = payload["name"]
            if name not in _OBSERVATION_TOOLS:
                continue
            args = payload.get("args", {})
            result = (payload.get("result") or "")[:200]
            if name == "read_file":
                content = f"{args.get('file_path')}: {result}"
            else:  # list_dir
                content = f"{args.get('dir_path', '.')}: {result}"
            obs.append({"kind": name, "content": content[:WORKING_OBSERVATION_MAX_CHARS]})
        return obs

    def snapshot_for_coder(self) -> str:
        observations = self.observations()
        if not observations and not self.plan:
            return "No observations or plan yet."

        parts = [f"[WorkingMemory] task_id = {self.task_id} "]
        if self.plan:
            parts.append("Current Plan:")
            for i, step in enumerate(self.plan, 1):
                parts.append(f" - {i}. {step}")

        if observations:
            parts.append("Recent observations from earlier steps:")
            for obs in observations[-MAX_WORKING_OBSERVATIONS:]:
                parts.append(f" - [{obs['kind']}] {obs['content']}")

        files_changed = self.files_changed
        if files_changed:
            parts.append(f"Files changed so far: {sorted(files_changed)}")

        return "\n".join(parts)


def _normalize_fact(fact: str) -> str:
    return " ".join(fact.strip().lower().split())


# ---------------------------------------------------------------------------
# Long-term Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Three-file layout (per benchmark instance):

      long_term_memory.json   ← required: project_context + task_history
                                  + this case's facts (filtered by source_task_id
                                  if global_facts_file is also set).
      working_memory.json     ← optional: full WorkingMemory snapshot, written
                                  at end_task. Holds event_log + plan +
                                  candidate_facts. Overwritten on each task end.
      global_facts.json       ← optional: cross-case fact pool with confidence
                                  reinforcement. Shared across all instances
                                  of one experiment.

    If `global_facts_file` is None, facts live entirely in long_term_memory.json
    (single-file mode, useful for one-off projects). If set, facts are pooled
    globally and the per-case file holds only that case's facts.
    """

    def __init__(
        self,
        long_term_file: str,
        working_memory_file: Optional[str] = None,
        global_facts_file: Optional[str] = None,
    ):
        self.long_term_file = long_term_file
        self.working_memory_file = working_memory_file
        self.global_facts_file = global_facts_file

        self.data = self._load()
        self._working: Optional[WorkingMemory] = None

    def _load(self) -> dict:
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = self._default_data()

        if self.global_facts_file is not None:
            data["facts"] = self._load_global_facts()

        return data

    def _load_global_facts(self) -> list:
        if not os.path.exists(self.global_facts_file):
            return []
        with open(self.global_facts_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload.get("facts", [])
        return payload

    def _save(self) -> None:
        if self.global_facts_file is not None:
            self._save_global_facts(self.data["facts"])
            # Per-case long_term_memory.json keeps only facts promoted by this
            # case's own tasks (filtered by source_task_id). The full pool
            # lives in global_facts.json.
            case_task_ids = {t["task_id"] for t in self.data["task_history"]}
            case_facts = [
                f for f in self.data["facts"]
                if f.get("source_task_id") in case_task_ids
            ]
            per_instance = {k: v for k, v in self.data.items() if k != "facts"}
            per_instance["facts"] = case_facts
            with open(self.long_term_file, 'w', encoding='utf-8') as f:
                json.dump(per_instance, f, indent=2, ensure_ascii=False)
        else:
            with open(self.long_term_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _save_global_facts(self, facts: list) -> None:
        os.makedirs(os.path.dirname(self.global_facts_file), exist_ok=True)
        payload = {
            "facts": facts,
            "updated_at": self._now(),
        }
        with open(self.global_facts_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _save_working_memory(self) -> None:
        """Dump the current WorkingMemory state to disk. Overwrites on each
        call — replans within a single task accumulate in the same WM, so
        the dump captures the complete trajectory of the task."""
        if self.working_memory_file is None or self._working is None:
            return
        wm = self._working
        payload = {
            "task_id": wm.task_id,
            "user_prompt": wm.user_prompt,
            "plan": wm.plan,
            "candidate_facts": wm.candidate_facts,
            "files_changed": sorted(wm.files_changed),
            "event_log": list(wm.event_log.events),
            "saved_at": self._now(),
        }
        os.makedirs(os.path.dirname(self.working_memory_file), exist_ok=True)
        with open(self.working_memory_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _default_data(self) -> dict:
        return {
            "project_context": {
                "project_name": "",
                "workspace": "",
                "language": "",
                "framework": "",
                "entry_file": "",
                "test_command": "",
                "test_timeout": None,
                "updated_at": "",
            },
            "task_history": [],
            "facts": [],
        }

    def update_project_context(self, **kwargs) -> None:
        self.data["project_context"].update(kwargs)
        self.data["project_context"]["updated_at"] = self._now()
        self._save()

    # working memory lifecycle

    def begin_task(self, task_id: str, user_prompt: str) -> WorkingMemory:
        self._working = WorkingMemory(task_id, user_prompt)
        return self._working

    def get_working(self) -> Optional[WorkingMemory]:
        return self._working

    def end_task(self, task_id: str, passed: bool, plan: list[str],
                 attempts: int, summary: str, error_history: list[dict] | None = None) -> None:
        wm = self._working
        files_changed = sorted(wm.files_changed) if wm else []

        current_task_idx = len(self.data["task_history"]) + 1

        if passed and wm:
            for cf in wm.candidate_facts:
                self._promote_fact(cf["fact"], cf["category"], task_id, current_task_idx)

        record = {
            "task_id": task_id,
            "timestamp": self._now(),
            "user_prompt": wm.user_prompt if wm else "",
            "plan": plan,
            "status": "passed" if passed else "failed",
            "files_changed": files_changed,
            "attempts": attempts,
            "summary": summary,
            # event_log lives in working_memory.json (full per-task trajectory),
            # not duplicated here.
        }
        if error_history:
            record["error_history"] = error_history

        self.data["task_history"].append(record)
        self._trim_task_history()

        self._evict_facts_if_needed(current_task_idx)

        # Snapshot working memory before clearing — captures every replan/retry
        # accumulated during this task.
        self._save_working_memory()

        self._save()

        self._working = None

    def _promote_fact(self, fact: str, category: str,
                      source_task_id: str, current_task_idx: int) -> None:
        norm = _normalize_fact(fact)
        for existing in self.data["facts"]:
            if _normalize_fact(existing["fact"]) == norm:
                existing["confidence"] = min(
                    existing["confidence"] + FACT_REINFORCE_DELTA,
                    FACT_MAX_CONFIDENCE,
                )
                existing["reinforce_count"] = existing.get("reinforce_count", 0) + 1
                existing["last_reinforced_at"] = self._now()
                existing.setdefault("reinforced_by", []).append(source_task_id)
                return

        self.data["facts"].append({
            "fact": fact,
            "category": category,
            "confidence": FACT_INITIAL_CONFIDENCE,
            "reinforce_count": 0,
            "created_at_task": current_task_idx,
            "source_task_id": source_task_id,
            "created_at": self._now(),
            "last_reinforced_at": self._now(),
            "reinforced_by": [source_task_id],
        })

    def _evict_facts_if_needed(self, current_task_idx: int) -> None:
        def age(f):
            return current_task_idx - f.get("created_at_task", current_task_idx)

        def score(f):
            return f.get("confidence", 0) * f.get("reinforce_count", 0)

        def sort_key(f):
            in_grace = age(f) < FACT_GRACE_PERIOD_TASKS
            return (in_grace, score(f), f.get("created_at_task", 0))

        while len(self.data["facts"]) > MAX_MEMORY_FACTS:
            victim = min(self.data["facts"], key=sort_key)
            self.data["facts"].remove(victim)

    def _trim_task_history(self) -> None:
        if len(self.data["task_history"]) > MAX_MEMORY_TASKS:
            self.data["task_history"] = self.data["task_history"][-MAX_MEMORY_TASKS:]

    def get_context_for_planner(self, max_facts: int = 10) -> str:
        parts = []

        ctx = self.data.get("project_context", {})
        non_empty = {k: v for k, v in ctx.items() if v and k not in ("updated_at",)}

        if non_empty:
            parts.append("Project info:" + ",".join(f"{k}={v}" for k, v in non_empty.items()))

        facts = sorted(
            self.data.get("facts", []),
            key=lambda f: f.get("confidence", 0),
            reverse=True,
        )[:max_facts]

        if facts:
            parts.append("Known facts about this project:")
            for f in facts:
                parts.append(f" - {f['category']}  confidence={f.get('confidence', 0):.2f}, {f['fact']}")

        history = self.data.get("task_history", [])[-5:]
        if history:
            parts.append("Recent tasks executed:")
            for h in history:
                parts.append(f" - [{h['status']}] {h['user_prompt']} (files changed: {h['files_changed']})")

        return "\n".join(parts)

    def _now(self) -> str:
        return datetime.now().isoformat(timespec='seconds')

    def generate_task_id(self) -> str:
        count = len(self.data.get("task_history", []))
        project = self.data.get("project_context", {}).get("project_name", "")
        if project:
            return f"{project}_{count + 1:04d}"
        return f"{count + 1:04d}"


# ---------------------------------------------------------------------------
# Cross-case merge — used by parallel runners
# ---------------------------------------------------------------------------

def merge_facts_into_global(local_fact_files: list[str], global_facts_file: str) -> dict:
    """Aggregate per-case facts into a single global pool.

    Used after a parallel run, where each case wrote its facts into its own
    long_term_memory.json (in single-file mode) instead of contending for a
    shared global file. This function reproduces the same dedup-and-reinforce
    semantics that _promote_fact would have applied if cases ran sequentially.

    Args:
      local_fact_files: paths to each case's long_term_memory.json
      global_facts_file: target path for the merged pool

    Returns:
      {"facts": [...], "stats": {"sources": N, "raw": M, "merged": K}}
    """
    pool: list[dict] = []
    by_norm: dict[str, dict] = {}
    raw_count = 0
    sources = 0

    for path in local_fact_files:
        if not os.path.exists(path):
            continue
        sources += 1
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        for fact in data.get("facts", []) or []:
            raw_count += 1
            norm = _normalize_fact(fact.get("fact", ""))
            if not norm:
                continue
            existing = by_norm.get(norm)
            if existing is None:
                # Copy so we don't mutate the source case's record.
                merged = dict(fact)
                merged.setdefault("reinforced_by", [merged.get("source_task_id", "?")])
                merged["reinforce_count"] = 0
                merged["confidence"] = FACT_INITIAL_CONFIDENCE
                by_norm[norm] = merged
                pool.append(merged)
            else:
                existing["reinforce_count"] = existing.get("reinforce_count", 0) + 1
                existing["confidence"] = min(
                    existing.get("confidence", 0) + FACT_REINFORCE_DELTA,
                    FACT_MAX_CONFIDENCE,
                )
                src = fact.get("source_task_id")
                if src and src not in existing.setdefault("reinforced_by", []):
                    existing["reinforced_by"].append(src)
                existing["last_reinforced_at"] = datetime.now().isoformat(timespec='seconds')

    payload = {
        "facts": pool,
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    }
    os.makedirs(os.path.dirname(global_facts_file), exist_ok=True)
    with open(global_facts_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "facts": pool,
        "stats": {"sources": sources, "raw": raw_count, "merged": len(pool)},
    }
