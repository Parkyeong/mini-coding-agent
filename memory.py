"""
Two-layer memory:

- Working memory: created per task, discarded after end_task. Holds the
  task-level event_log (every tool call/result during this task), the plan,
  and candidate facts. observations / files_changed are *derived* from the
  event_log, not stored directly.

- MemoryManager (long-term memory): Only candidate facts that passed the
  tasks are stored here, with confidence scores.
"""

import os
import json
from datetime import datetime
from typing import Optional

import config
from config import (
    WORKSHOP,
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
            for obs in observations[-MAX_WORKING_OBSERVATIONS:][-10:]:
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
    Two operating modes:

    1. **Single-file mode** (default): all data — project_context, task_history,
       and facts — live in the same memory.json. This is the P0 / single-project
       use case.

    2. **Dual-file mode**: pass `global_facts_file=...` to split storage:
         - `memory.json` (per-instance): project_context + task_history
         - `global_facts.json` (shared): facts only
       This is the benchmark mode (MBPP / SWE-bench), where each instance has
       its own memory.json for inspection ("how did this instance fail?"), but
       facts are pooled across instances so reinforcement actually happens.

    The two modes are transparent to callers: `self.data["facts"]` always
    returns the right set of facts; `_save()` writes to whichever file owns
    each piece of data.
    """

    def __init__(
        self,
        memory_file: Optional[str] = None,
        global_facts_file: Optional[str] = None,
    ):
        if memory_file is None:
            memory_dir = os.path.join(config.WORKSHOP, config.PROJECT_NAME)
            os.makedirs(memory_dir, exist_ok=True)
            memory_file = os.path.join(memory_dir, "memory.json")
        self.memory_file = memory_file
        self.global_facts_file = global_facts_file   # None = single-file mode

        self.data = self._load()
        self._working: Optional[WorkingMemory] = None

    def _load(self) -> dict:
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
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
            per_instance = {k: v for k, v in self.data.items() if k != "facts"}
            per_instance["facts"] = []
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(per_instance, f, indent=2, ensure_ascii=False)
        else:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _save_global_facts(self, facts: list) -> None:
        os.makedirs(os.path.dirname(self.global_facts_file), exist_ok=True)
        payload = {
            "facts": facts,
            "updated_at": self._now(),
        }
        with open(self.global_facts_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _default_data(self) -> dict:
        return {
            "project_context": {
                "project_name": config.PROJECT_NAME,
                "workspace": config.WORKSPACE,
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
            # Persist the per-task event stream so we can replay LLM decisions
            # offline without rerunning. Useful for debugging — without this the
            # event_log is discarded when working memory drops.
            "event_log": list(wm.event_log.events) if wm else [],
        }
        if error_history:
            record["error_history"] = error_history

        self.data["task_history"].append(record)
        self._trim_task_history()

        self._evict_facts_if_needed(current_task_idx)

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
