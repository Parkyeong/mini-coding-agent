"""
Two-layer memory:

- Working memory: created per task, discarded after end_task; holds candidate facts.

- MemoryManager (long-term memory): Only candidate facts that passed the tasks
  are stored here, with confidence scores.
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
# Working Memory
# ---------------------------------------------------------------------------

class WorkingMemory:
    def __init__(self, task_id: str, user_prompt: str):
        self.task_id = task_id
        self.user_prompt = user_prompt
        self.plan: list[str] = []
        self.observations: list[dict] = []     # [{kind, content, timestamp}]
        self.candidate_facts: list[dict] = []  # [{fact, category}]
        self.files_changed: set[str] = set()

    def set_plan(self, plan: list[str]) -> None:
        self.plan = list(plan)

    def add_observation(self, kind: str, content: str) -> None:
        if len(self.observations) >= MAX_WORKING_OBSERVATIONS:
            self.observations.pop(0)

        self.observations.append({
            "kind": kind,
            "content": content[:WORKING_OBSERVATION_MAX_CHARS],
            "timestamp": datetime.now().isoformat(timespec='seconds'),
        })

    def add_candidate_fact(self, fact: str, category: str) -> None:
        norm = _normalize_fact(fact)

        if any(_normalize_fact(f["fact"]) == norm for f in self.candidate_facts):
            return

        self.candidate_facts.append({"fact": fact, "category": category})

    def add_file_changed(self, path: str) -> None:
        self.files_changed.add(path)

    def snapshot_for_coder(self) -> str:
        """Format the current working memory into a string block for coder input."""
        if not self.observations and not self.plan:
            return "No observations or plan yet."

        parts = [f"[WorkingMemory] task_id = {self.task_id} "]
        if self.plan:
            parts.append("Current Plan:")
            for i, step in enumerate(self.plan, 1):
                parts.append(f" - {i}. {step}")

        if self.observations:
            parts.append("Recent observations from earlier steps:")
            for obs in self.observations[-10:]:
                parts.append(f" - [{obs['kind']}] {obs['content']}")

        if self.files_changed:
            parts.append(f"Files changed so far: {sorted(self.files_changed)}")

        return "\n".join(parts)


def _normalize_fact(fact: str) -> str:
    return " ".join(fact.strip().lower().split())


# ---------------------------------------------------------------------------
# Long-term Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    def __init__(self, memory_file: Optional[str] = None):
        if memory_file is None:
            memory_dir = os.path.join(config.WORKSHOP, config.PROJECT_NAME)
            os.makedirs(memory_dir, exist_ok=True)
            memory_file = os.path.join(memory_dir, "memory.json")
        self.memory_file = memory_file
        self.data = self._load()
        self._working: Optional[WorkingMemory] = None

    def _load(self) -> dict:
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._default_data()

    def _save(self) -> None:
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

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

    # project context
    def update_project_context(self, **kwargs) -> None:
        """Update project info, e.g. language='python', test_command='pytest'."""
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
                 attempts: int, summary: str) -> None:
        wm = self._working
        files_changed = sorted(wm.files_changed) if wm else []

        # Current task index (1-based): the new task we're about to record.
        # task_history hasn't been appended yet, so this is len + 1.
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
        }

        self.data["task_history"].append(record)
        self._trim_task_history()

        self._evict_facts_if_needed(current_task_idx)

        self._save()

        # Drop working memory
        self._working = None

    def _promote_fact(self, fact: str, category: str,
                      source_task_id: str, current_task_idx: int) -> None:
        """Promote a candidate fact to long-term memory.

        If the fact already exists, reinforce it (confidence += delta, count += 1).
        Otherwise insert as a new fact starting at confidence=0, count=0.
        """
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
            "created_at_task": current_task_idx,         # used by grace period
            "source_task_id": source_task_id,            # which task first created it
            "created_at": self._now(),
            "last_reinforced_at": self._now(),
            "reinforced_by": [source_task_id],           # task ids that reinforced it
        })

    def _evict_facts_if_needed(self, current_task_idx: int) -> None:
        """
        Evict facts when long-term exceeds MAX_MEMORY_FACTS.

        Policy (two-tier sort):
          1. Primary key: in_grace_period? (True sorts LAST = protected)
             Facts whose age < FACT_GRACE_PERIOD_TASKS are deprioritized
             from eviction unless we have no other choice.
          2. Secondary key: score = confidence × reinforce_count (lowest first).
             Within the same protection tier, the lowest score gets evicted.
          3. Tertiary key: created_at_task (oldest first).
             Same score → older fact had more chances, evict it first.

        This is NOT a hard exemption — grace period only LOWERS eviction
        priority. If memory pressure forces it (e.g. all facts are new),
        the lowest-score new fact still gets evicted instead of crashing.
        """
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
        return f"{count + 1:04d}"
