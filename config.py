import os

# ---------------------------------------------------------------------------
# LLM (OpenRouter)
# ---------------------------------------------------------------------------
# All comparison-experiment knobs live here so a single edit changes the run.
MODEL = "openai/gpt-4o-mini"          # global default, used by llm.chat() when no per-role override
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
# WORKSHOP is the project's Execution/ folder, sibling to config.py. Resolving
# it via __file__ makes the path work regardless of cwd or where the project
# is moved on disk.
WORKSHOP = os.path.abspath(os.path.join(os.path.dirname(__file__), "Execution"))
MAX_STEPS = 8                          # LLMNode default; overridden per-role via ROLE_CONFIGS
COMMAND_TIMEOUT = 20


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
MAX_REPLANS = 2
MAX_RETRIES_PER_STEPS = 2

MAX_MEMORY_TASKS = 40
MAX_MEMORY_FACTS = 40

ENABLE_METRICS = True


# ---------------------------------------------------------------------------
# Per-role hyperparameters
# ---------------------------------------------------------------------------
# Single source of truth for the per-role knobs each role gets at runtime.
# engine.build_llm_nodes() iterates this dict to construct LLMNodes; runners
# read ROLE_CONFIGS["dedup"] when constructing the dedup node.
#
# Schema per role:
#   model       — OpenRouter model id, or None for "no LLM" (verifier).
#                 Set to e.g. "anthropic/claude-sonnet-4" for a stronger role.
#   max_steps   — LLM rounds the role's loop runs. planner / summarizer /
#                 dedup are one-shot (1). coder is multi-turn (8) because it
#                 iterates with tools. None for non-LLM roles.
#   uses_tools  — whether the role's LLMNode gets the full tool_pool/ops.py
#                 toolkit. Only coder needs this today.
#   temperature — sampling temperature (0.0 = deterministic, higher = more
#                 random). Tuned per role: planner/coder low for stability,
#                 summarizer mid for natural phrasing, dedup zero for
#                 consistent binary judgment. None = use API default.
#   max_tokens  — cap on response length. None = use API default. Override
#                 when a role needs a hard cap (e.g. extended-thinking budget
#                 or expensive long-form roles).
#
# Adding a new role: add an entry here + a module under role_pool/ + wire it
# into the orchestrator. Brain (Track A) lives here too once added.
# Adding a new param: add a key to every role + thread it through llm.chat.
ROLE_CONFIGS: dict[str, dict] = {
    "planner": {
        "model": "openai/gpt-4o-mini",
        "max_steps": 1,
        "uses_tools": False,
        "temperature": 0.2,
        "max_tokens": None,
    },
    "coder": {
        "model": "openai/gpt-4o-mini",
        "max_steps": 8,
        "uses_tools": True,
        "temperature": 0.1,
        "max_tokens": None,
    },
    "verifier": {
        "model": None,                 # no LLM — pure pytest function
        "max_steps": None,
        "uses_tools": False,
        "temperature": None,
        "max_tokens": None,
    },
    "summarizer": {
        "model": "openai/gpt-4o-mini",
        "max_steps": 1,
        "uses_tools": False,
        "temperature": 0.4,
        "max_tokens": None,
    },
    "dedup": {
        "model": "openai/gpt-4o-mini",
        "max_steps": 1,
        "uses_tools": False,
        "temperature": 0.0,
        "max_tokens": None,
    },
    # Story task — gpt-4o-mini is locked here per the story-task spec
    # (the worker writing the actual prose must be 4o-mini for fairness across
    # baseline / Track A / Track B). Temperature is moderate to allow some
    # creative variability while staying coherent.
    "writer": {
        "model": "openai/gpt-4o-mini",
        "max_steps": 1,
        "uses_tools": False,
        "temperature": 0.7,
        "max_tokens": None,
    },
}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
VERIFIER_RUN_TESTS = True
VERIFIER_TEST_TIMEOUT_DEFAULT = 90
VERIFIER_OUTPUT_MAX_CHARS = 4000


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
MAX_WORKING_OBSERVATIONS = 30
WORKING_OBSERVATION_MAX_CHARS = 500
FACT_INITIAL_CONFIDENCE = 0.0
FACT_REINFORCE_DELTA = 0.2
FACT_MAX_CONFIDENCE = 1.0
FACT_GRACE_PERIOD_TASKS = 5

# LLM-based semantic dedup for global facts pool (used at merge time).
# String-normalize-only dedup misses near-duplicates with different wording
# ("uses pytest -q" vs "tests run quietly via pytest -q"), which dominate the
# fact pool in practice. The dedup model itself is configured in
# ROLE_CONFIGS["dedup"]["model"]; this flag is the on/off switch.
ENABLE_LLM_DEDUP = True

# Note: global facts files are NOT a config constant anymore — each experiment
# owns its own facts file under Execution/<exp_name>/mbpp_global_facts.json,
# and the runner passes that path to MemoryManager(global_facts_file=...).
