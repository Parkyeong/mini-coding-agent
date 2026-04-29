import os

# ---------------------------------------------------------------------------
# LLM (OpenRouter)
# ---------------------------------------------------------------------------
# All comparison-experiment knobs live here so a single edit changes the run.
MODEL = "openai/gpt-4o-mini"          # OpenRouter model id, e.g. "anthropic/claude-sonnet-4"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
# WORKSHOP is the project's Execution/ folder, sibling to config.py. Resolving
# it via __file__ makes the path work regardless of cwd or where the project
# is moved on disk.
WORKSHOP = os.path.abspath(os.path.join(os.path.dirname(__file__), "Execution"))
PROJECT_NAME = "mini coding agent"
WORKSPACE = f"{WORKSHOP}/{PROJECT_NAME}"
MAX_STEPS = 8
COMMAND_TIMEOUT = 20


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
MAX_REPLANS = 2
MAX_RETRIES_PER_STEPS = 2

MAX_CONTEXT_MESSAGES = 20
MAX_MEMORY_TASKS = 40
MAX_MEMORY_FACTS = 40

ENABLE_METRICS = True


def set_project(name: str):
    """Override PROJECT_NAME / WORKSPACE at startup (used by main.py CLI)."""
    global PROJECT_NAME, WORKSPACE
    PROJECT_NAME = name
    WORKSPACE = f"{WORKSHOP}/{PROJECT_NAME}"


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
# fact pool in practice. We add a tiny LLM judge to catch those before they
# pile up. Cheaper model than the main agent — judgment is binary, not creative.
ENABLE_LLM_DEDUP = True
DEDUP_MODEL = "openai/gpt-4.1-mini"

# Summarizer role: after a passed case, distills 1-2 project-level lessons
# from the full task trace (plan + tool actions + verifier results + final
# code). Replaces coder's save_memory tool — facts now come from a whole-
# system review, not from coder's mid-execution side-thoughts. Same model as
# the main agent (gpt-4o-mini) so summary quality matches the tasks the agent
# actually solves; cost is tiny (~$0.10 per 257 cases).
SUMMARIZER_MODEL = "openai/gpt-4o-mini"

# Note: global facts files are NOT a config constant anymore — each experiment
# owns its own facts file under Execution/<exp_name>/mbpp_global_facts.json,
# and the runner passes that path to MemoryManager(global_facts_file=...).
