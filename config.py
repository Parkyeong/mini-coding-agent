import os

PROVIDER = "openai"   # gemini / local / openai / claude
MODEL = "gpt-4o-mini"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = ""



# Tool definitions
WORKSHOP = "/home/obob/Research Project/mini coding agent/Execution"
PROJECT_NAME = "mini coding agent"
WORKSPACE = f"{WORKSHOP}/{PROJECT_NAME}"
MAX_STEPS = 8
COMMAND_TIMEOUT = 20

MAX_REPLANS = 2
MAX_RETRIES_PER_STEPS = 2

MAX_CONTEXT_MESSAGES = 20
MAX_MEMORY_TASKS = 40
MAX_MEMORY_FACTS =40

ENABLE_METRICS = True

def set_project(name: str):
    """Called from main.py to override PROJECT_NAME and WORKSPACE at startup"""
    global PROJECT_NAME, WORKSPACE
    PROJECT_NAME = name
    WORKSPACE = f"{WORKSHOP}/{PROJECT_NAME}"


# Verifier settings
VERIFIER_RUN_TESTS = True
VERIFIER_TEST_TIMEOUT_DEFAULT = 90
VERIFIER_OUTPUT_MAX_CHARS = 4000


# Memory settings
MAX_WORKING_OBSERVATIONS = 20
WORKING_OBSERVATION_MAX_CHARS = 500
FACT_INITIAL_CONFIDENCE = 0.0
FACT_REINFORCE_DELTA = 0.2
FACT_MAX_CONFIDENCE = 1.0
FACT_GRACE_PERIOD_TASKS = 5

# Global facts files (used by benchmark runners for cross-instance fact sharing).
# When MemoryManager is initialized with global_facts_file=this path, facts are
# read/written here instead of the per-instance memory.json. task_history and
# project_context still go to per-instance memory.json so each instance can be
# inspected independently.
MBPP_GLOBAL_FACTS_FILE = f"{WORKSHOP}/mbpp_global_facts.json"
SWEBENCH_GLOBAL_FACTS_FILE = f"{WORKSHOP}/swebench_global_facts.json"