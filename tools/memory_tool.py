"""save_memory tool — the only tool that takes memory as a dependency.

Kept in its own file so fs/shell tools stay free of memory imports.
"""


def save_memory(memory, fact: str, category: str) -> str:
    if memory is None:
        return "Memory manager not initialized."

    wm = memory.get_working()
    if wm is None:
        return "No active task to associate memory with."

    wm.add_candidate_fact(fact, category)
    return (
        f"Recorded candidate fact: [{category}] {fact} "
        f"(will be saved if task passes verification)."
    )
