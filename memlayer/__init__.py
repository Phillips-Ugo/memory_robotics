"""memlayer — the memory layer for robots (v0.1 in progress).

    from memlayer import MemoryLayer
    mem = MemoryLayer("site.db")
    mem.observe(episode_log)          # after each task
    beliefs = mem.recall(task)        # before each task (structured)
    text = mem.recall_text(task)      # same, rendered for an LLM/VLA prompt
"""
from .core import MemoryLayer, RevisionPolicy

__all__ = ["MemoryLayer", "RevisionPolicy"]
