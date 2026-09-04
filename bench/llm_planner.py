"""LLM-planner variant: memory context is TEXT, and a language model turns
(task, skills, context) into a skill plan — replanning after failures.

This is the bridge from scripted planners to VLAs: a real policy will consume the
memory layer as text in its prompt, exactly as here. It also makes "does structure
beat raw retrieval?" a fair question: last-k and retrieval hand the planner raw
episode logs; the consolidated store hands it fact sentences. Same planner, same
budget — only the memory's *representation* differs.

Backends: `anthropic` (needs ANTHROPIC_API_KEY; Haiku by default) or `mock`, a
rule-based reader used to test the plumbing and as a deterministic reference.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .env import EpisodeLog
from .memory import ConsolidatedKB, LastK, Memory, NoMemory, Retrieval
from .world import Task

SKILLS = {
    "open": "open(drawer): gentle pull. Fails (jam) if the drawer sticks.",
    "pull_hard": "pull_hard(drawer): strong pull, ~2x the cost of open. Opens sticky drawers.",
    "pick": "pick(object): light grip. Fails (drop) if the object is heavy.",
    "pick_two_hand": "pick_two_hand(object): firm grip, ~2x the cost of pick. Holds heavy objects.",
    "place": "place(object, drawer): put the held object into the open drawer. Ends the task.",
}


# ---- memory -> text ------------------------------------------------------------
def recall_text(memory: Memory, task: Task) -> str:
    """Render what a memory knows as text for the planner. Raw logs for the
    log-based memories, fact sentences for the consolidated store."""
    if isinstance(memory, NoMemory):
        return ""
    if isinstance(memory, LastK):
        return "\n".join(log.text for log in memory.logs)
    if isinstance(memory, Retrieval):
        q = set(task.text.split())
        scored = sorted(memory.logs, key=lambda l: (len(q & set(l.task.text.split())), l.episode_idx), reverse=True)
        return "\n".join(log.text for log in scored[: memory.top_k])
    if isinstance(memory, ConsolidatedKB):
        lines = []
        for d, f in memory.drawers.items():
            if f.evidence:
                age = memory.t - f.last_confirmed
                lines.append(f"The {d} drawer {'sticks' if f.value else 'opens normally'} "
                             f"(evidence: {f.evidence} observation(s), last confirmed {age} episode(s) ago).")
        for o, f in memory.objects.items():
            if f.evidence:
                age = memory.t - f.last_confirmed
                lines.append(f"The {o} {'is heavy' if f.value else 'is light'} "
                             f"(evidence: {f.evidence} observation(s), last confirmed {age} episode(s) ago).")
        return "\n".join(lines)
    raise TypeError(type(memory))


# ---- backends ------------------------------------------------------------------
SYSTEM = (
    "You control a robot arm with a small set of skills. Plan the cheapest skill sequence "
    "that completes the task within the step budget. Use robust skills only when the memory "
    "context gives you reason to. Memory can be stale: facts last confirmed long ago may have "
    "changed. Reply with ONLY a JSON list of steps, e.g. "
    '[{"skill":"open","target":"top"},{"skill":"pick","target":"butter"},{"skill":"place","target":"top"}].'
)


def build_prompt(task: Task, context: str, history: list[str]) -> str:
    skills = "\n".join(f"- {v}" for v in SKILLS.values())
    ctx = context.strip() or "(no memory)"
    hist = "\n".join(history) or "(none yet)"
    return (f"Task: {task.text}\nObject: {task.obj}. Drawer: {task.drawer}.\n\nSkills:\n{skills}\n\n"
            f"Memory context:\n{ctx}\n\nWhat happened so far in THIS episode:\n{hist}\n\n"
            "Plan the remaining steps.")


class MockBackend:
    """Deterministic reader: robust skill iff the context mentions a failure/fact for
    this drawer/object that is not older than `stale_after` episodes (when stated)."""

    name = "mock"

    def __init__(self, stale_after: int = 8) -> None:
        self.stale_after = stale_after

    def _fresh(self, sentence: str) -> bool:
        m = re.search(r"last confirmed (\d+) episode", sentence)
        return True if not m else int(m[1]) < self.stale_after

    def plan(self, task: Task, context: str, history: list[str]) -> list[dict]:
        sticky = heavy = False
        for line in context.splitlines():
            if task.drawer in line and ("jam" in line or "sticks" in line) and self._fresh(line):
                sticky = True
            if task.obj in line and ("drop" in line or "is heavy" in line) and self._fresh(line):
                heavy = True
        done = " ".join(history)
        steps = []
        if "drawer open" not in done:
            if "jam" in done or sticky:
                steps.append({"skill": "pull_hard", "target": task.drawer})
            else:
                steps.append({"skill": "open", "target": task.drawer})
        if "holding" not in done:
            if "drop" in done or heavy:
                steps.append({"skill": "pick_two_hand", "target": task.obj})
            else:
                steps.append({"skill": "pick", "target": task.obj})
        steps.append({"skill": "place", "target": task.drawer})
        return steps


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.calls = 0
        self.tokens = 0

    def plan(self, task: Task, context: str, history: list[str]) -> list[dict]:
        msg = self.client.messages.create(
            model=self.model, max_tokens=300, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(task, context, history)}],
        )
        self.calls += 1
        self.tokens += msg.usage.input_tokens + msg.usage.output_tokens
        text = msg.content[0].text
        m = re.search(r"\[.*\]", text, re.S)
        try:
            plan = json.loads(m.group(0) if m else text)
        except json.JSONDecodeError:
            plan = []
        return [s for s in plan if isinstance(s, dict) and s.get("skill") in SKILLS]


def make_backend(name: str):
    return AnthropicBackend() if name == "anthropic" else MockBackend()


# ---- execution ------------------------------------------------------------------
@dataclass
class LLMRunStats:
    calls: int = 0


def run_llm_planner(env, task: Task, context: str, backend, max_calls: int = 4) -> LLMRunStats:
    """Execute backend plans against a SkillEnv/SimSkillEnv, replanning after any
    failed skill (jam/drop) with the failure appended to the in-episode history."""
    history: list[str] = []
    stats = LLMRunStats()
    drawer_open = False
    holding = False
    while stats.calls < max_calls and not env.done:
        plan = backend.plan(task, context, history)
        stats.calls += 1
        if not plan:
            break
        failed = False
        for step in plan:
            if env.done:
                break
            skill, target = step["skill"], step.get("target", "")
            if skill == "place":
                ev = env.place(task.obj, target or task.drawer)
            else:
                ev = getattr(env, skill)(target)
            history.append(f"{ev.skill}({ev.target}) -> {ev.outcome}")
            if ev.outcome == "ok" and ev.skill in ("open", "pull_hard"):
                drawer_open = True
                history[-1] += " (drawer open)"
            if ev.outcome == "ok" and ev.skill in ("pick", "pick_two_hand", "pick_firm"):
                holding = True
                history[-1] += " (holding object)"
            if ev.outcome not in ("ok",):
                failed = True
                break
        if not failed:
            break
    return stats
