"""Episode-understanding, structured-log edition (architecture doc §"The bridge").

A deployment rarely hands the memory typed skill events. It hands *stage logs*:
a task with named sub-goals, which of them passed, and when. This module turns
stage logs into memlayer facts, with the same evidence semantics as the benchmark:

  a stage that repeatedly fails for the same entity  -> a "trouble" fact on that
                                                        entity (like sticky/heavy)
  a stage that passes with unusual cost              -> a "slow" fact (like a
                                                        sticky drawer's cost)
  a stage that passes quickly                        -> "easy" fact (like fast)
  a success after a run of failures                  -> contradiction/revision

`StageAttempt` is the generic record; adapters (memlayer.adapters.*) produce it
from a harness's own format. Facts land in the same SQLite store as everything
else, so recall()/recall_text()/explain() work unchanged, and the per-fact-type
RevisionPolicy applies ("trouble" facts revise by contradiction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

from .core import MemoryLayer


@dataclass
class StageAttempt:
    name: str            # harness stage name, e.g. "02_Place_Tomato_Basket"
    verb: str            # place | open | close | pour | pick | ...
    entities: list[str]  # canonical entity keys, e.g. ["object:tomato_sauce", "place:basket"]
    passed: bool
    steps: int | None = None       # steps spent on this stage (None if unknown)
    attempted: bool = True         # False if an earlier stage failed and this one was never reached


@dataclass
class StageEpisode:
    episode_idx: int
    task: str
    stages: list[StageAttempt]
    success: bool
    total_steps: int | None = None
    meta: dict = field(default_factory=dict)


_CAMEL = re.compile(r"[A-Z][a-z]+|[0-9]+")


def parse_stage_name(name: str, entity_hints: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """'02_Place_Tomato_Basket' -> ('place', ['object:tomato', 'place:basket']).
    entity_hints maps lowercase words to canonical keys ("tomato" -> "object:tomato_sauce")."""
    words = [w.lower() for w in name.replace("_", " ").split() if not w.isdigit()]
    if not words:
        return "unknown", []
    verb, rest = words[0], words[1:]
    rest = [w for w in rest if w not in ("again", "final", "the", "into", "in", "to", "on")]
    hints = entity_hints or {}
    ents: list[str] = []
    i = 0
    while i < len(rest):
        # greedy 2-word match first ("top drawer", "tomato sauce"), then 1-word
        two = " ".join(rest[i:i + 2])
        if two in hints:
            ents.append(hints[two]); i += 2; continue
        if rest[i] in hints:
            ents.append(hints[rest[i]]); i += 1; continue
        if i + 1 < len(rest) and rest[i + 1] in ("drawer", "cabinet", "basket", "bowl", "frypan", "microwave", "drainer", "mug", "cabinet2"):
            ents.append(f"place:{rest[i]}_{rest[i + 1]}"); i += 2; continue
        ents.append(f"object:{rest[i]}"); i += 1
    return verb, ents


class StageIngester:
    """Consolidates StageEpisodes into facts on a MemoryLayer."""

    def __init__(self, mem: MemoryLayer, window: int = 6, trouble_rate: float = 0.5, min_attempts: int = 3,
                 slow_ratio: float = 1.5, fast_ratio: float = 0.6) -> None:
        self.mem = mem
        self.window, self.trouble_rate, self.min_attempts = window, trouble_rate, min_attempts
        self.slow_ratio, self.fast_ratio = slow_ratio, fast_ratio
        self._hist: dict[tuple[str, str], list[bool]] = {}   # (verb, entity) -> recent pass/fail
        self._costs: dict[tuple[str, str], list[int]] = {}   # (verb, entity) -> step costs of passes

    def observe(self, ep: StageEpisode) -> None:
        """Deployment quirks are *rates*, not streaks (a 70%-failing grasp still succeeds
        sometimes), so 'trouble' is a windowed failure rate; a window that swings back
        below the threshold is the contradiction that flips the fact."""
        mem, t = self.mem, ep.episode_idx
        mem.t = t
        for st in ep.stages:
            if not st.attempted:
                continue
            for ent in st.entities:
                key = (st.verb, ent)
                attr = f"trouble:{st.verb}"
                h = self._hist.setdefault(key, [])
                h.append(st.passed)
                del h[:-self.window]
                if len(h) >= self.min_attempts:
                    rate = 1 - sum(h) / len(h)
                    trouble = rate >= self.trouble_rate
                    f = mem._fact(ent, attr)
                    if f is None or (f.value == "True") != trouble:
                        mem._write(ent, attr, str(trouble), True, t)   # new fact, or a flip
                    elif st.passed == (not trouble):
                        mem._write(ent, attr, str(trouble), True, t)   # confirming observation
                if st.passed and st.steps is not None:
                    base = self._costs.setdefault(key, [])
                    if len(base) >= 3:
                        m = median(base)
                        if st.steps > self.slow_ratio * m:
                            mem._write(ent, f"slow:{st.verb}", "True", True, t)
                        elif st.steps < self.fast_ratio * m:
                            mem._write(ent, f"easy:{st.verb}", "True", True, t)
                        elif mem._fact(ent, f"slow:{st.verb}") or mem._fact(ent, f"easy:{st.verb}"):
                            mem._write(ent, f"slow:{st.verb}", "False", True, t)
                            mem._write(ent, f"easy:{st.verb}", "False", True, t)
                    base.append(st.steps)
                    del base[:-20]
        mem.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?)",
                       (t, ep.task, "stage", "", "", int(ep.success), ep.total_steps or 0,
                        __import__("json").dumps([st.__dict__ for st in ep.stages]), ep.task))
        mem.db.commit()

    def recall_text(self) -> str:
        lines = []
        for f in self.mem.facts():
            age = self.mem.t - f.last_confirmed
            kind, attr = f.attribute.split(":", 1) if ":" in f.attribute else (f.attribute, "")
            ent = f.entity.split(":", 1)[1].replace("_", " ")
            if kind == "trouble" and f.value == "True":
                lines.append(f"Trouble with '{attr}' on the {ent}: failed {f.evidence}+ times (last {age} episode(s) ago).")
            elif kind == "slow" and f.value == "True":
                lines.append(f"'{attr}' on the {ent} is unusually slow (last seen {age} episode(s) ago).")
            elif kind == "easy" and f.value == "True":
                lines.append(f"'{attr}' on the {ent} is quick (last seen {age} episode(s) ago).")
        return "\n".join(lines)
