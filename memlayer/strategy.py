"""L5 — strategy memory: learn *how to run a task* from per-stage outcomes across episodes.

X5 (research log Day 13c) showed the pattern: a policy has several ways it can be driven
(full-task prompt vs. trained primitives, gentle vs. firm grasp, ...). Which one a stage
needs is not known up front; it is learned from failures. Two rules fell out of X5:

  1. Evidence is per stage and per strategy: facts are `trouble:<verb>@<strategy>` on the
     manipulated entity, so switching strategy never erases the evidence that motivated it.
  2. The *action* on the evidence has a granularity. `scope="episode"` (default): once any
     stage is a trouble fact under the default strategy, the whole episode runs on the
     fallback. `scope="stage"` switches only that stage — which lost to no memory at all in
     X5, because primitives trained from primitive end-states do not start from full-prompt
     end-states. Keep it as the ablation, not the default.

    sm = StrategyMemory(MemoryLayer(path), default="fixed", fallback="primitive")
    plan = sm.plan(["01_Place_Cookies_Basket", "02_Place_Tomato_Basket"])   # {stage: strategy}
    ... run the episode, then
    sm.record(episode_idx, stage_outcomes=[(name, passed, steps), ...], plan=plan, task=prompt)
"""
from __future__ import annotations

from dataclasses import dataclass

from .adapters.robomemarena import ENTITY_HINTS
from .core import MemoryLayer
from .ingest import StageAttempt, StageEpisode, StageIngester, parse_stage_name


@dataclass
class StageOutcome:
    name: str
    passed: bool
    steps: int | None = None


class StrategyMemory:
    def __init__(self, mem: MemoryLayer, default: str = "fixed", fallback: str = "primitive",
                 scope: str = "episode", entity_hints: dict[str, str] | None = None, **ingest_kw) -> None:
        assert scope in ("episode", "stage"), scope
        self.mem, self.default, self.fallback, self.scope = mem, default, fallback, scope
        self.hints = entity_hints or ENTITY_HINTS
        self.ingest = StageIngester(mem, **ingest_kw)

    # ---- decide -------------------------------------------------------------------
    def needs_fallback(self, stage: str) -> bool:
        """True if the default strategy is a known trouble fact for this stage's manipulated object."""
        verb, ents = parse_stage_name(stage, self.hints)
        objs = [e for e in ents if e.startswith("object:")] or ents
        for ent in objs:
            f = self.mem._fact(ent, f"trouble:{verb}@{self.default}")
            if f is not None and f.value == "True":
                return True
        return False

    def plan(self, stages: list[str]) -> dict[str, str]:
        plan = {s: (self.fallback if self.needs_fallback(s) else self.default) for s in stages}
        if self.scope == "episode" and self.fallback in plan.values():
            plan = {s: self.fallback for s in stages}
        return plan

    # ---- learn --------------------------------------------------------------------
    def record(self, episode_idx: int, stage_outcomes: list[StageOutcome | tuple], plan: dict[str, str],
               task: str = "", total_steps: int | None = None, sequential: bool = True) -> None:
        stages, blocked = [], False
        for o in stage_outcomes:
            o = o if isinstance(o, StageOutcome) else StageOutcome(*o)
            verb, ents = parse_stage_name(o.name, self.hints)
            stages.append(StageAttempt(name=o.name, verb=f"{verb}@{plan.get(o.name, self.default)}", entities=ents,
                                       passed=bool(o.passed), steps=o.steps, attempted=not blocked))
            if sequential and not o.passed:
                blocked = True
        self.ingest.observe(StageEpisode(episode_idx=episode_idx, task=task, stages=stages,
                                         success=all(s.passed for s in stages), total_steps=total_steps))

    def explain(self, stages: list[str]) -> str:
        lines = []
        for s in stages:
            verb, ents = parse_stage_name(s, self.hints)
            objs = [e for e in ents if e.startswith("object:")] or ents
            for ent in objs:
                for strat in (self.default, self.fallback):
                    f = self.mem._fact(ent, f"trouble:{verb}@{strat}")
                    if f is not None:
                        lines.append(f"{s}: '{verb}' under {strat} -> trouble={f.value} (evidence {f.evidence}, "
                                     f"last confirmed ep {f.last_confirmed})")
        return "\n".join(lines) or "no strategy evidence yet"
