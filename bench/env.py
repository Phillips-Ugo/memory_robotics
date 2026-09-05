"""Abstract skill-level simulator.

Each skill costs steps and returns an outcome. Hidden properties decide whether the
cheap skill works or the robust (more expensive) one is needed. This is the interface
robosuite skills replace later — the memory module never sees anything else.

Skills: open / pull_hard / pick / pick_two_hand / place / look_in.
Outcomes: ok | jam | drop | found | empty. The 'fast' drawer reveals itself only
through a cheaper successful open (the step count on the event).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .world import Task, World

# step costs: cheap skill / robust skill / wasted steps on failure
COST = {
    "open": 5,
    "open_fast": 2,  # the fast drawer's open (success-shaped secret)
    "pull_hard": 10,
    "pick": 5,
    "pick_two_hand": 9,
    "place": 5,
    "look_in": 4,  # peek into a drawer (+ close 2 if empty = 6 wasted per wrong drawer)
    "close": 2,
    "jam": 8,  # wasted when open() jams
    "drop": 6,  # wasted when pick() drops
}
# per-task-kind nominal (no-secret, no-belief) cost + slack; robust skills cost +4/+5,
# a drop recovery +10, a jam recovery +13, two wasted looks +12 -> slack 11 makes jams
# and full searches decisive, drops and single stale actions survivable (Day 6 rule)
NOMINAL = {"put": 15, "put_any": 15, "fetch": 19}  # fetch = open 5 + look 4 + pick 5 + place 5
SLACK = 11
STEP_BUDGET = 26  # kept for bench.sim.calibrate / legacy callers


def budget_for(task: Task) -> int:
    return NOMINAL[task.kind] + SLACK


@dataclass
class SkillEvent:
    skill: str
    target: str
    outcome: str  # ok | jam | drop | found | empty
    steps: int


@dataclass
class EpisodeLog:
    episode_idx: int
    task: Task
    events: list[SkillEvent] = field(default_factory=list)
    success: bool = False
    steps: int = 0
    stale_actions: int = 0  # robust skill used where the cheap one would have worked
    wasted_looks: int = 0  # look_in on a drawer that did not hold the object

    @property
    def text(self) -> str:
        ev = "; ".join(f"{e.skill}({e.target})->{e.outcome}[{e.steps}]" for e in self.events)
        return f"[ep {self.episode_idx}] {self.task.text}: {ev} | {'success' if self.success else 'FAIL'} in {self.steps} steps"


class SkillEnv:
    def __init__(self, world: World, task: Task, episode_idx: int, step_budget: int | None = None) -> None:
        self.world = world
        self.task = task
        self.step_budget = step_budget or budget_for(task)
        self.log = EpisodeLog(episode_idx=episode_idx, task=task)
        self.open_drawers: set[str] = set()
        self.holding: str | None = None
        self.done = False

    def _record(self, skill: str, target: str, outcome: str, steps: int) -> SkillEvent:
        ev = SkillEvent(skill, target, outcome, steps)
        self.log.events.append(ev)
        self.log.steps += steps
        if self.log.steps > self.step_budget:
            self.done = True
        return ev

    @property
    def drawer_open(self) -> bool:  # legacy: any drawer open
        return bool(self.open_drawers)

    # ---- skills ------------------------------------------------------------
    def open(self, drawer: str) -> SkillEvent:
        if self.world.props.is_sticky(drawer):
            return self._record("open", drawer, "jam", COST["jam"])
        self.open_drawers.add(drawer)
        return self._record("open", drawer, "ok", COST["open_fast"] if self.world.props.is_fast(drawer) else COST["open"])

    def pull_hard(self, drawer: str) -> SkillEvent:
        if not self.world.props.is_sticky(drawer):
            self.log.stale_actions += 1
        self.open_drawers.add(drawer)
        return self._record("pull_hard", drawer, "ok", COST["pull_hard"])

    def look_in(self, drawer: str) -> SkillEvent:
        """Peek into a drawer (opens it if needed at its open cost). found / empty."""
        if drawer not in self.open_drawers:
            ev = self.open(drawer)
            if ev.outcome != "ok":
                return ev
        here = self.world.props.location_of(self.task.obj) == drawer
        if not here:
            self.log.wasted_looks += 1
        return self._record("look_in", drawer, "found" if here else "empty", COST["look_in"])

    def close(self, drawer: str) -> SkillEvent:
        self.open_drawers.discard(drawer)
        return self._record("close", drawer, "ok", COST["close"])

    def pick(self, obj: str) -> SkillEvent:
        loc = self.world.props.location_of(obj)
        if loc is not None and loc not in self.open_drawers:
            return self._record("pick", obj, "not_here", COST["pick"])  # it's in a closed drawer
        if self.world.props.is_heavy(obj):
            return self._record("pick", obj, "drop", COST["drop"])
        self.holding = obj
        return self._record("pick", obj, "ok", COST["pick"])

    def pick_two_hand(self, obj: str) -> SkillEvent:
        loc = self.world.props.location_of(obj)
        if loc is not None and loc not in self.open_drawers:
            return self._record("pick_two_hand", obj, "not_here", COST["pick_two_hand"])
        if not self.world.props.is_heavy(obj):
            self.log.stale_actions += 1
        self.holding = obj
        return self._record("pick_two_hand", obj, "ok", COST["pick_two_hand"])

    def place(self, obj: str, where: str) -> SkillEvent:
        """where: a drawer (put / put_any) or 'table' (fetch)."""
        ev = self._record("place", where, "ok", COST["place"])
        t = self.task
        ok = self.holding == obj and (
            (t.kind == "put" and where == t.drawer and where in self.open_drawers)
            or (t.kind == "put_any" and where in self.open_drawers)
            or (t.kind == "fetch" and where == "table")
        )
        if ok:
            self.log.success = self.log.steps <= self.step_budget
        self.done = True
        return ev
