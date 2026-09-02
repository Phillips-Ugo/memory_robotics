"""Abstract skill-level simulator.

Each skill costs steps and returns an outcome. Hidden properties decide whether the
cheap skill works or the robust (more expensive) one is needed. This is the interface
robosuite skills replace later — the memory module never sees anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .world import Task, World

# step costs: cheap skill / robust skill / wasted steps on failure
COST = {
    "open": 5,
    "pull_hard": 10,
    "pick": 5,
    "pick_two_hand": 9,
    "place": 5,
    "jam": 8,  # wasted when open() jams
    "drop": 6,  # wasted when pick() drops
}
STEP_BUDGET = 26  # optimal path = 15; a single jam+recovery (28) or drop+recovery (25) decides success (calibrated 2026-09-02, see log)


@dataclass
class SkillEvent:
    skill: str
    target: str
    outcome: str  # ok | jam | drop
    steps: int


@dataclass
class EpisodeLog:
    episode_idx: int
    task: Task
    events: list[SkillEvent] = field(default_factory=list)
    success: bool = False
    steps: int = 0
    stale_actions: int = 0  # robust skill used where the cheap one would have worked

    @property
    def text(self) -> str:
        ev = "; ".join(f"{e.skill}({e.target})->{e.outcome}" for e in self.events)
        return f"[ep {self.episode_idx}] {self.task.text}: {ev} | {'success' if self.success else 'FAIL'} in {self.steps} steps"


class SkillEnv:
    def __init__(self, world: World, task: Task, episode_idx: int) -> None:
        self.world = world
        self.task = task
        self.log = EpisodeLog(episode_idx=episode_idx, task=task)
        self.drawer_open = False
        self.holding: str | None = None
        self.done = False

    def _record(self, skill: str, target: str, outcome: str, steps: int) -> SkillEvent:
        ev = SkillEvent(skill, target, outcome, steps)
        self.log.events.append(ev)
        self.log.steps += steps
        if self.log.steps > STEP_BUDGET:
            self.done = True
        return ev

    # ---- skills ------------------------------------------------------------
    def open(self, drawer: str) -> SkillEvent:
        if self.world.props.is_sticky(drawer):
            return self._record("open", drawer, "jam", COST["jam"])
        self.drawer_open = True
        return self._record("open", drawer, "ok", COST["open"])

    def pull_hard(self, drawer: str) -> SkillEvent:
        if not self.world.props.is_sticky(drawer):
            self.log.stale_actions += 1
        self.drawer_open = True
        return self._record("pull_hard", drawer, "ok", COST["pull_hard"])

    def pick(self, obj: str) -> SkillEvent:
        if self.world.props.is_heavy(obj):
            return self._record("pick", obj, "drop", COST["drop"])
        self.holding = obj
        return self._record("pick", obj, "ok", COST["pick"])

    def pick_two_hand(self, obj: str) -> SkillEvent:
        if not self.world.props.is_heavy(obj):
            self.log.stale_actions += 1
        self.holding = obj
        return self._record("pick_two_hand", obj, "ok", COST["pick_two_hand"])

    def place(self, obj: str, drawer: str) -> SkillEvent:
        ev = self._record("place", drawer, "ok", COST["place"])
        if self.holding == obj and self.drawer_open and drawer == self.task.drawer:
            self.log.success = self.log.steps <= STEP_BUDGET
            self.done = True
        return ev
