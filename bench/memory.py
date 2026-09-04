"""The memory API (identical to the planned Phase 4 library API) and the baselines.

    memory.observe(episode_log)                -> None      after each episode
    memory.recall(task, initial_obs) -> Beliefs            before each episode

Beliefs is what the planner consumes: for each drawer/object the task touches, does
the memory believe it needs the robust skill? Baselines differ only in how they turn
past logs into beliefs. Everything else in the benchmark is held fixed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .env import EpisodeLog
from .world import Task


@dataclass
class Beliefs:
    sticky_drawers: set[str] = field(default_factory=set)
    heavy_objects: set[str] = field(default_factory=set)
    object_in: dict[str, str] = field(default_factory=dict)  # obj -> drawer it was last found in
    object_not_in: dict[str, set[str]] = field(default_factory=dict)  # obj -> drawers seen empty since
    fast_drawers: set[str] = field(default_factory=set)  # drawers whose open was cheap (success-shaped)
    # optional: facts the memory wants the planner to re-test cheaply (revision)
    probe_drawers: set[str] = field(default_factory=set)
    probe_objects: set[str] = field(default_factory=set)


class Memory:
    name = "base"

    def surfaced(self, task: Task) -> list[int]:
        """Episode indices this memory would put in front of the planner for `task`
        (for retrieval precision/recall against generator ground truth)."""
        return []

    def observe(self, log: EpisodeLog) -> None:  # noqa: D401
        raise NotImplementedError

    def recall(self, task: Task, initial_obs: dict) -> Beliefs:
        raise NotImplementedError

    def bytes_stored(self) -> int:
        return 0


# --- baseline 1: no memory --------------------------------------------------------
class NoMemory(Memory):
    name = "none"

    def observe(self, log: EpisodeLog) -> None:
        pass

    def recall(self, task: Task, initial_obs: dict) -> Beliefs:
        return Beliefs()


# --- baseline 2: last-k episodes verbatim ------------------------------------------
class LastK(Memory):
    """Keeps the last k episode logs. Believes whatever failures appear in them."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.logs: list[EpisodeLog] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"last-{self.k}"

    def observe(self, log: EpisodeLog) -> None:
        self.logs.append(log)
        self.logs = self.logs[-self.k :]

    def recall(self, task: Task, initial_obs: dict) -> Beliefs:
        return _beliefs_from_logs(self.logs)

    def surfaced(self, task: Task) -> list[int]:
        return [l.episode_idx for l in self.logs]

    def bytes_stored(self) -> int:
        return sum(len(l.text) for l in self.logs)


# --- baseline 3: retrieval over all episodes ---------------------------------------
class Retrieval(Memory):
    """Stores every episode; retrieves the ones whose task mentions the same
    drawer/object (v0 stand-in for embedding similarity: token overlap on the task
    text). Never forgets, never revises — the RoboMME-Interference fix, in spirit."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.logs: list[EpisodeLog] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"retrieval-top{self.top_k}"

    def observe(self, log: EpisodeLog) -> None:
        self.logs.append(log)

    def _top(self, task: Task) -> list[EpisodeLog]:
        q = set(task.text.split())
        return sorted(self.logs, key=lambda l: (len(q & set(l.task.text.split())), l.episode_idx),
                      reverse=True)[: self.top_k]

    def recall(self, task: Task, initial_obs: dict) -> Beliefs:
        return _beliefs_from_logs(self._top(task))

    def surfaced(self, task: Task) -> list[int]:
        return [l.episode_idx for l in self._top(task)]

    def bytes_stored(self) -> int:
        return sum(len(l.text) for l in self.logs)


# --- baseline 4: LLM summary (stub) -------------------------------------------------
# Needs an LLM call; deferred. Left here so the baseline list matches the spec.


# --- ours (prototype): consolidated facts with evidence, freshness, and a probe rule --
@dataclass
class Fact:
    value: bool = False  # True = needs robust skill
    evidence: int = 0
    last_confirmed: int = -1  # episode index of last direct evidence


class ConsolidatedKB(Memory):
    """Consolidates logs into per-entity facts. Revision rule: a fact that has not
    been directly re-confirmed for `probe_after` episodes is re-tested with the cheap
    skill once (costs a little if still true, saves a lot if it changed)."""

    def __init__(self, probe_after: int = 8) -> None:
        self.probe_after = probe_after
        self.drawers: dict[str, Fact] = defaultdict(Fact)
        self.objects: dict[str, Fact] = defaultdict(Fact)
        self.fast: dict[str, Fact] = defaultdict(Fact)  # drawer -> opens fast
        self.where: dict[str, tuple[str, int]] = {}  # obj -> (drawer, episode last found)
        self.not_in: dict[str, dict[str, int]] = defaultdict(dict)  # obj -> {drawer: episode seen empty}
        self.t = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"consolidated-probe{self.probe_after}"

    def observe(self, log: EpisodeLog) -> None:
        self.t = log.episode_idx
        for e in log.events:
            if e.skill == "open":
                f = self.drawers[e.target]
                f.value, f.evidence, f.last_confirmed = (e.outcome == "jam"), f.evidence + 1, self.t
            elif e.skill == "pick" and e.outcome in ("ok", "drop"):
                f = self.objects[e.target]
                f.value, f.evidence, f.last_confirmed = (e.outcome == "drop"), f.evidence + 1, self.t
            elif e.skill == "look_in":
                obj = log.task.obj
                if e.outcome == "found":
                    self.where[obj] = (e.target, self.t)
                    self.not_in[obj].pop(e.target, None)
                else:
                    self.not_in[obj][e.target] = self.t
                    if self.where.get(obj, (None,))[0] == e.target:
                        del self.where[obj]  # it moved
            # pull_hard / pick_two_hand give no evidence either way (they always work)
            if e.skill == "open" and e.outcome == "ok":  # success-shaped: cheap open => fast drawer
                f = self.fast[e.target]
                f.value, f.evidence, f.last_confirmed = (e.steps <= 2), f.evidence + 1, self.t

    def recall(self, task: Task, initial_obs: dict) -> Beliefs:
        b = Beliefs()
        fd = self.drawers.get(task.drawer)
        if fd and fd.value:
            if self.t - fd.last_confirmed >= self.probe_after:
                b.probe_drawers.add(task.drawer)
            else:
                b.sticky_drawers.add(task.drawer)
        fo = self.objects.get(task.obj)
        if fo and fo.value:
            if self.t - fo.last_confirmed >= self.probe_after:
                b.probe_objects.add(task.obj)
            else:
                b.heavy_objects.add(task.obj)
        for d, f in self.fast.items():
            if f.value:
                b.fast_drawers.add(d)
        for d, f in self.drawers.items():  # all known sticky drawers matter for free-choice tasks
            if f.value and self.t - f.last_confirmed < self.probe_after:
                b.sticky_drawers.add(d)
        if task.obj in self.where:
            b.object_in[task.obj] = self.where[task.obj][0]
        found_t = self.where.get(task.obj, (None, -1))[1]
        b.object_not_in[task.obj] = {d for d, t in self.not_in.get(task.obj, {}).items() if t > found_t}
        return b

    def bytes_stored(self) -> int:
        return 40 * (len(self.drawers) + len(self.objects) + len(self.fast) + len(self.where))


def _beliefs_from_logs(logs: list[EpisodeLog]) -> Beliefs:
    """Raw-log readers: believe whatever the (chronological) logs show, last write wins."""
    b = Beliefs()
    for l in sorted(logs, key=lambda l: l.episode_idx):
        for e in l.events:
            if e.outcome == "jam":
                b.sticky_drawers.add(e.target)
            elif e.outcome == "drop":
                b.heavy_objects.add(e.target)
            elif e.skill == "look_in" and e.outcome == "found":
                b.object_in[l.task.obj] = e.target
                b.object_not_in.setdefault(l.task.obj, set()).discard(e.target)
            elif e.skill == "look_in" and e.outcome == "empty":
                b.object_not_in.setdefault(l.task.obj, set()).add(e.target)
                if b.object_in.get(l.task.obj) == e.target:
                    del b.object_in[l.task.obj]
            elif e.skill == "open" and e.outcome == "ok" and e.steps <= 2:
                b.fast_drawers.add(e.target)
    return b


BASELINES = {
    "none": lambda: NoMemory(),
    "last-5": lambda: LastK(5),
    "retrieval": lambda: Retrieval(5),
    "consolidated": lambda: ConsolidatedKB(8),
}
