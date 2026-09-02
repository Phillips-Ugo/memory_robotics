"""Worlds, hidden properties, tasks, change events."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

DRAWERS = ("left", "middle", "right")
OBJECTS = ("blue_mug", "red_mug", "scissors", "spoon")


@dataclass(frozen=True)
class HiddenProps:
    """The secrets. Never shown to the agent; only discoverable by acting."""

    sticky_drawer: str  # open() jams; needs pull_hard()
    heavy_object: str  # pick() drops; needs pick_two_hand()

    def is_sticky(self, drawer: str) -> bool:
        return drawer == self.sticky_drawer

    def is_heavy(self, obj: str) -> bool:
        return obj == self.heavy_object


@dataclass(frozen=True)
class Task:
    """'put <obj> in <drawer>'. Object starts on the table, drawer starts closed."""

    obj: str
    drawer: str

    @property
    def text(self) -> str:
        return f"put the {self.obj} in the {self.drawer} drawer"


@dataclass
class World:
    world_id: int
    props: HiddenProps
    rng: np.random.Generator
    history: list[HiddenProps] = field(default_factory=list)  # for ground truth

    def sample_task(self, p_touch: float = 0.7) -> Task:
        """With prob p_touch the task involves at least one hidden property."""
        if self.rng.random() < p_touch:
            # touch the sticky drawer, the heavy object, or both
            r = self.rng.random()
            if r < 1 / 3:
                return Task(self.props.heavy_object, self.props.sticky_drawer)
            if r < 2 / 3:
                obj = self.rng.choice([o for o in OBJECTS if o != self.props.heavy_object])
                return Task(str(obj), self.props.sticky_drawer)
            drawer = self.rng.choice([d for d in DRAWERS if d != self.props.sticky_drawer])
            return Task(self.props.heavy_object, str(drawer))
        obj = self.rng.choice([o for o in OBJECTS if o != self.props.heavy_object])
        drawer = self.rng.choice([d for d in DRAWERS if d != self.props.sticky_drawer])
        return Task(str(obj), str(drawer))

    def apply_change_event(self) -> None:
        """The sticky drawer gets fixed and a *different* drawer starts sticking.

        Tests both directions: learning a new fact (any memory can) and unlearning a
        stale one (only revision can).
        """
        self.history.append(self.props)
        new_sticky = str(self.rng.choice([d for d in DRAWERS if d != self.props.sticky_drawer]))
        self.props = replace(self.props, sticky_drawer=new_sticky)


def make_world(world_id: int, seed: int) -> World:
    rng = np.random.default_rng([seed, world_id])
    props = HiddenProps(
        sticky_drawer=str(rng.choice(DRAWERS)),
        heavy_object=str(rng.choice(OBJECTS)),
    )
    return World(world_id=world_id, props=props, rng=rng)
