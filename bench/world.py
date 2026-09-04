"""Worlds, hidden properties, tasks, change events.

Phase 3 property library (abstract env):
  mechanism   sticky drawer   open() jams; needs pull_hard            failure-shaped
  physics     heavy object    pick() drops; needs pick_two_hand        failure-shaped
  location    hidden object   starts inside a drawer; fetch tasks need look_in  search-shaped
  shortcut    fast drawer     open() costs 2 instead of 5              SUCCESS-shaped: only a
                              successful open reveals it; pays off on free-choice tasks
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

DRAWERS = ("left", "middle", "right")
OBJECTS = ("blue_mug", "red_mug", "scissors", "spoon")
PROPERTY_TYPES = ("sticky", "heavy", "location", "fast")


@dataclass(frozen=True)
class HiddenProps:
    """The secrets. Never shown to the agent; only discoverable by acting."""

    sticky_drawer: str
    heavy_object: str
    hidden_object: str | None = None  # starts inside `hidden_in` instead of on the table
    hidden_in: str | None = None
    fast_drawer: str | None = None

    def is_sticky(self, drawer: str) -> bool:
        return drawer == self.sticky_drawer

    def is_heavy(self, obj: str) -> bool:
        return obj == self.heavy_object

    def is_fast(self, drawer: str) -> bool:
        return drawer == self.fast_drawer

    def location_of(self, obj: str) -> str | None:
        return self.hidden_in if obj == self.hidden_object else None


@dataclass(frozen=True)
class Task:
    """kind: 'put' = put <obj> in <drawer>; 'put_any' = put <obj> away (any drawer);
    'fetch' = bring <obj> to the table (it is somewhere in a drawer)."""

    obj: str
    drawer: str  # target for 'put'; ignored for 'put_any'/'fetch' (filled with "any"/"table")
    kind: str = "put"

    @property
    def text(self) -> str:
        if self.kind == "put_any":
            return f"put the {self.obj} away in any drawer"
        if self.kind == "fetch":
            return f"bring the {self.obj} to the table"
        return f"put the {self.obj} in the {self.drawer} drawer"


@dataclass
class World:
    world_id: int
    props: HiddenProps
    rng: np.random.Generator
    history: list[HiddenProps] = field(default_factory=list)  # for ground truth
    drawers: tuple[str, ...] = DRAWERS
    objects: tuple[str, ...] = OBJECTS
    property_types: tuple[str, ...] = PROPERTY_TYPES
    task_kinds: tuple[str, ...] = ("put", "put_any", "fetch")

    def sample_task(self, p_touch: float = 0.7) -> Task:
        """Sample a task kind, then (with prob p_touch) make it touch a hidden property."""
        kinds = [k for k in self.task_kinds if k != "fetch" or self.props.hidden_object]
        kind = str(self.rng.choice(kinds))
        touch = self.rng.random() < p_touch
        p = self.props
        if kind == "fetch":
            return Task(p.hidden_object, "table", "fetch")  # location is always the secret here
        # object: heavy or not
        objs = [o for o in self.objects if o != p.hidden_object]
        if touch and self.rng.random() < 0.5:
            obj = p.heavy_object
        else:
            obj = str(self.rng.choice([o for o in objs if o != p.heavy_object]))
        if kind == "put_any":
            return Task(obj, "any", "put_any")  # drawer choice is the robot's; fast/sticky drawers matter
        if touch and (obj != p.heavy_object or self.rng.random() < 0.5):
            drawer = p.sticky_drawer
        else:
            drawer = str(self.rng.choice([d for d in self.drawers if d != p.sticky_drawer]))
        return Task(obj, drawer, "put")

    def apply_change_event(self, ptype: str | None = None) -> str:
        """Flip one property type. Returns the type flipped."""
        p = self.props
        ptype = ptype or str(self.rng.choice([t for t in self.property_types if t != "location" or p.hidden_object]))
        self.history.append(p)
        if ptype == "sticky":
            self.props = replace(p, sticky_drawer=str(self.rng.choice([d for d in self.drawers if d != p.sticky_drawer])))
        elif ptype == "heavy":
            self.props = replace(p, heavy_object=str(self.rng.choice([o for o in self.objects if o not in (p.heavy_object, p.hidden_object)])))
        elif ptype == "location":
            self.props = replace(p, hidden_in=str(self.rng.choice([d for d in self.drawers if d != p.hidden_in])))
        elif ptype == "fast":
            self.props = replace(p, fast_drawer=str(self.rng.choice([d for d in self.drawers if d != p.fast_drawer])))
        return ptype


def make_world(world_id: int, seed: int, drawers: tuple[str, ...] = DRAWERS,
               objects: tuple[str, ...] = OBJECTS, property_types: tuple[str, ...] = PROPERTY_TYPES,
               task_kinds: tuple[str, ...] = ("put", "put_any", "fetch")) -> World:
    rng = np.random.default_rng([seed, world_id])
    sticky = str(rng.choice(drawers))
    heavy = str(rng.choice(objects))
    hidden = hidden_in = fast = None
    if "location" in property_types:
        hidden = str(rng.choice([o for o in objects if o != heavy]))
        hidden_in = str(rng.choice(drawers))
    if "fast" in property_types:
        fast = str(rng.choice([d for d in drawers if d != sticky]))
    props = HiddenProps(sticky_drawer=sticky, heavy_object=heavy, hidden_object=hidden, hidden_in=hidden_in, fast_drawer=fast)
    return World(world_id=world_id, props=props, rng=rng, drawers=drawers, objects=objects,
                 property_types=property_types, task_kinds=task_kinds)
