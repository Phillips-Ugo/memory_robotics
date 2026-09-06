"""Cross-episode complexity / headroom score for a world (Phase 3 metric).

headroom(world) = success(oracle memory: ground-truth beliefs every episode)
                - success(no memory)
It is the most any memory could add in that world under the protocol; a world with
zero headroom tests nothing. Reported per world and for the protocol, and used to
normalise memory scores: normalised = (mem - none) / headroom.

    uv run python -m bench.headroom
"""
from __future__ import annotations

import numpy as np

from .env import SkillEnv
from .memory import BASELINES, Beliefs
from .planner import run_planner
from .world import make_world


class Oracle:
    """Ground-truth beliefs (reads the world's current secrets): the upper bound."""
    name = "oracle"

    def __init__(self):
        self.world = None

    def observe(self, log):
        pass

    def recall(self, task, initial_obs):
        p = self.world.props
        b = Beliefs(sticky_drawers={p.sticky_drawer}, heavy_objects={p.heavy_object})
        if p.hidden_object:
            b.object_in[p.hidden_object] = p.hidden_in
        if p.fast_drawer:
            b.fast_drawers.add(p.fast_drawer)
        b.preferred_drawer = p.preferred_drawer
        return b

    def bytes_stored(self):
        return 0


def run_world(memory, world_id, seed, episodes=50, change_at=25, extra=2, p_touch=0.7):
    world = make_world(world_id, seed)
    if isinstance(memory, Oracle):
        memory.world = world
    changes = {change_at} | set(int(x) for x in world.rng.choice(range(change_at + 5, episodes - 3), size=extra, replace=False))
    succ, steps = [], []
    for ep in range(episodes):
        if ep in changes:
            world.apply_change_event()
        task = world.sample_task(p_touch=p_touch)
        env = SkillEnv(world, task, ep)
        run_planner(env, memory.recall(task, {}))
        memory.observe(env.log)
        succ.append(env.log.success); steps.append(env.log.steps)
    return float(np.mean(succ)), float(np.mean(steps))


def main():
    worlds, seeds = 30, 3
    per = {}
    for name, mk in (("none", BASELINES["none"]), ("oracle", Oracle)):
        per[name] = np.array([[run_world(mk(), w, s)[0] for s in range(seeds)] for w in range(worlds)])
    head = per["oracle"] - per["none"]
    print(f"protocol headroom: oracle {per['oracle'].mean():.2f} - none {per['none'].mean():.2f} = {head.mean():.2f} "
          f"(per-world range {head.min():.2f}..{head.max():.2f})")
    print("normalised scores (mem - none) / headroom:")
    for name in ("last-5", "retrieval", "consolidated"):
        m = np.array([[run_world(BASELINES[name](), w, s)[0] for s in range(seeds)] for w in range(worlds)])
        print(f"  {name:14s} raw {m.mean():.2f}  normalised {((m - per['none']).mean() / head.mean()):.2f}")


if __name__ == "__main__":
    main()
