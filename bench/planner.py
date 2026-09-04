"""Hand-written planner: the fixed policy the benchmark ships.

Uses the cheap skill unless the memory believes the robust one is needed. Recovers
from failures within the episode (a jam -> pull_hard; a drop -> pick_two_hand), so
every task is completable without memory — just over budget.
"""

from __future__ import annotations

from .env import SkillEnv
from .memory import Beliefs


def _open_drawer(env: SkillEnv, beliefs: Beliefs, drawer: str) -> bool:
    if drawer in beliefs.sticky_drawers:
        env.pull_hard(drawer)
    else:  # cheap first (also the probe path: re-test a possibly-stale belief)
        ev = env.open(drawer)
        if ev.outcome == "jam" and not env.done:
            env.pull_hard(drawer)
    return not env.done


def _grab(env: SkillEnv, beliefs: Beliefs, obj: str) -> bool:
    if obj in beliefs.heavy_objects:
        env.pick_two_hand(obj)
    else:
        ev = env.pick(obj)
        if ev.outcome == "drop" and not env.done:
            env.pick_two_hand(obj)
    return not env.done


def run_planner(env: SkillEnv, beliefs: Beliefs) -> None:
    task = env.task
    drawers = list(env.world.drawers)

    if task.kind == "fetch":
        # search order: believed location first, then unknown drawers (sticky ones last),
        # then drawers seen empty since the last sighting
        empty = beliefs.object_not_in.get(task.obj, set())
        rest = [d for d in drawers if d not in empty]
        rest.sort(key=lambda d: d in beliefs.sticky_drawers)
        order = rest + [d for d in drawers if d in empty]
        if task.obj in beliefs.object_in and beliefs.object_in[task.obj] in order:
            order.remove(beliefs.object_in[task.obj])
            order.insert(0, beliefs.object_in[task.obj])
        found = False
        for d in order:
            if not _open_drawer(env, beliefs, d):
                return
            ev = env.look_in(d)
            if env.done:
                return
            if ev.outcome == "found":
                found = True
                break
        if not found:
            return
        if not _grab(env, beliefs, task.obj):
            return
        env.place(task.obj, "table")
        return

    # choose the drawer: given for 'put'; for 'put_any' prefer a known fast drawer,
    # avoid known sticky ones, else the first drawer
    if task.kind == "put":
        drawer = task.drawer
    else:
        fast = [d for d in drawers if d in beliefs.fast_drawers and d not in beliefs.sticky_drawers]
        clean = [d for d in drawers if d not in beliefs.sticky_drawers]
        drawer = (fast or clean or drawers)[0]

    if not _open_drawer(env, beliefs, drawer):
        return
    if not _grab(env, beliefs, task.obj):
        return
    env.place(task.obj, drawer)
