"""Hand-written planner: the fixed policy the benchmark ships.

Uses the cheap skill unless the memory believes the robust one is needed. Recovers
from failures within the episode (a jam -> pull_hard; a drop -> pick_two_hand), so
every task is completable without memory — just over budget.
"""

from __future__ import annotations

from .env import SkillEnv
from .memory import Beliefs


def _open_drawer(env: SkillEnv, beliefs: Beliefs, drawer: str) -> bool:
    if drawer in beliefs.probe_drawers and hasattr(env, "test_drawer"):
        # the memory wants this belief re-checked: a cheap tug, then act on the answer
        sticky = env.test_drawer(drawer).outcome == "sticky"
        if env.done:
            return False
        env.pull_hard(drawer) if sticky else env.open(drawer)
        return not env.done
    if drawer in beliefs.sticky_drawers:
        env.pull_hard(drawer)
    else:  # cheap first
        ev = env.open(drawer)
        if ev.outcome == "jam" and not env.done:
            env.pull_hard(drawer)
    return not env.done


def _grab(env: SkillEnv, beliefs: Beliefs, obj: str) -> bool:
    if obj in beliefs.probe_objects and hasattr(env, "test_object"):
        heavy = env.test_object(obj).outcome == "heavy"
        if env.done:
            return False
        env.pick_two_hand(obj) if heavy else env.pick(obj)
        return not env.done
    if obj in beliefs.heavy_objects:
        env.pick_two_hand(obj)
    else:
        ev = env.pick(obj)
        if ev.outcome == "drop" and not env.done:
            env.pick_two_hand(obj)
    return not env.done


def run_planner(env: SkillEnv, beliefs: Beliefs) -> None:
    task = env.task
    drawers = list(getattr(env, "drawer_names", None) or env.world.drawers)

    if task.kind == "fetch":
        # search order: believed location first, then unknown drawers (sticky ones last),
        # then drawers seen empty since the last sighting
        empty = beliefs.object_not_in.get(task.obj, set())
        rest = [d for d in drawers if d not in empty]
        rng = getattr(env, "rng", None) or getattr(getattr(env, "world", None), "rng", None)
        if rng is not None:  # unknown drawers in random order (no free information from a fixed order)
            rest = [rest[i] for i in rng.permutation(len(rest))]
        rest.sort(key=lambda d: d in beliefs.sticky_drawers)
        order = rest + [d for d in drawers if d in empty]
        if task.obj in beliefs.object_in and beliefs.object_in[task.obj] in order:
            order.remove(beliefs.object_in[task.obj])
            order.insert(0, beliefs.object_in[task.obj])
        found = False
        for d in order:
            if not _open_drawer(env, beliefs, d):
                return
            if d not in env.open_drawers:  # couldn't get it open; try the next drawer
                continue
            ev = env.look_in(d)
            if env.done:
                return
            if ev.outcome == "found":
                found = True
                break
            env.close(d)  # leave it as you found it, or it blocks the drawers below
            if env.done:
                return
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
        if beliefs.preferred_drawer in drawers:
            cands = [beliefs.preferred_drawer]  # the house rule beats everything else
        else:
            allowed = [d for d in drawers if d not in beliefs.rejected_drawers] or drawers
            fast = [d for d in allowed if d in beliefs.fast_drawers and d not in beliefs.sticky_drawers]
            clean = [d for d in allowed if d not in beliefs.sticky_drawers]
            cands = fast or clean or allowed
        # no knowledge => no preference: a random pick, so the default cannot coincide
        # with the good drawer by construction
        rng = getattr(env, "rng", None) or getattr(getattr(env, "world", None), "rng", None)
        drawer = str(rng.choice(cands)) if rng is not None else cands[0]

    if not _open_drawer(env, beliefs, drawer):
        return
    if not _grab(env, beliefs, task.obj):
        return
    env.place(task.obj, drawer)
