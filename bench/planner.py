"""Hand-written planner: the fixed policy the benchmark ships.

Uses the cheap skill unless the memory believes the robust one is needed. Recovers
from failures within the episode (a jam -> pull_hard; a drop -> pick_two_hand), so
every task is completable without memory — just over budget.
"""

from __future__ import annotations

from .env import SkillEnv
from .memory import Beliefs


def run_planner(env: SkillEnv, beliefs: Beliefs) -> None:
    task = env.task

    # 1. get the drawer open
    if task.drawer in beliefs.sticky_drawers:
        env.pull_hard(task.drawer)
    else:  # cheap first (also the probe path: re-test a possibly-stale belief)
        ev = env.open(task.drawer)
        if ev.outcome == "jam" and not env.done:
            env.pull_hard(task.drawer)
    if env.done:
        return

    # 2. get the object in hand
    if task.obj in beliefs.heavy_objects:
        env.pick_two_hand(task.obj)
    else:
        ev = env.pick(task.obj)
        if ev.outcome == "drop" and not env.done:
            env.pick_two_hand(task.obj)
    if env.done:
        return

    # 3. place
    env.place(task.obj, task.drawer)
