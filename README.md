# memory_robotics — the memory layer for robots

A benchmark for **cross-episode robot memory** (does the robot stop repeating a mistake
it made last week? does it notice when a fact stops being true?) and the memory layer
that passes it. Built in public, from zero robotics experience, starting 27 Aug 2026.

**Status (5 Sep 2026):** benchmark v0.5 runs in an abstract skill simulator (seconds) and
in MuJoCo/robosuite physics (minutes); four baselines + the library prototype; results
in `docs/research-log.md` (dated, with intervals) and `docs/leaderboard.md`.

## The idea in one paragraph

A world is a kitchen with **secrets** — a drawer that sticks, a box that is heavier
than it looks, an object hidden in a drawer, a house rule — that no single task reveals
for free. The robot runs 30–50 tasks in that world; the score is the **shape of the
success curve across tasks**. Halfway through, the world quietly changes. The
benchmark ships a fixed, deliberately dumb planner; entrants submit only a **memory
module** behind two calls: `observe(episode)` after each task, `recall(task)` before.

## Layout

| path | what |
|---|---|
| `bench/` | abstract env, worlds/properties, planner, baselines, runners (`run`, `run_llm`, `run_x3`, `evaluate`) |
| `bench/sim/` | the same benchmark on robosuite/LIBERO physics (`skill_env`, `calibrate`, `run`, renderers) |
| `memlayer/` | the library: SQLite fact store, per-fact-type revision policy, `explain()` |
| `scripts/` | RoboMemArena/openpi harness adapters and GPU-box setup (Phase 1) |
| `docs/` | roadmap, research log, benchmark design, library architecture, report draft, paper notes, posts, figures |

## Run it

```bash
uv sync                                        # main env
uv run python -m bench.run                     # four curves, abstract env, ~2 s
uv run python -m bench.evaluate --baselines    # leaderboard
uv run python -m bench.evaluate --memory memlayer:MemoryLayer --name mine
bash scripts/setup_rma_env.sh                  # physics env (macOS; Linux: setup_gpu_box.sh)
MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.calibrate
MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.run --properties sticky,heavy,location,fast --kinds put,put_any,fetch --calib-log outputs/sim_calibrate4.log
```

## Findings so far (each sourced in the research log)

1. Any memory beats none by ~30 points within three tasks — abstract, physics, and with a language model reading the memory.
2. A memory that never forgets is optimal while the world is static and the only kind that gets worse when it changes (LLM reader: 99% → 66%).
3. Failures are the information-dense episodes — with few options. With many, a single confirming success outweighs any number of eliminations.
4. Revision is worth it only when P(change) × cost(stale) > cost(probe); a success-only leaderboard rewards never revising. Revision policy must be per fact type.
5. Physics teaches what abstraction cannot: close drawers behind you; handles above block lifts; a brushed wall against a sticky drawer breaks a light grip; fixed defaults leak information.

## Author

Belu
