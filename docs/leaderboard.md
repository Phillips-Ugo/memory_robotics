# Leaderboard — cross-episode memory benchmark (abstract env, Phase 3 protocol)

Protocol: 30 worlds × 50 episodes × 3 seeds; properties sticky, heavy, location, fast, preference; tasks put, put_any, fetch; change at 25 + 2 random. Higher AUC is better; read stale actions and steps next to it (a success-only ranking rewards never revising).

| memory | AUC | pre-change [95% CI] | post-change (10) | stale (post) | steps | put | put_any | fetch | bytes | ret P/R | date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| consolidated | **0.865** | 0.93 [0.92, 0.95] | 0.84 | 1.0 | 19.9 | 0.87 | 0.83 | 0.90 | 457 | — | 2026-09-05 |
| memlayer-L2 | **0.865** | 0.93 [0.92, 0.95] | 0.84 | 1.0 | 19.9 | 0.87 | 0.83 | 0.90 | 508 | — | 2026-09-05 |
| retrieval | **0.837** | 0.90 [0.88, 0.92] | 0.81 | 2.5 | 19.9 | 0.82 | 0.80 | 0.89 | 7233 | 0.56/0.58 | 2026-09-05 |
| last-5 | **0.766** | 0.81 [0.78, 0.83] | 0.75 | 0.8 | 20.2 | 0.76 | 0.71 | 0.83 | 740 | 0.31/0.34 | 2026-09-05 |
| none | **0.337** | 0.33 [0.30, 0.36] | 0.35 | 0.0 | 24.8 | 0.48 | 0.22 | 0.32 | 0 | — | 2026-09-05 |

Submit: `uv run python -m bench.evaluate --memory <module_or_file>:<factory> --name <name>` (see `bench/evaluate.py` for the memory interface).
