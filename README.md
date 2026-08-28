# memory_robotics

Building toward a **memory layer for robots** — a policy-agnostic system that lets robots
learn from accumulated experience across episodes: adapt to new environments, stop
repeating old mistakes, and revise beliefs when the world changes.

**The thesis:** per-deployment facts ("this drawer sticks", "this object slips under side
grasps") change faster than fleet retraining cycles, so they can't live in foundation-model
weights. They need a store/consolidate/retrieve/revise layer that sits between a robot's
past episodes and its current policy. The deliverables are (1) a benchmark that measures
that claim and (2) a library that makes it true.

See [ROADMAP.md](ROADMAP.md) for the full plan.

## Repo layout

```
ROADMAP.md            The plan. Phases, experiments, metrics, reading order.
docs/research-log.md  Dated log of what was tried, what happened, what's next.
docs/paper-notes/     One-page notes per paper (four-column format — see TEMPLATE.md).
scripts/              Runnable experiments, numbered in order.
```

## Setup

```bash
# uv manages python + deps (installed via https://astral.sh/uv)
uv sync
uv run python scripts/00_eval_pretrained.py
```

## Author

Belu — learning robotics in public. Progress posted on X and LinkedIn.
