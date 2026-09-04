"""X2 with the LLM planner and text memory context (abstract env — fast).

    uv run python -m bench.run_llm --backend mock
    ANTHROPIC_API_KEY=... uv run python -m bench.run_llm --backend anthropic --worlds 10 --seeds 1

Same worlds, protocol, budget and metrics as bench.run; only the planner and the
memory representation (text) differ. Outputs outputs/bench_llm_<backend>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import env as _env
from .env import SkillEnv
from .llm_planner import make_backend, recall_text, run_llm_planner
from .memory import BASELINES
from .run import plot, summarize
from .world import make_world


def run_sequence(memory, backend, world_id: int, seed: int, episodes: int, change_at: int, trace_f=None) -> dict:
    world = make_world(world_id, seed)
    rows = []
    for ep in range(episodes):
        if ep == change_at:
            world.apply_change_event()
        task = world.sample_task()
        context = recall_text(memory, task)
        env = SkillEnv(world, task, ep)
        stats = run_llm_planner(env, task, context, backend, trace=trace_f is not None)
        memory.observe(env.log)
        if trace_f is not None:
            trace_f.write(json.dumps({"memory": memory.name, "world": world_id, "ep": ep, "task": task.text,
                                      "props": [world.props.sticky_drawer, world.props.heavy_object],
                                      "context": context, "trace": stats.trace, "events": [e.__dict__ for e in env.log.events],
                                      "success": env.log.success, "steps": env.log.steps}) + "\n")
        rows.append({"ep": ep, "success": int(env.log.success), "steps": env.log.steps,
                     "stale": env.log.stale_actions, "calls": stats.calls, "ctx_chars": len(context)})
    return {"rows": rows, "bytes": memory.bytes_stored()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--change-at", type=int, default=25)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", action="store_true", help="write every prompt context/plan/outcome to trace.jsonl")
    args = ap.parse_args()
    if args.budget is not None:
        _env.STEP_BUDGET = args.budget
    out = Path(args.out or f"outputs/bench_llm_{args.backend}")
    out.mkdir(parents=True, exist_ok=True)

    backend = make_backend(args.backend)
    trace_f = (out / "trace.jsonl").open("w") if args.trace else None
    results = {}
    for name, factory in BASELINES.items():
        runs = [run_sequence(factory(), backend, w, s, args.episodes, args.change_at, trace_f)
                for s in range(args.seeds) for w in range(args.worlds)]
        results[name] = summarize(runs, args.episodes, args.change_at)
        r = results[name]
        calls = sum(x["calls"] for run in runs for x in run["rows"]) / (len(runs) * args.episodes)
        ctx = sum(x["ctx_chars"] for run in runs for x in run["rows"]) / (len(runs) * args.episodes)
        lo, hi = r["success_pre_change_ci"]
        print(f"{name:14s} AUC={r['auc_success']:.3f} pre={r['success_pre_change']:.2f} [{lo:.2f},{hi:.2f}] "
              f"post10={r['success_post_change_first10']:.2f} stale(post)={r['stale_actions_post_change']:.1f} "
              f"llm-calls/ep={calls:.2f} ctx-chars/ep={ctx:.0f}", flush=True)
    if hasattr(backend, "tokens"):
        print(f"backend {backend.name}: {backend.calls} calls, {backend.tokens} tokens")
    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    plot(results, args.episodes, args.change_at, out / "curves.png",
         title=f"Cross-episode memory benchmark v0 — LLM planner ({args.backend}), text context")
    print(f"wrote {out/'curves.png'}")


if __name__ == "__main__":
    main()
