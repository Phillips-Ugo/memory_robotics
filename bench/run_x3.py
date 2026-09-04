"""Experiment X3: does failure-aware memory beat success-only memory?

Same mechanism (last-5, retrieval, consolidated) fed (a) every episode, (b) only
successful episodes, (c) only failed episodes. Abstract env, scripted planner.

    uv run python -m bench.run_x3
"""
from __future__ import annotations

import json
from pathlib import Path

from .memory import BASELINES, Memory
from .run import run_sequence, summarize


class Filtered(Memory):
    def __init__(self, inner: Memory, mode: str) -> None:
        self.inner, self.mode = inner, mode

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.inner.name}/{self.mode}"

    def observe(self, log):
        if self.mode == "all" or (self.mode == "success-only" and log.success) or (self.mode == "failures-only" and not log.success):
            self.inner.observe(log)

    def recall(self, task, initial_obs):
        return self.inner.recall(task, initial_obs)

    def bytes_stored(self):
        return self.inner.bytes_stored()


def main() -> None:
    worlds, episodes, seeds, change_at = 30, 50, 3, 25
    out = Path("outputs/bench_x3"); out.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"{'memory / fed with':32s} {'AUC':>5s} {'pre':>5s} {'post10':>6s} {'stale(post)':>11s} {'bytes':>6s}")
    for base in ("last-5", "retrieval", "consolidated"):
        for mode in ("all", "success-only", "failures-only"):
            runs = [run_sequence(Filtered(BASELINES[base](), mode), w, s, episodes, change_at)
                    for s in range(seeds) for w in range(worlds)]
            r = summarize(runs, episodes, change_at)
            results[f"{base}/{mode}"] = r
            print(f"{base + ' / ' + mode:32s} {r['auc_success']:5.2f} {r['success_pre_change']:5.2f} "
                  f"{r['success_post_change_first10']:6.2f} {r['stale_actions_post_change']:11.1f} {r['bytes_stored_mean']:6.0f}")
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
