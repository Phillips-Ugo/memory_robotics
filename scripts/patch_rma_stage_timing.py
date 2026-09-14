"""Make RoboMemArena's harness record WHEN each stage completed and dump results to JSON.

Two idempotent edits to vendor/RoboMemArena/evaluation_benchmark/scripts/eval_common.py:
  1. run_episode_with_stages also returns {stage_name: step_index}  (stage_steps)
  2. run_eval writes {video_dir}/results.json with the full per-episode list
Needed by memlayer.adapters.robomemarena (stage cost -> slow/easy facts).

    python3 scripts/patch_rma_stage_timing.py
"""
from __future__ import annotations

import pathlib
import re

p = pathlib.Path("vendor/RoboMemArena/evaluation_benchmark/scripts/eval_common.py")
s = p.read_text()
if "stage_steps" in s:
    print("already patched"); raise SystemExit

s = s.replace('''    stage_done = {name: False for name, _ in stage_checks}
    t = 0''', '''    stage_done = {name: False for name, _ in stage_checks}
    stage_steps: dict[str, int] = {}
    t = 0''')
s = s.replace('''                    if prev_all_done and check_fn(env):
                        stage_done[name] = True
                        logging.info(f"  [t={t}] Stage completed: {name}")''', '''                    if prev_all_done and check_fn(env):
                        stage_done[name] = True
                        stage_steps[name] = t
                        logging.info(f"  [t={t}] Stage completed: {name}")''')
s = s.replace('''                    if not stage_done[name] and check_fn(env):
                        stage_done[name] = True
                        logging.info(f"  [t={t}] Stage completed: {name}")''', '''                    if not stage_done[name] and check_fn(env):
                        stage_done[name] = True
                        stage_steps[name] = t
                        logging.info(f"  [t={t}] Stage completed: {name}")''')
s = s.replace('''    goal_success = bool(stage_done) and all(stage_done.values())
    return score, stage_done, goal_success, replay, replay_wrist''', '''    goal_success = bool(stage_done) and all(stage_done.values())
    stage_done["__stage_steps__"] = stage_steps  # type: ignore[assignment]
    stage_done["__total_steps__"] = t  # type: ignore[assignment]
    return score, stage_done, goal_success, replay, replay_wrist''')
s = s.replace('''                score, stage_done, goal_success, replay, replay_wrist = run_episode_with_stages(''', '''                score, stage_done_raw, goal_success, replay, replay_wrist = run_episode_with_stages(''')
s = s.replace('''                for name in stage_done:
                    stage_totals[name] += int(stage_done[name])
                tsr_success = bool(stage_done) and all(stage_done.values())''', '''                _stage_steps = stage_done_raw.pop("__stage_steps__", {})
                _total_steps = stage_done_raw.pop("__total_steps__", None)
                stage_done = stage_done_raw
                for name in stage_done:
                    stage_totals[name] += int(stage_done[name])
                tsr_success = bool(stage_done) and all(stage_done.values())''')
s = s.replace('''                    "stage_done": stage_done,
                }''', '''                    "stage_done": stage_done,
                    "stage_steps": _stage_steps,
                    "total_steps": _total_steps,
                }''')
s = s.replace('''    return {
        "task_id": tid if tid is not None else task_id,
        "task_key": task_key,''', '''    result = {
        "task_id": tid if tid is not None else task_id,
        "task_key": task_key,''')
s = s.replace('''        "CSR": float(goal_pct),
        "episodes": episodes,
    }''', '''        "CSR": float(goal_pct),
        "episodes": episodes,
    }
    try:
        (Path(video_dir) / "results.json").write_text(json.dumps(result, indent=1, default=str))
        logging.info(f"Results JSON: {Path(video_dir) / 'results.json'}")
    except Exception as e:  # noqa: BLE001
        logging.warning(f"could not write results.json: {e}")
    return result''')
assert s.count("stage_steps[name] = t") == 2 and '"stage_steps": _stage_steps' in s and "results.json" in s, "patch anchors changed"
p.write_text(s)
print("patched", p)
