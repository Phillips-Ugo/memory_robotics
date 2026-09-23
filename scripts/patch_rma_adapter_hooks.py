"""Patch RoboMemArena's eval_common.py so a policy adapter can react to stage progress.

Adds two optional adapter hooks (no-ops if the adapter lacks them):
  adapter.on_stage_done(name, t)     -- called the step a sequential stage check passes
  adapter.on_episode_end(ep_summary) -- called with the per-episode dict (stage_done, stage_steps, TSR...)

Idempotent; run after scripts/patch_rma_stage_timing.py. Used by scripts/03_rma_memory_adapter.py (X5).
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "vendor/RoboMemArena/evaluation_benchmark/scripts/eval_common.py"
s = p.read_text()
if "on_stage_done" in s:
    print("already patched"); raise SystemExit(0)

old1 = '''                    if prev_all_done and check_fn(env):
                        stage_done[name] = True
                        stage_steps[name] = t
'''
new1 = '''                    if prev_all_done and check_fn(env):
                        stage_done[name] = True
                        stage_steps[name] = t
                        _hook = getattr(adapter, "on_stage_done", None)
                        if callable(_hook):
                            _hook(name, t)
'''
assert old1 in s, "stage-timing patch must be applied first"
s = s.replace(old1, new1)

old2 = '''                    "stage_steps": _stage_steps,
                    "total_steps": _total_steps,
                }
'''
new2 = '''                    "stage_steps": _stage_steps,
                    "total_steps": _total_steps,
                }
                _end_hook = getattr(adapter, "on_episode_end", None)
                if callable(_end_hook):
                    _end_hook(ep_summary)
'''
assert old2 in s
s = s.replace(old2, new2)
p.write_text(s)
print("patched", p)
