# Research log

## 2026-08-29 — Day 2: RoboMemArena harness running on the Mac (Phase 1 M1 ✓)

**Did:** Got the RoboMemArena eval harness running end-to-end locally with a dummy
policy adapter (`scripts/01_rma_dummy_adapter.py`): env creation, BDDL task 1,
adapter query, 60 sim steps, stage scoring (TSR/CSR = 0%, as a do-nothing policy
should), main + wrist-cam videos saved. ~4s for a 60-step episode on CPU MuJoCo.
Reproducible via `scripts/setup_rma_env.sh` (own venv in gitignored vendor/).

**Four dependency landmines, in the order they fired:**
1. LIBERO's first import blocks on an interactive dataset-folder prompt →
   EOFError in scripts; pipe `N` in once.
2. robosuite 1.4.1 + mujoco 3.x = AssertionError in `get_joint_qpos_addr`
   (joint indexing changed in mujoco 3) → pin mujoco==2.3.7.
3. mujoco 2.3.7 hardcodes the pre-Sequoia OpenGL framework path → one-line sed
   patch to cgl.py.
4. Harness defaults MUJOCO_GL=egl (Linux headless); macOS needs MUJOCO_GL=glfw
   (it's a setdefault, so exporting first wins). Plus imageio[ffmpeg] for videos.

**Learned about the benchmark itself:**
- Adapter contract is genuinely minimal: `infer_actions(obs, prompt, resize_size)
  -> [horizon, action_dim]` float32; obs comes pre-processed
  ('observation/image', 'observation/wrist_image', 'observation/state') with raw
  env obs in `obs['_raw_obs']`. reset() between episodes. This is the plug point
  for a memory layer.
- Scoring is stage-based: CSR = average stage completion, TSR = all required
  stages complete. Counting-pour tasks reject a third pour via a 30-step monitor.
- Official protocol: 51 trials/task, seed 50, max 2500 steps, replan every 10.
- Task prompts are two-part sequential instructions ("pick A into basket, then
  pick B into same basket") — memory-dependence is in the sequencing/occlusion.

**Next (M2):** π₀.₅ inference through this adapter needs a GPU box — the openpi
runtime won't fly on MPS. Before renting, answer the open questions in
docs/phase1-plan.md from the openpi + MemER READMEs.


One entry per working session. Keep it honest: what I tried, what actually happened
(numbers, errors), what I concluded, what's next. This log is the raw material for
weekly public posts — write it so a stranger could follow it.

---

## 2026-08-27 — Day 0: workspace setup

**Did:** Set up the repo, uv environment, LeRobot install, and the Phase 0 eval script.
Wrote the roadmap.

**Next:** Run `scripts/00_eval_pretrained.py` to get my first self-produced success-rate
number. Then paper note #1: RoboMemArena.

**Post idea:** "Day 0 of building the memory layer for robots — here's the plan."

## 2026-08-27 — Day 0, part 2: the normalization bug

**Did:** First eval of `lerobot/diffusion_pusht` on PushT: **0% success over 5 episodes**,
best reward ~0.03. A pretrained policy should get ~60%+.

**Debugging trail (keep this format — it's the post):**
1. Suspected my observation formatting → printed shapes: env gives (96, 96, 3) pixels +
   2-dim state, policy expects exactly that. Not it.
2. Suspected MPS (Apple GPU) numerics → same observation on CPU and MPS gave nearly the
   same action. Not it. But the action VALUE was the clue: `[0.85, 1.0]` — PushT wants
   pixel coordinates in [0, 512]. The policy was outputting *normalized* actions, so the
   agent was pinned into a corner every step.
3. Root cause: the checkpoint is old-format — its normalization stats live inside
   `model.safetensors` under keys like `normalize_inputs.buffer_observation_state.min`.
   LeRobot 0.6 moved normalization out of the policy into processor pipelines, so
   `from_pretrained` drops those buffers with only a log warning
   ("Unexpected key(s) when loading model"). Garbage in (raw pixels where the net expects
   [-1,1]), garbage out (actions never scaled back to pixel space).

**Fix:** load the stats straight from the checkpoint file and apply them manually —
MIN_MAX → [-1, 1] for state/action (lerobot's convention, verified in
`processor/normalize_processor.py`), MEAN_STD for the image.

**Lesson:** silent normalization mismatches don't crash — they produce a policy that
"works" at 0%. Check the *units* of what goes in and comes out of a network before
suspecting anything deeper.

## 2026-08-28 — Day 1: tracking, honest error bars, Phase 1 recon

**Did:**
- Wired Weights & Biases into the eval script (`--wandb off|offline|online`,
  default offline so it works without an account). Logs per-episode success +
  running rate, summary with success rate and CI.
- Added a Wilson 95% confidence interval to the eval output — at n=5, "60%"
  really means "somewhere between ~23% and ~88%", which is why the honesty
  metrics in the roadmap matter.
- Launched a 20-episode eval (detached with nohup + a log-file monitor, since
  it outlives the shell-command timeout).
- Verified all four Phase 0 readings are real; created paper-note stubs with
  links. Found MemER is Oct 2025 and RoboMME has a follow-up (RoboMME-Interference).
- Phase 1 recon (see `docs/phase1-plan.md`): RoboMemArena repo = data + BDDL/LIBERO-style
  eval harness + generic policy adapter; baselines live in external repos (openpi,
  memer-policy/memer — Qwen2.5-VL-3B + π₀.₅). Reproduction will be assembly work
  across three repos, so the milestone ladder starts with the harness + a dummy
  policy, no VLA.

**Lesson:** report an interval, not a point. 3/5 and 12/20 are both "60%" but they
are very different amounts of evidence.

**Gotcha #2 (long-running jobs on a laptop):** the first 20-episode run "ran" for an
hour but consumed only ~1.7 CPU-minutes — macOS put the machine to sleep and froze
the detached process. Also, Python block-buffers stdout when redirected to a file,
so the log looked empty even for completed episodes. Fixes, now standard practice:
- wrap long jobs in `caffeinate -i` (keeps macOS awake while the command runs)
- set `PYTHONUNBUFFERED=1` (or `python -u`) so logs stream line-by-line
- monitors must also detect process death, not just the success line — silence
  looks identical to "still running"
On a rented Linux GPU box none of the sleep issues apply, but unbuffered logs and
death-aware monitors stay best practice.

**Result (attempt 2, ~35 min wall): 13/20 SUCCESS = 65%, 95% CI [43%, 82%],
mean best reward 0.932.** Consistent with yesterday's 3/5. Logged to W&B (offline).
Note how wide the interval still is at n=20 — comparing two policies within ~20
points of each other needs far more episodes than intuition suggests.

## 2026-08-27 — Day 0 result

**Result after the fix: 60% success over 5 episodes (mean best reward 0.988), on MPS.**
Matches the checkpoint's reported ballpark. The two failures reached 0.96–0.98 coverage —
near-misses just under the 0.95-coverage success threshold, not blow-ups. First
self-produced eval number: Phase 0 milestone done. Video of episode 0 in
`outputs/rollout_ep0.mp4`.
