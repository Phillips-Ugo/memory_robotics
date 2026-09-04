# Research log

## 2026-09-04 — Day 7: X2 in physics — the four curves survive the move to robosuite

**Result** (`bench/sim/run.py`, 5 worlds × 30 episodes × 1 seed, change event at
episode 15, per-task budget = nominal + 260; figure
`docs/figures/bench_sim_curves_2026-09-03.png`):

| memory | AUC | pre-change | post-change (first 10) | stale actions post-change |
|---|---|---|---|---|
| none | 0.49 | 0.52 [0.39, 0.65] | 0.44 | 0.4 |
| last-5 | 0.79 | 0.78 [0.65, 0.87] | 0.80 | 2.2 |
| retrieval (never forgets) | 0.82 | **0.86** [0.74, 0.93] | 0.80 | **3.4, rising** |
| consolidated + probe-after-8 | **0.83** | 0.84 [0.71, 0.92] | **0.84** | 2.4, flattening |

Same shape as the abstract v0 (53/82/89/87 pre-change there): any memory adds ~30
points within 3 episodes; retrieval wins while static and keeps paying for the
fixed drawer afterwards; consolidated is the only one that doesn't lose ground at
the change event. **Caveat stated plainly:** n=5 worlds → ±10-point intervals, so
retrieval vs consolidated is not statistically separated yet; none vs any-memory
is. More worlds/seeds is a background job (~2.5 min per 30-episode sequence).

**Two bugs that would have silently corrupted the result, caught mid-run:**
1. A skill interrupted by the step budget reported `jam`/`drop` — a budget timeout
   was being written into memory as evidence ("this object is heavy"). Now reports
   `timeout`, which no memory treats as evidence. *Lesson: the episode log is the
   memory's training data; anything that isn't an observation must not look like one.*
2. One object sat at the edge of the arm's workspace: reach times were bimodal
   (383 vs 587 steps for the same task) and it occasionally failed a grasp for no
   reason — which the calibration (one seed) had baked into that task's budget.
   Moved it; calibration now runs the nominal rows over several placements and
   the runner takes the median.
Noise floor after fixes: 0 spurious jams, 4 spurious drops in 150 no-memory episodes.

**Also:** the abstract sim earned its keep. Every design decision it forced
(calibrated budget, additive slack, "robust must cost less than failing", success as
a weak revision signal) transferred to physics without change. Fast sim first,
physics second is the right order for benchmark design.

**Next:** scale worlds/seeds in the background for real intervals; then the LLM
planner variant so memory context can be text (needed before any VLA plugs in).

## 2026-09-03 — Day 6: the benchmark gets physics (Phase 2 v0.5)

**Did:** rebuilt the v0 benchmark's skill environment on real physics — a
LIBERO/robosuite scene (Panda, 3-drawer cabinet, 3 box objects) from our own BDDL
file, with the *same* skill interface as the abstract env, so the planner and all
four memory baselines run unchanged (`bench/sim/`). 13.6 ms/step; episodes 5–12 s.
Hidden properties are now physical: sticky drawer = 40 N joint friction; heavy
object = 0.5 kg (light boxes are ~0.01 kg). Calibration matrix (108 episodes, 9
tasks × 4 secret combos × 3 belief states) runs in ~19 min. Video:
`outputs/sim_videos/` — same world with and without memory (1179 vs 922 steps).

**What I learned building it (each a post):**
1. **The Panda can't grab this cabinet's handles.** The handle slot is 1.6 cm deep;
   the closed fingertips are ~1.7 cm. Hours of grasp tuning would test nothing
   about memory. Decision: a *magnetic grasp* (MuJoCo weld constraint) with a
   **force limit** — gentle hook 20 N / light grip 3 N / firm 80 N. The secrets stay
   physical (measured: normal drawer needs 1–4 N sustained, sticky ~32 N; light
   object loads the grasp at 0.1 N, heavy at ~5 N). Documented simplification, not
   a hidden one. Also needed: a 90° gripper yaw so the wide hand clears the handle
   above the one it's grabbing; the bottom drawer needs a higher hook point.
2. **"Robust" skills must cost less than failing.** My first firm skills braced for
   80 steps and knowing a secret cost *more* than failing and recovering — the
   benchmark would have rewarded ignorance. Now: pull_hard +100 vs jam-recovery
   +225; pick_firm +47 vs drop-recovery +111. The v0 abstract ratio (~1.5–2×), rediscovered.
3. **Heavy is slow.** At 1–1.5 kg the arm's force limit makes the firm carry
   physically twice as slow, so knowing "heavy" saved almost nothing. 0.5 kg keeps
   the carry near normal speed and the light grip still fails 30× over margin.
4. **Budgets must be task-relative and additive.** Task lengths vary 350–620 steps
   (near object into top drawer vs far object into bottom), while robust extras
   are ~constant, so budget = nominal(task) + slack, not a multiplier. Slack 175
   makes a jam decisive and a drop survivable — same regime as v0, where success
   curves were driven by jams and drops showed up in steps.

**Next:** run X2 in physics (5 worlds × 30 episodes × 4 memories ≈ 1.5 h) and compare
the four curves to the abstract ones. If the shape holds, the abstract sim earned
its keep as the fast calibration tool.

## 2026-09-03 — Day 5b: M2b scoped — what "reproduce the π₀.₅ baseline" really costs

**Found the baseline's recipe in code, not in the paper.** RoboMemArena's vendored
openpi config carries `_PI05_ROBOMEMARENA_TRAINING_DETAILS`: init from `pi05_base`,
*full* fine-tune, batch 128, 40k steps, cosine LR 5e-5, EMA 0.999, trained on
subtask segments with primitive instructions parsed from filenames ("pick cookies").
The paper itself states none of this (nor how the reactive baseline is prompted at
eval), only category results: task 1's group (Transferring) = 20.0% TSR / 42.8% CSR.
Dataset: 1 TB on HF, 26 tasks × 100 AnyGrasp-generated demos, ~1,076 steps each.

**Decision:** like-for-like is out of budget (4×H100 territory). M2b = a *task-1
specialist*: 27 GB of data, HDF5→LeRobot converter, openpi's LoRA recipe applied to
π₀.₅ (init `pi05_libero`, batch 16, 8k steps), eval task 1 with the full prompt.
≈ $5–10 on an A6000/A100. Reported as what it is — not the paper's number. All
scripts written and load-tested where possible (converter, config patch, data
download, runbook in setup_gpu_box.sh); openpi's LoRA path is π₀-documented, so
the first 20 training steps are the real test.

**Lesson:** "the baseline has code" ≠ "the baseline is reproducible." Check the
training recipe *and* its compute before promising a number.

## 2026-09-03 — Day 5: M2 — π₀.₅ running inside RoboMemArena on a rented 4090

**Did:** first paid GPU session (RunPod, RTX 4090, ~2.5 h). π₀.₅ (`pi05_libero`)
served by openpi over a websocket, RoboMemArena harness driving it through
`scripts/02_rma_pi05_adapter.py`. Full round-trip confirmed: connect → first
inference → 63 s episodes at the full 2500-step horizon (verified: 2500 video
frames), stage scoring + videos.

**Result (official task-1 protocol, 51 trials, seed 50): TSR 0/51, CSR 0.0%.**
Not one first stage (cookies into basket) completed. Diagnostics from the adapter:
the policy *is* acting — mean |delta-pose| 0.05–0.2 per step, gripper toggling, end
effector wandering across the workspace — and a 3-trial test with the images
mirrored (openpi's LIBERO example rotates 180°, the harness only flips vertically)
fails identically. Inputs match openpi's LIBERO contract (256×256 images, 8-dim
state = eef pos + axis-angle + 2 gripper joints).

**Interpretation:** the stock `pi05_libero` checkpoint is fine-tuned on the original
LIBERO suites; RoboMemArena's scenes, 2500-step horizon and two-part prompts are out
of distribution and it flails. The paper's ~21.5% π₀.₅ baseline is almost certainly
π₀.₅ *fine-tuned on RoboMemArena's own dataset* — consistent with the repo shipping
training data + a pointer to openpi's training code, and no checkpoint. To verify
against the paper text next session. **So "reproduce the baseline" = fine-tune first**
(LoRA, >22.5 GB per openpi → borderline on a 4090, comfortable on an A6000/A100),
which is a bigger step than planned. Zero-shot 0/51 is itself a publishable data
point: a frontier VLA fine-tuned for one LIBERO distribution transfers nothing to a
neighbouring one.

**Everything that went wrong, in order (all fixed in the repo now):**
1. `uv` not on PATH in a fresh shell → persisted in `~/.bashrc` by the setup script.
2. **Checkpoint download filled the wrong disk.** RunPod pods have a small container
   disk (`/`, 20 GB) and a volume (`/workspace`, 60 GB); openpi caches under
   `~/.cache` = container disk. 11.6 GB checkpoint → `No space left on device`, and
   the *symptom* was an unrelated-looking TensorStore `OUT_OF_RANGE` byte-range
   error from the truncated file. Fix: `OPENPI_DATA_HOME=/workspace/openpi_cache`.
   Also `uv cache clean` freed 14 GB of wheel cache from the container disk.
3. **gcsfs stalls silently near the end of large objects** on this box (twice, at
   ~95% of a file), and restarting openpi's downloader *appends* to the partial
   files → 23 GB "checkpoint" with one shard truncated and others doubled. Wrote
   `scripts/download_pi05.py`: list via gcsfs, move bytes with `curl` over plain
   HTTPS (`storage.googleapis.com` serves the public bucket) with resume and a
   stall timeout, verify every file's size. The "stuck" 2.18 GB shard was actually
   complete — curl finished it in 2 s. openpi accepts the directory as cached.
4. **First inference kills the websocket.** JAX/XLA compiles on the first request
   (30–90 s); the client's default 20 s keepalive ping times out → `1011 keepalive
   ping timeout`. RoboMemArena's own reference eval has a `StableWebsocketClientPolicy`
   for exactly this; adapter now disables pings the same way.

**Cost:** ~3 h of 4090 (~$1–2 at Community rates) including the download detours. The dead-man's
switch (`sleep 7200; runpodctl stop pod`) is now standard; reset it before any batch.

**Lesson:** on rented boxes, *disk layout* is the first thing to check, not the GPU.
And "the download is stuck at 95%" and "the download is done but the call never
returned" look identical from the log — check bytes on disk, not progress bars.

## 2026-09-02 — Day 4b: benchmark v0 runs — the four-curves chart exists (X2 ✓, X4 preview)

**Did:** built `bench/` — the Phase 2 v0 benchmark as an abstract skill-level
simulator (skills cost steps; hidden properties decide whether the cheap skill works),
a hand-written planner that always recovers in-episode, the `observe()/recall()`
memory API, three baselines (none · last-5 · retrieval) and a consolidated-KB
prototype with a probe-after-N revision rule. 30 worlds × 50 episodes × 3 seeds ×
4 memories runs in ~2 s on the laptop. Chart: `outputs/bench_v0/curves.png`.

**Calibration (the thing v0 was for):** with step budget 30 a single failure still
fit inside the budget (optimal 15 + jam recovery 13 = 28), so success only dropped
when *both* secrets bit → 92–96% ceiling, curves compressed. Budget 26 makes one
failure decide the episode → no-memory 53%, memories 82–89%. That's the memory
dividend, tuned. Budget is now a CLI flag (`--budget`).

**Results (budget 26, 90 runs per curve, pre-change = eps 15–24 with Wilson CI):**

| memory | pre-change success | post-change (first 10) | stale actions after change | bytes |
|---|---|---|---|---|
| none | 0.53 [0.50, 0.57] | 0.54 | 0 | 0 |
| last-5 | 0.82 [0.80, 0.85] | 0.76 | 0.6 | 647 |
| retrieval (never forgets) | **0.89** [0.87, 0.91] | 0.81 | **3.6, growing linearly** | 6409 |
| consolidated + probe | 0.87 [0.85, 0.89] | **0.84** | 1.2, plateaus | 280 |

**What the chart says, in one paragraph:** any memory beats none within 3 episodes
(X2 answered: yes). Before the world changes, raw retrieval is best — perfect
retention is optimal when nothing is stale. After the change, retrieval keeps paying
for a fact that stopped being true (stale actions grow forever; it never re-tests
the fixed drawer because `pull_hard` never produces evidence). last-5 revises "for
free" by forgetting, but pays a periodic re-learning tax in the static phase (0.82
vs 0.89). The consolidated KB is second-best in both regimes and best on the sum,
at 4% of retrieval's storage — the tunable middle of a retention-vs-revision
trade-off that the other two sit at the extremes of.

**Two design lessons for the real benchmark:**
1. **Success is a weak revision signal** when the policy recovers in-episode; every
   memory "recovers" in 1–2 episodes on success. Stale-action count and
   steps-per-episode are the discriminating metrics. Keep them primary for X4.
2. **Revision needs evidence, and robust actions produce none.** A memory that
   always uses the safe skill can never learn the world got better. Probing (spend a
   little to re-test) is the mechanism; its rate is the knob. This is the core
   design problem of Phase 4, found on day one of Phase 2.

**Caveats, stated plainly:** this is an abstract simulator, not robosuite — it tests
the memory logic, not perception or control. Retrieval is token-overlap, not
embeddings. LLM-summary baseline is a stub. The consolidated prototype is ~40 lines;
its probe rule is hand-set. All of these are the point of v0: find the shape before
paying for physics.

**Next:** post #3 material is here (the chart). Then robosuite skills behind the same
`SkillEnv` interface, and an LLM-planner variant so the memory context can be text.

## 2026-09-02 — Day 4: prior-work correction (RoboMME-Interference) + M2 prep + benchmark spec

**Belu found the one paper that could have sunk post #1.** RoboMME-Interference
(arXiv 2606.22338, June 2026) is a genuinely cross-episode benchmark — sessions are
separate episodes with resets. The post's "no benchmark for Half B. None." line was
wrong and is now fixed everywhere (post, roadmap, benchmark spec; new note
`paper-notes/04-robomme-interference.md`).

**Why it sharpens rather than kills the thesis:** their memory is *given* (a demo
video the query explicitly references) and their world is *static* (distractors are
irrelevant, never conflicting). Result: 45% → 19% under 7 distractors, fully restored
to 45% by a SigLIP visual-similarity retrieval step. Retrieval solves it because
retrieval is all it demands — roadmap principle #4 confirmed by someone else's data.
Their stated limitation ("depends on the query resembling its demonstration") and
future work ("experience spread across several sessions") are our two open axes:
earned memory and revision. Their retrieval fix becomes a mandatory baseline for X4.

**Also today:** M2 code ready (`scripts/02_rma_pi05_adapter.py` load-tested in the
harness venv, `scripts/setup_gpu_box.sh` with runbook); benchmark v0 spec written
(`docs/benchmark-design.md`); post #1 drafted in LinkedIn + X versions.

**Lesson:** "none exists" is the most dangerous sentence you can post. Say instead
what the existing thing doesn't do — more specific, more defensible, and it forces
you to read the thing.

## 2026-08-30 — Day 3: Phase 0 reading done + M2 fully scoped (no GPU rented yet)

**Did (reading):** Read all four Phase 0 papers; notes in the paper-notes stubs +
notebook. The two ideas that survived the pressure-test and will anchor blog post
#1: (1) every open question in the workshop's list has a within-episode version
(Problem A, what everyone works on) and a cross-episode version (Problem B, open);
(2) RoboMemArena the *benchmark* and PrediMem the *method* are two different
contributions in one paper — I care about the benchmark.

**Did (M2 recon):** answered all four pre-rental questions in docs/phase1-plan.md.
The findings that changed the plan:
- RoboMemArena vendors openpi (`third_party/openpi_minimal`) with a websocket
  policy server whose LIBERO mode defaults to the exact checkpoint we need
  (`pi05_libero` from `gs://openpi-assets`). The eval side ships the matching
  client + obs adapter. M2 is assembly, not integration.
- openpi README: π₀.₅ **inference needs >8 GB VRAM** — a single RTX 4090 (the
  cheapest mainstream rental tier), not the 24–48 GB I guessed. LoRA >22.5 GB,
  full fine-tune >70 GB. Ubuntu 22.04 only.
- The MemER repo cannot reproduce the benchmark's MemER number: high-level code
  only, no sim configs, no π₀.₅ checkpoint, one single-task HF checkpoint — and
  the authors' own materials disagree on the model size (3B page / 7B abstract /
  4B checkpoint). RoboMemArena publishes no MemER adapter either, so nobody's
  MemER-on-RoboMemArena setup is public. → M3 rescoped to RoboMemArena's own
  Qwen-VL keyframe pipeline; official-MemER reproduction demoted to stretch.
- MemER interface fact I had wrong from memory: keyframes never go to the
  low-level policy — the VLM emits `{current_subtask, keyframe_positions}` and
  the low level gets *language only*. The median-frame thing is per-cluster
  dedup after single-linkage clustering, not the selection rule.

**Lesson:** a "reproduce the baselines" plan is only as real as the repos behind
it. One hour of recon (two parallel doc-reading agents + grepping the vendored
clone) moved M2 from "rent a big box and figure it out" to a one-line serve
command on the cheapest GPU tier, and killed an M3 that would have burned a week.

**Next:** rent a 4090 (RunPod/Lambda/Vast, Ubuntu 22.04), run M2, get the first
reproduced π₀.₅ number to compare against the paper's 21.5%.

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
