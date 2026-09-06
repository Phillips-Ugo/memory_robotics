# Worlds with Secrets: a benchmark for cross-episode robot memory

*Draft v0.1 — 2026-09-05. Working report; numbers cite research-log entries. Target:
4–6 page arXiv report, early 2027. Every claim here is reproducible from `bench/`.*

## Abstract (draft)

Robot memory benchmarks measure whether a policy remembers what happened earlier in
the *same* episode. Deployment runs on a different question: does the robot remember
what it learned *last week* — that this drawer sticks, that this box is heavy, where
the scissors are — and does it notice when those facts stop being true? We introduce
a benchmark for this cross-episode setting. A world is a scene with hidden, persistent
properties that no single episode reveals for free; the robot runs a sequence of
tasks in it, and the score is the shape of the success curve across the sequence.
The benchmark ships the policy: entrants submit only a memory module behind a
two-call API (`observe`, `recall`). Change events flip properties mid-sequence to
measure revision, and because we generate the worlds, we know which past episodes
carry the evidence for each secret, so retrieval quality is scored against ground
truth. We report four baselines in an abstract skill simulator, in MuJoCo physics,
and with a language-model planner reading the memory as text. Memory is worth ~30
points over no memory within three episodes; a memory that never forgets is the
best while the world is static and the only one that gets worse when it changes
(with an LLM reader: 99% → 66%); and memory fed only failures is the strongest of
all but cannot un-learn. [numbers: log Days 4b, 7b, 7d, 7c]

## 1. The gap

- Within-episode memory (RoboMemArena, RoboMME, MemER): mature — benchmarks,
  leaderboards, π₀.₅ ≈ 21.5% avg TSR. Cross-episode: one benchmark
  (RoboMME-Interference, Jun 2026) hands the robot a demonstration and inserts
  irrelevant sessions; memory is *given*, the world never *changes*, and a
  retrieval-only fix solves it (45% → 19% → 45%). Nobody measures memory the
  robot *earns* by acting, or revision when a fact stops being true.
  [paper-notes/04]
- Why it matters: per-deployment facts change faster than fleet retraining cycles
  and are often private to the site — they must live outside the weights. The
  failure mode of raw retrieval is staleness.

## 2. Benchmark design

**Worlds with secrets.** `seed -> (scene, hidden properties)`. Property library:

| type | instantiation | how it manifests | memory dividend | shape |
|---|---|---|---|---|
| mechanism | sticky drawer | gentle `open` jams; `pull_hard` works | skip the jam | failure |
| object physics | heavy object | light `pick` drops; firm grip holds | skip the drop | failure |
| location | object hidden in a drawer | `fetch` needs `look_in`; wrong drawers waste steps | skip the search | search |
| shortcut | fast drawer | `open` cheaper | pick it on free-choice tasks | **success** |

Design rule for every property: completable without the secret, cheaper with it.

**Task kinds.** put X in drawer D · put X away (free choice) · bring X to the table.

**Fixed policy.** A hand-written planner over skills (`open`, `pull_hard`, `pick`,
`pick_two_hand`, `look_in`, `close`, `place`); recovers within the episode; uses the
robust skill only when the memory says so. The benchmark ships this; entrants swap
the memory. (RoboMME's shared-backbone lesson.)

**Memory API.** `memory.observe(episode_log)` after each episode;
`beliefs = memory.recall(task, obs)` before each; `recall_text()` for LLM planners.

**Change events.** At episode 25 of 50 (abstract) / 15 of 30 (physics) a random
property type flips (the sticky drawer is fixed and another sticks; the hidden
object moves; …). Optional extra events at random episodes.

**Budget.** Success = task completed within `nominal(task) + slack` steps, calibrated
so that knowing every secret always fits, a single unrecovered failure never does,
and a drop/one stale action survives (abstract: nominal 15/15/19 + 11; physics:
per-task nominal + 260, fetch 360 + 400). [log Days 4b, 6]

**Metrics.** Experience curve + AUC; pre/post-change success with Wilson intervals;
episodes-to-recovery; stale-action count; wasted looks; retrieval precision/recall
vs ground truth; bytes stored; tokens per decision (LLM).

**Baselines.** none · last-k episodes · retrieval over all episodes (token overlap,
never forgets) · consolidated facts with evidence, freshness and a probe-after-N
revision rule (the library prototype). LLM-summary baseline: TODO.

## 3. Three instantiations

1. **Abstract skill simulator** (`bench/env.py`): 30 worlds × 50 episodes × 3 seeds
   in ~2 s. The design/calibration instrument.
2. **Physics** (`bench/sim/`): LIBERO/robosuite scene — Panda, 3-drawer cabinet, 3
   objects. Sticky = 40 N joint friction; heavy = 0.5 kg; hidden = teleported into a
   drawer; fast = low damping. Magnetic force-limited grasps (documented
   simplification: gentle 20 N / light 3 N / firm 80 N; a sustained overload
   breaks the grasp). 13.6 ms/step, 5–10 s per episode.
3. **LLM planner** (`bench/llm_planner.py`): memories render to text (raw logs or
   fact sentences); claude-haiku-4-5 plans and replans after failures.

## 4. Results (to be consolidated; see research log for tables)

- **X2, abstract** (Day 4b): none 0.54 / last-5 0.80 / retrieval 0.86 / consolidated
  0.85 AUC; retrieval stale actions grow linearly after the change.
- **X2, physics, 600 eps/memory** (Day 7d): none 0.53 [0.46,0.59] → last-5 0.80,
  retrieval 0.87 [0.82,0.91] pre / 0.83 post, consolidated 0.84 / 0.85. Retrieval is
  the only memory whose success falls at the change; 40% more stale actions.
- **X2, LLM reader, 10 worlds** (Day 7b): none 0.52 / last-5 0.93 / retrieval 0.87
  (0.99 → 0.66 at the change) / consolidated 0.95.
- **X3** (Day 7c, 8): success-only memory = no memory (0.54); failures-only best
  (0.95) even with a success-shaped secret present, but cannot un-learn (stale 6.6).
- **Retrieval P/R** vs ground truth: last-5 0.13/0.33, retrieval 0.16/0.50.
- **Phase 3 world, abstract** (Day 8): per-kind — memory +24 on put, +27 on put_any,
  +50 on fetch. **Physics, clean** (Day 9f, 480 eps): none 0.42 → consolidated
  0.85 AUC; put 0.44 → 0.86, put_any 0.48 → 0.86, fetch 0.30 → 0.82; consolidated
  best after the change (0.82) with half retrieval's stale actions.
- **X3 at 10 drawers** (Day 9c): failures-only memory collapses (0.26) when
  elimination is expensive; fed-everything consolidated 0.81. Branching factor is an
  axis of the benchmark.
- **X4 probe-rate sweep** (Day 9e): never re-testing wins on success (0.93) in all
  settings tried — for facts whose staleness only costs steps, probing has negative
  expected value; where staleness fails the task (location, rule) revision happens
  by contradiction. Revision policy must be per fact type.

## 5. Findings (the sentences the paper is for)

1. Any memory beats none by ~30 points within three episodes; in physics too.
2. A memory that never forgets is optimal while the world is static and the only
   kind that gets worse when the world changes. With a language-model reader the
   collapse is 33 points.
3. Revision requires probing: a robot that always plays safe never learns the world
   got better. The probe rate is the central knob of a memory layer.
4. Failures are the information-dense episodes: success-only memory is worthless;
   failures-only is best per byte but cannot un-learn. Store failures to learn; store
   the successes that contradict them to revise.
5. Consolidated facts are the representation a language model uses best, at 4% of
   raw retrieval's storage.
6. Revision is worth it only when P(change) × cost(stale) > cost(probe). For facts
   whose staleness fails the task, the failure is the probe; for the rest, a
   schedule — or never. A success-only leaderboard rewards never revising, which is
   why the benchmark reports steps and stale actions alongside success.
7. Failure memory is best only while elimination is cheap: at 10 options a single
   confirming success outweighs any number of eliminations.
8. Methodological: the episode log is the memory's training data — a budget timeout
   must never look like an observation; an LLM harness must reproduce the
   no-memory row before any LLM-vs-memory claim; and every bug this week was
   invisible in the aggregate and obvious per task kind.

## 6. Limitations / honest caveats

- Scripted planner with privileged state; grasps are magnetic. The policy is not a
  VLA (that is the point — memory is the only variable — but transfer to a learned
  policy is untested; the LLM planner is the first step).
- Retrieval baseline is token overlap, not embeddings; LLM-summary baseline missing.
- Physics: retrieval vs consolidated not separated on success (intervals overlap);
  separated on stale actions and in the LLM run. put_any is trivial in physics
  (shortcut too small); fetch budget needs per-kind calibration.
- Worlds are one scene; property library is 4 types (target 5–6, incl. a property
  revealed only by a fully successful episode).
- RoboMME-Interference's retrieval fix (SigLIP) should be run as a baseline.

## 7. Reproducibility

`uv run python -m bench.run` (abstract, 2 s) · `bench.sim.run` (physics) ·
`bench.run_llm --backend anthropic` · `bench.run_x3`. Figures in `docs/figures/`,
raw episodes in `outputs/*/episodes.jsonl`, every result dated in
`docs/research-log.md`.
