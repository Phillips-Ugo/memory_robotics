# Cross-episode memory benchmark — v0 design spec

*Drafted 2026-09-02. This is the Phase 2 target and the Phase 3 skeleton. Revise as
the v0 experiment (X2) teaches us what's wrong with it.*

## The one idea

A cross-episode benchmark is not a set of tasks. It is a set of **worlds with
secrets**: hidden, persistent properties that no single episode reveals for free but
that a robot with working memory exploits on the next visit. **The score is not
success on any episode — it is the shape of the curve across episodes.**

## Positioning vs. the one existing cross-episode benchmark

RoboMME-Interference (June 2026, `paper-notes/04-robomme-interference.md`) is
cross-episode but tests one axis only: **interference** (k irrelevant sessions between
a given demonstration and the query). Its memory is *given* (a demo video the query
references) and its world is *static*, so a retrieval-only fix solves it completely
(45% → 19% under interference, back to 45% with SigLIP retrieval).

Ours differs on the two axes it leaves open, and reuses its third:

| Axis | RoboMME-Interference | Ours |
|---|---|---|
| Where memory comes from | handed a demonstration | **earned** through the robot's own success/failure |
| Does the world change | never | **yes** — change events, recovery + stale-action metrics |
| Interference | k ∈ {0,1,3,7} irrelevant sessions | reuse as a dimension (Phase 3) |

Their retrieval fix is therefore a **mandatory baseline** for us: it should match the
consolidated-KB approach while the world is static and fall behind at the first
change event. That comparison is X4's headline chart.

## Components

### 1. World generator
`seed -> (scene, hidden_properties)`. A scene is a fixed robosuite/LIBERO-style
tabletop (drawers, containers, objects). Hidden properties are sampled from the
library below. Need **≥30 worlds** for any reported number; one world is an anecdote.

### 2. Hidden-property library (v0: first two types; Phase 3: all)

| Type | Example instantiation | How it manifests | Memory dividend |
|---|---|---|---|
| Mechanism | left drawer sticks | `open(left)` fails/jams; needs `pull_hard` or other drawer | skip the failed attempt (~40–70 steps) |
| Object physics | blue mug is heavy | `pick(mug)` drops unless `pick_two_hand` | skip the drop + re-grasp |
| Location | scissors live in middle drawer | not visible at start; found only by opening | skip the search |
| Preference/rule | "cups go on the top shelf" (feedback given once) | wrong placement → task counted incomplete | skip the correction |
| Procedure | microwave needs a nudge before `open` | `open` no-ops until `nudge` | skip the stall |
| Hazard | the corner tile is slippery / a spot is off-limits | penalty step count / failure on entry | avoid entirely |

Design rule for every property: **completable without the secret, cheaper with it.**
Too easy (discoverable at no cost in-episode) → all baselines saturate. Too brutal
(failure without memory) → nothing gets off the floor. Calibrate the dividend in v0.

### 3. Task sampler
Per episode: a fresh instruction + fresh object placement in the same world.
Each task touches ≥1 hidden property with a known probability (v0: p = 0.7).
Task specs in BDDL (LIBERO's language — already in use for RoboMemArena tasks), so
success predicates are declarative and reusable.

### 4. The fixed policy (the decision that makes or breaks the benchmark)
The benchmark **ships the policy**; entrants submit **only a memory module**.
Otherwise good memory is indistinguishable from good motor skill (RoboMME's shared-
backbone lesson; roadmap principle #3).

- v0 (Phase 2): scripted skills over robosuite primitives —
  `open(x)`, `pull_hard(x)`, `pick(o)`, `pick_two_hand(o)`, `place(o, loc)`,
  `nudge(x)`, `look_in(x)`. CPU-only, seconds per episode, thousands of episodes on a
  laptop. Planner = either a hand-written task planner or a small VLM/LLM that picks
  skills given the instruction + the memory context.
- Later: VLM planner over the same skills; VLA on RoboMemArena-style tasks if time.

**Memory API (identical to the Phase 4 library API — benchmark and library are
co-designed on purpose):**
```python
memory.observe(episode_log)                       # after each episode
ctx = memory.recall(task, initial_obs)            # before each; ctx is injected into the planner prompt/state
```
`episode_log` = instruction, skill sequence, per-skill outcome (ok / fail / jam /
drop …), final success, step count, plus any explicit feedback events.

### 5. Change events
At scheduled episode indices, flip a property (drawer gets fixed; scissors move).
Measures the thing nobody measures: **revision**. Raw retrieval and frozen summaries
fail here by construction; consolidation-with-revision is the only thing that passes.
v0: one change event at episode 25 of 50. Phase 3: multiple, randomized.

### 6. Ground truth for free
Because we generate the world, we know *which past episodes contain the evidence*
for each property. So retrieval can be scored directly (precision/recall of surfaced
episodes vs. relevant ones). No other robotics memory benchmark can offer this.

## Protocol (v0)

- 1 world (v0) → 30 worlds (Phase 3); 50 episodes per world; 3 seeds.
- Baselines, always (roadmap principle #2): **no memory · last-k episodes verbatim
  in context · embedding retrieval over episode logs · LLM summary of all past
  episodes · consolidated KB (ours, Phase 4)**.
- Report at **matched cost** (tokens per decision, bytes stored) — a fancy
  mechanism only counts if it wins at the same budget.

## Metrics

| Metric | Definition | What it tests |
|---|---|---|
| Experience curve + AUC | success (and steps-to-success) vs episode index | primary: does memory bend the curve up early? |
| Episodes-to-recovery | episodes after a change event until performance returns to pre-change level | revision |
| Stale-action count | actions taken on a belief that is no longer true | revision, sharper |
| Retrieval P/R | surfaced vs. ground-truth-relevant past episodes | is the memory retrieving the right things? |
| Cost | bytes stored, tokens/decision, latency | efficiency |
| Honesty | Wilson CIs, seeds, number of worlds | as always |

## Worked example

World #17: left drawer sticks, blue mug heavy.
- Ep 4: drawer task → tries left, jams, recovers via right. Success, 130 steps.
  Log → `observe()`.
- Ep 9: drawer task → `recall()` fires. Does the planner context now say "left
  drawer jams"? Right drawer first try: 60 steps.
- Ep 25: change event — drawer silently fixed.
- Ep 26–30: does the system ever re-try left and update, or avoid it forever?
  Log the curve, recovery lag, stale actions. Repeat ×30 worlds ×3 seeds.

## Phase 2 = v0 = experiment X2

One world, two hidden properties (mechanism + object physics), scripted skills, the
four dumb baselines, one chart of four curves. If any daylight shows between the
curves, the full benchmark is worth building and the chart is a post. If none does,
the calibration is wrong (see design rule) — fix the dividend before scaling.

## v0 status (2026-09-02): built and run — `bench/`, `uv run python -m bench.run`

Abstract skill-level simulator (not robosuite yet), hand-written planner, three
baselines + consolidated prototype, 30 worlds × 50 eps × 3 seeds in ~2 s. Findings
that feed back into this spec (details in research log, Day 4b):

- **Calibration is a step budget.** Budget must make a *single* failure decide the
  episode (v0: budget 26, optimal path 15). Otherwise the ceiling compresses curves.
- **Success is a weak revision metric** when the policy recovers in-episode. Stale
  actions and steps/episode are the discriminating metrics → make them primary for X4.
- **Robust actions produce no evidence.** A memory that always plays safe can never
  learn the world got better; revision requires probing, and the probe rate is the
  central knob. Retrieval-never-forgets accumulates stale actions linearly forever.
- The retention-vs-revision trade-off is real and measurable: retrieval wins static,
  last-k wins on revision by forgetting (but pays a re-learning tax), consolidated
  sits in the tunable middle at ~4% of retrieval's storage.

## Open questions (answer in v0)

- Which sim: our existing robosuite/LIBERO fork (already running, BDDL tasks) vs
  ManiSkill3 (faster, more objects, but a new stack)? Default: **stay on robosuite** —
  the stack is paid for.
- How much of the "planner" should be an LLM vs hand-written? Hand-written first: it
  makes memory the *only* moving part.
- ~~Does the memory context get injected as text, as structured state, or both?~~
  Both, now (`bench/llm_planner.py`, 2026-09-04): `recall()` returns structured
  beliefs for the scripted planner; `recall_text()` renders each memory as text
  (raw logs for last-k/retrieval, fact sentences with evidence + age for the
  consolidated store) for an LLM planner that replans after failures. A mock
  reader reproduces the scripted results exactly; the LLM run (`bench/run_llm.py
  --backend anthropic`) is the first place "structure beats raw retrieval" is a
  genuinely open question, because the planner has to *interpret* the context.
- What's the smallest property library that still separates the baselines? Find out.
