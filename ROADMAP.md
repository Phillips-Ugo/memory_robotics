# Roadmap: the memory layer for robots

*Belu · started 27 Aug 2026 · six-month horizon · revise as reality intrudes*

## North star

> A policy-agnostic memory layer — **store, consolidate, retrieve, revise** — that sits
> between a robot's past episodes and its current policy, measurably improves success as
> experience accumulates, and does so more efficiently than raw retrieval over trajectories.
> Deliverables: (1) a benchmark that measures that claim, (2) a library that makes it true.

## The one distinction that organizes everything

- **Problem A — within-episode memory.** Did the policy remember which cup it filled
  800 steps ago? Mature field: benchmarks (RoboMemArena, RoboMME, RMBench), leaderboards,
  strong baselines (π₀.₅ ~21.5%, MemER ~27.3%, PrediMem ~38.5% on RoboMemArena).
  *Use it to learn the VLA stack, not as the main target.*
- **Problem B — cross-episode memory.** Does the robot stop repeating a mistake it made
  last week? Immature. The one existing cross-episode benchmark, RoboMME-Interference
  (June 2026, `docs/paper-notes/04-robomme-interference.md`), hands the robot a
  demonstration and inserts *irrelevant* sessions before the query — memory is given,
  not earned, and nothing ever changes, so a retrieval-only fix fully solves it.
  Nobody measures **experiential** memory (facts the robot earns by acting) or
  **revision** (facts that stop being true). Other prior work is LLM-agent memory
  (Voyager, Reflexion, Generative Agents, MemGPT/Letta, Mem0).
  **This is the target — a new entrant can define the frontier here.**

## Why this survives the "models will eat it" bear case

Long context + fine-tuning could absorb generic memory. What they can't absorb:
**per-deployment facts change faster than fleet retraining cycles.** "This drawer sticks"
is local, mutable, sometimes private to the customer — it must live outside the weights.
And raw retrieval fails at **staleness** (the drawer got fixed; stop avoiding it).
Consolidation-with-revision is the moat. Experiment X4 is the proof.

## Operating principles

1. **The benchmark is the product wedge.** No cross-episode benchmark exists; building one
   is a research contribution, a visibility engine, and the proof my library works
   (the Mem0 playbook: ship benchmark + library together).
2. **Always beat the dumbest thing that could work.** Baselines for every experiment:
   no memory · last-k episodes in context · embedding retrieval · LLM summary of the past.
   A fancy mechanism only counts if it wins at matched compute/cost.
3. **Keep the policy swappable.** Scripted skills → VLM planner → fine-tuned VLA. Never
   entangle "is my memory good" with "is my policy good."
4. **Design for change.** A world that never changes only tests retrieval. Inject change
   events; measure revision speed.
5. **One causal claim per experiment.** "Under conditions C, mechanism X raises success by
   Y at Z% less memory." Design so negative results are also publishable.
6. **Every phase ends in something public.** If a phase runs long, cut scope inside the
   phase — never skip the next phase.

## Phases

### Phase 0 — Run a robot policy, any policy (weeks 1–2) ← *now*
- [x] Repo, uv environment, LeRobot installed
- [x] Success-rate number I produced myself (`scripts/00_eval_pretrained.py` — 65% on PushT, n=20, CI [43%, 82%], 28 Aug)
- [x] Experiment tracking set up (W&B, offline mode; `uv run wandb login` to sync online — 28 Aug)
- [x] Four one-page paper notes: CoRL memory-workshop page, RoboMemArena, MemER, RoboMME (read 29–30 Aug; notes in `docs/paper-notes/` + notebook)
- [ ] In my own words: the Problem A / Problem B distinction (blog post #1)
- **Learn:** Python/PyTorch basics, git, running on remote GPU, LeRobot tutorial.
- **Skip:** kinematics, control theory, ROS.

### Phase 1 — Reproduce the frontier on Problem A (weeks 3–7)
- [x] RoboMemArena running on cloud GPU with π₀.₅ served by openpi (3 Sep; stock checkpoint scores 0/51 zero-shot — see log Day 5)
- [ ] Reproduce π₀.₅ reactive baseline — requires fine-tuning on RoboMemArena data first (M2b); MemER official repo can't reproduce it (see phase1-plan)
- [ ] Probes: shrink/grow keyframe bank, add distractors, find where memory *hurts* (**X1**)
- [ ] Public write-up: "I reproduced two memory-VLA baselines; here is exactly where they break"
- **Learn:** behavior cloning and its failures, diffusion/flow policies (inputs/outputs level),
  VLA architecture, POMDPs as a concept, the benchmark code itself.
- **Skip:** offline RL, world models, sim-to-real.

### Phase 2 — Smallest possible cross-episode experiment (weeks 8–11)
- [x] One world in LIBERO (robosuite), 2 hidden persistent properties (sticky drawer = joint friction, heavy object = mass), scripted-skill policy with force-limited magnetic grasps (3 Sep)
- [x] Baselines: none · last-k · retrieval (token-overlap stand-in) · consolidated-KB prototype; LLM-summary still a stub (2 Sep, `bench/`)
- [x] **X2 (v0, abstract sim):** any memory beats none within 3 episodes; retrieval best while static, consolidated best after a change event (2 Sep)
- [x] Output: `outputs/bench_v0/curves.png` — four curves + the retention-vs-revision hypothesis (research log Day 4b)
- [x] Same experiment with robosuite skills behind the same interface (`bench/sim/`) — four curves hold in physics: none 0.49 / last-5 0.79 / retrieval 0.82 / consolidated 0.83 AUC (4 Sep; 5 worlds, needs more seeds for tight intervals)
- [ ] LLM-planner variant so memory context can be free text
- **Learn:** one sim framework in depth, procedural task generation, LLM agent/tool-use
  patterns, embedding retrieval, the LLM-agent memory papers.

### Phase 3 — Build the benchmark (weeks 12–18)
- [ ] 8–12 tasks, 5–6 hidden-property types, procedural world generation
- [ ] Change events + episodes-to-recovery metric; cross-episode complexity score
- [ ] Ground-truth "which past episodes were relevant" annotations (I control the generator)
- [ ] Evaluation harness, baseline table, leaderboard page
- [x] **X3** (abstract, 4 Sep): success-only memory = no memory (0.54); failures-only best (0.95) but cannot revise; principle: store failures to learn, contradicting successes to revise
- [ ] Output: public repo + 4–6 page report (arXiv early 2027)

### Phase 4 — The memory layer itself (weeks 19–24)
- [ ] Experience store → consolidation into structured facts (evidence, confidence, freshness)
- [ ] Retrieval keyed on task/object/place/failure; revision rule for contradictions
- [ ] **X4:** recovery speed after change events vs all baselines — *the* differentiating result
- [ ] API: `memory.observe(episode)` / `memory.recall(task, observation)`
- [ ] Plug into a VLA on RoboMemArena-style tasks if time permits
- [ ] Output: library v0.1 + results table + paper draft for a 2027 venue (CoRL 2027,
  or RSS/ICRA 2027 workshops)

## The four experiments

| # | Question | Comparison | Decides |
|---|----------|-----------|---------|
| X1 | Where does in-episode memory help or hurt? | π₀.₅ vs MemER vs ablated MemER, across task categories | Whether I understand the stack; first public result |
| X2 | Does cross-episode memory beat none? Does structure beat raw retrieval? | none · last-k · embeddings · LLM summary · consolidated KB | Whether the knowledge-base thesis survives baselines |
| X3 | Does failure-aware memory beat success-only? | same mechanism fed successes only / everything / failures+recovery annotations | Whether "failure memory" is real or a slogan |
| X4 | How fast does each mechanism recover from a world change? | all X2 mechanisms + change events; episodes-to-recovery, stale-action count | The moat: whether consolidation-with-revision earns its existence |

Follow-on project (not now): adaptive storage policies, memory→simulation, 1,000+ episode scale.

## Metrics

- **Problem A primary:** task success rate as RoboMemArena defines it (comparable to leaderboard).
- **Problem B primary:** the **experience curve** — success (or steps-to-success) vs episode
  index, and its AUC. A memory system is good to the extent it bends this curve up early.
- **Staleness:** episodes-to-recovery after a change event; count of actions taken on stale beliefs.
- **Retrieval quality:** precision/recall against generator ground truth.
- **Cost:** bytes stored, tokens per decision, latency. Report results at matched cost.
- **Honesty:** seeds, confidence intervals, number of worlds. 5 points over 3 seeds on
  10 worlds is noise — say so.

## Learning in public (weekly, not per-phase)

Phase outputs alone are ~1 post/month — too sparse to build an audience. Post **weekly**:
a paper note, a "here's the bug that ate my Tuesday," a single chart. Failure logs
outperform polished write-ups. The research log (`docs/research-log.md`) is the raw
material; write it so a stranger could follow it.

## Startup groundwork (parallel track)

By end of Phase 2: **five conversations** with people who deploy robots (LeRobot Discord,
humanoid startups, lab researchers, CoRL hallway). One question: *"What does your robot
forget, and what does it cost you?"* Their answers shape Phase 3's tasks. A benchmark
built from real forgetting stories is better research **and** the first sales artifact.

## Calendar anchors

- **Sept 24, 2026** — CoRL memory-workshop deadline. *Do not submit* a thin paper; attend
  instead with the Phase 1 write-up in hand and two sharp questions for the speakers.
- **Nov 12, 2026** — CoRL 2026 "Remember, Retrieve, Act" workshop, Austin. Also read
  accepted papers of "Continually Self-Improving Robots" and "Learn@Deploy" (adjacent
  Problem-B communities).
- **Early 2027** — benchmark report on arXiv.
- **CoRL 2027 / RSS 2027** — memory-layer paper target.

## Reading order (resist reading ahead)

1. **Frame (Phase 0):** CoRL 2026 memory-workshop page · RoboMemArena · RoboMME · MemER
2. **In-episode architectures (Phase 1):** MemoryVLA · RMBench/Mem-0 · one of MEM/ECHO/Chameleon (pick the one with code)
3. **Cross-episode, LLM-agent side (Phase 2):** Voyager · Reflexion · Generative Agents · MemGPT/Letta or Mem0 · RoboMemory
4. **Structured/spatial memory (Phase 3):** one scene-graph/3D-memory paper (3D-Mem or ConceptFusion)
5. **Methodology (Phase 3):** "Beyond Binary Success" workshop · LIBERO paper
6. **Foundations (one early weekend):** Sutton & Barto ch. 1–3, 6 + what a POMDP is

*Caveat: the 2026 papers and their numbers came from a Claude-web survey — verify each
against the actual arXiv page when writing its paper note.*
