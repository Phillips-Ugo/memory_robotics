# Research log

## 2026-09-05 — Day 9h: interference reproduces once the world is large

Large-entity world: 10 drawers × 12 objects, put tasks, sticky + heavy secrets,
20 worlds × 80 episodes × 2 seeds. Success on secret-touching tasks:

| interference (1 − p_touch) | last-5 | retrieval | consolidated |
|---|---|---|---|
| 0.1 | 0.77 | 0.82 | 0.89 |
| 0.5 | 0.66 | 0.80 | 0.86 |
| 0.8 | **0.49** | 0.75 | 0.76 |

Now the RoboMME-Interference pattern appears: the window memory (last-5) loses 28
points as unrelated tasks fill its window, keyed retrieval loses 7, consolidated 13
(at 0.8 both keyed memories are evidence-starved — a secret is touched every ~5
episodes across 22 entities — not clutter-limited). In the small world (Day 9g) all
three were flat. So interference is a real axis, but it only bites when the
entity count is large relative to the memory's horizon; the benchmark should
expose entity count as a first-class parameter (`run_sequence(..., drawers=,
objects=)` does).

Leaderboard refreshed with the TF-IDF baseline: 0.74 (fetch 0.67 — similarity
retrieval is worst exactly where the relevant episode is a *different task kind*).

## 2026-09-05 — Day 9g: similarity retrieval loses to entity keys; interference does not touch keyed memories

**TF-IDF retrieval baseline** (`retrieval-tfidf`: cosine over the full log text of
every stored episode, recency as tie-break — the closest cheap stand-in for
embedding retrieval): AUC 0.77 vs 0.86 for entity-keyed retrieval (token overlap on
the *task* text), and it collapses harder at the change event (post 0.68 vs 0.84).
Similarity over event-rich logs surfaces episodes that *look like* the query rather
than ones *about the same drawer*, and long old logs dominate. RoboMME-Interference's
own caveat ("the fix depends on the query resembling its demonstration") shows up
here as a 9-point gap. Entity keys first, similarity as fallback — as the
architecture doc says.

**Interference axis** (`--p-touch`; 1 − p_touch = share of tasks that touch no
secret). Success on secret-touching tasks only, 30 × 50 × 3:

| interference | last-5 | retrieval | retrieval-tfidf | consolidated |
|---|---|---|---|---|
| 0.1 | 0.75 | 0.85 | 0.75 | 0.87 |
| 0.3 | 0.76 | 0.85 | 0.75 | 0.87 |
| 0.5 | 0.76 | 0.84 | 0.74 | 0.86 |
| 0.7 | 0.77 | 0.85 | 0.75 | 0.86 |

Flat. Where RoboMME-Interference's in-context memories fell 45 → 19 as unrelated
sessions piled up, keyed and consolidated memories are immune by construction, and
even last-5 holds because with three drawers and four objects the relevant entity
recurs within its window. To reproduce their decay one needs a window shorter than
the gap between relevant episodes — a large-entity world, which is the 10-drawer
variant's cousin and another axis worth exposing. The two benchmarks measure
different things: theirs, whether a memory survives clutter; ours, whether it
survives change.

## 2026-09-05 — Day 9f: Phase 3 physics, clean (5 worlds × 24 episodes, 480 episodes)

After the fetch fixes (Day 9d): `docs/figures/bench_sim_p3c_*`.

| memory | AUC | pre → post | put | put_any | fetch | wasted looks/fetch | stale (post) |
|---|---|---|---|---|---|---|---|
| none | 0.42 | 0.46 → 0.32 | 0.44 | 0.48 | 0.30 | 0.83 | 0.2 |
| last-5 | 0.78 | 0.88 → 0.74 | 0.74 | 0.93 | 0.63 | 0.40 | 1.6 |
| retrieval | 0.82 | 0.90 → 0.80 | 0.82 | 0.84 | 0.77 | 0.17 | 2.2 |
| consolidated | **0.85** | 0.90 → **0.82** | **0.86** | 0.86 | **0.82** | 0.18 | 1.0 |

Memory is worth **+43 points** in the full physics world; fetch goes 0.30 → 0.82; no
fetch failure involves a drop any more. Consolidated leads on AUC, on put, on fetch,
and after the change event, with half retrieval's stale actions; retrieval leads
before the change (its usual pattern). Steps tell the same story: fetch 723 → 547.
Intervals are still ~±10 at n=5 worlds — a 10×2 run is the next background job.

**Physics vs abstract, same world design:** abstract none 0.34 / consolidated 0.87;
physics 0.42 / 0.85. The abstract simulator predicted the physics ranking and the
size of every effect to within a few points, after six physics-only lessons were
folded back in. That is the argument for the two-tier design.

## 2026-09-05 — Day 9e: X4 — when is revision worth it? (probe-rate sweep)

Abstract env, 30 worlds × 60 episodes × 3 seeds, three change events (ep 20 + two
random). Consolidated memory with `probe_after` ∈ {2, 4, 8, 16, never}. Probing was
first "try the cheap skill on a believed-sticky drawer"; then a cheap **test skill**
was added (6 steps: a light tug / small lift that reveals the property without a full
failure — jam 13, drop 10); then the robust skills were made expensive (stale action
+11 / +8 instead of +5 / +4).

| probe_after | 2 | 4 | 8 | 16 | never | (retrieval) |
|---|---|---|---|---|---|---|
| AUC, original costs | 0.73 | 0.79 | 0.84 | 0.89 | **0.93** | 0.84 |
| AUC, cheap test skill | 0.80 | 0.84 | 0.87 | 0.90 | **0.93** | 0.84 |
| AUC, + expensive staleness | 0.80 | 0.84 | 0.88 | 0.90 | **0.93** | 0.84 |
| stale actions (post) | 0.1 | 0.4 | 0.8 | 1.7 | 6.4 | 3.5 |

**Never probing wins on success in all three settings.** The reason is structural,
not a bug: for sticky/heavy, the robust action is *always safe* — acting on a stale
belief costs a few steps and never fails the task — while a probe costs steps on
every fact that aged, whether or not it changed. Expected value of probing =
P(changed) × cost(stale) − cost(probe), and with three drawers, a change every ~15
episodes, and stale ≈ probe cost, it is negative. A memory that never re-tests
sticky/heavy facts is *rationally* right in this world; the sweep exposes the knob,
it does not vindicate probing.

**Where revision does pay** is where a stale belief *fails the task*: a stale
location sends the robot to an empty drawer (wasted look, budget blown); a stale
house rule gets the placement rejected. There the failure itself is the probe, and
all memories revise by contradiction — no schedule needed. So the memory layer's
revision policy should be **per fact type**: contradiction-driven for facts whose
staleness fails tasks, schedule-driven (with the probe rate set by P(change) ×
cost(stale) / cost(probe)) for facts whose staleness merely costs efficiency, and
"never" when that ratio is below one.

**Benchmark consequence:** to *test* revision policies, the world must contain
facts across that spectrum and report steps and stale actions next to success —
which it now does. A success-only leaderboard would reward never revising. Left
as-is (costs restored) for the physics run; `bench/env.py` has the `test_*` skills.

## 2026-09-05 — Day 9d: Phase 3 physics, corrected (bugs out), and three more physics lessons

Rerun after the in-drawer-test and random-default fixes (4 worlds × 24 episodes,
change at 12; `docs/figures/bench_sim_p3b_*`):

| memory | AUC | put | put_any | fetch | wasted looks/fetch |
|---|---|---|---|---|---|
| none | 0.43 | 0.44 | 0.46 | 0.36 | 0.86 |
| last-5 | 0.77 | 0.74 | 0.92 | 0.58 | 0.42 |
| retrieval | 0.77 | 0.84 | 0.84 | 0.59 | 0.22 |
| consolidated | **0.80** | 0.85 | 0.86 | 0.65 | 0.23 |

No-memory fell from 0.68 to 0.43 once the naive planner stopped getting free
information from a fixed default drawer; memory is now worth +34 to +37 points in
the full physics world (was +11–14). put_any separates properly (0.46 vs 0.84–0.92).
`not_here` phantom failures: 0.

**Fetch still ~0.6 with memory — traced to three physics effects, all fixed:**
1. After a *hard* pull a sticky drawer opened only 11 cm (vs 14), leaving the hidden
   object under the handle bar of the drawer above; the lift jammed. Firm pull now
   travels farther/longer so a sticky drawer ends as open as a normal one.
2. The yawed hand spans the drawer's full width and brushes the walls on the way up.
   A normal drawer yields a little; a sticky one (40 N friction) does not, so the brush
   becomes a sustained load that pops the 3 N light grip. Resolution: lifting out of
   a drawer is always the careful maneuver — the env uses the firm grip for in-drawer
   picks without recording the robust skill (the hidden object is never the heavy
   one, so nothing is masked). Known-location fetch now costs ~420 steps; budget
   nominal moved 360 → 420.
3. Objects are slid forward 5 cm before lifting out of a drawer.
All six (drawer × sticky/normal) fetch cases now succeed. Rerun launched.

**Meta-lesson for the report:** every one of today's bugs was invisible in the
aggregate and obvious in a per-kind or per-event breakdown. The physics env has
now taught six things the abstract sim structurally cannot (close drawers behind
you; handles above block lifts; walls + friction make grips break; hard pulls travel
less; heavy is slow; fixed defaults leak information).

## 2026-09-05 — Day 9c: X3 flips when elimination is expensive

Follow-up to Day 9b: 10 drawers instead of 3, `put_any` tasks only, secrets = sticky
drawer + house rule (30 worlds × 50 × 3 seeds, abstract env). No-memory = 0.08.

| fed with | last-5 | retrieval | consolidated |
|---|---|---|---|
| everything | 0.76 | 0.67 | **0.81** |
| successes only | 0.51 | 0.51 | 0.51 |
| failures only | **0.26** | 0.24 | 0.62 |

With three drawers, two rejections identified the right one and failures-only
memory won (Days 7c, 9b). With ten, the raw-log failure-only memories collapse
(0.26 — the last five failures never contain the one praised placement), and only
the consolidated store salvages it by accumulating eliminations (0.62). Fed
everything, consolidated is best (0.81).

**Revised principle:** failures are the information-dense episodes when the option
space is small; when it is large, a single confirming success is worth more than
any number of eliminations. A memory layer must keep both — failures to learn what
to avoid, the confirming success to learn what to do — and the balance is a property
of the *world's branching factor*, which the benchmark should vary explicitly
(3 vs 10 options is now a one-flag change: `run_sequence(..., drawers=...)`).

## 2026-09-05 — Day 9b: the success-only property, and what X3 looks like with it

**Added** `preference` (abstract env): a house rule for "put X away" — only one
drawer counts. Placing there yields `place->praised` (evidence that exists only in a
*successful* episode); elsewhere yields `place->rejected` (elimination evidence).
No-memory success on the full world drops to 0.34 (random 1-in-3 on put_any).

**X3 with it present** (30 × 50 × 3): overall, failures-only is *still* best
(0.92 vs 0.79 fed-everything for last-5). On put_any specifically — the only task the
rule touches — consolidated fed everything (0.85) now matches failures-only (0.84),
and success-only memory is worthless everywhere (0.44–0.57).

**Reading:** in a three-drawer world, rejection is nearly as informative as praise
(two rejections identify the third drawer), so failure-only learners still catch up.
"Failures carry the signal" is robust to a success-confirmed property; it would only
flip where elimination is expensive (many options) — a 10-drawer variant is the
right test, and a cheap one. Kept `preference` in the default property set: it
makes put_any a real memory task (no-knowledge = 1/3) rather than a free one.

**Bugs fixed today from reading failures, not averages:** the "object is inside
drawer X" test used an 8 cm z-window with drawers 7 cm apart (an object in the open
bottom drawer also tested as inside the closed middle one → `pick -> not_here`);
and the naive planner's fixed default drawer (top) coincided with the fast drawer in
5/6 sampled worlds — no-knowledge choices now draw from the world's RNG.

## 2026-09-05 — Day 9: Phase 3 world in physics — first numbers, and two calibration gaps

`bench/sim/run.py --properties sticky,heavy,location,fast --kinds put,put_any,fetch`,
6 worlds × 30 episodes, change at 15 (random property type). 720 episodes.
Figure `docs/figures/bench_sim_p3_curves_2026-09-05.png`.

| memory | AUC | put | put_any | fetch (success / steps / wasted looks) |
|---|---|---|---|---|
| none | 0.68 | 0.54 | 1.00 | 0.47 / 652 / 0.97 |
| last-5 | 0.79 | 0.71 | 1.00 | 0.63 / 454 / 0.21 |
| retrieval | 0.82 | 0.79 | 1.00 | 0.65 / 427 / 0.16 |
| consolidated | 0.82 | 0.77 | 1.00 | 0.66 / 430 / 0.16 |

**What transferred:** location memory works in physics — a blind fetch averages one
wasted look and 652 steps; with memory 0.16 wasted looks and ~430 (video:
`outputs/sim_videos/fetch_side_by_side.mp4`, 1003 vs 361 steps on one world).
Memory is worth +14 to +19 points overall, +17 on put, +18 on fetch.

**Two calibration gaps the physics exposed:**
1. *put_any is trivial in physics* (100% for everyone, no memory needed). The fast
   drawer saves only ~12 of ~140 open steps — the pull is speed-limited by the
   controller command, not by drawer damping — and the free-choice budget has no
   secret in it. Needs either a physically larger shortcut (e.g. a drawer that is
   already ajar: skip the hook entirely) or a tighter put_any budget. In the abstract
   env the same task separates cleanly (0.71 vs 0.98) because the shortcut is 3 of 15.
2. *fetch tops out at ~65% even with memory.* Budget 760 vs known-location cost
   ~361: a jam on the hidden object's drawer (+350) survives, but any second failure
   does not, and post-change relocations force a fresh search. Worth a per-kind
   calibration matrix like the one that fixed `put`.

**Physics-only lessons:** searching a cabinet requires *closing* drawers behind you
(an open drawer's front blocks the handles below; the abstract sim cannot teach
this); a hidden object must sit in the *front* of its drawer or a straight lift hits
the handle bar above; grasps should break on *sustained* overload, not a momentary
wall bump. All three are in `bench/sim/skill_env.py`.

## 2026-09-04 — Day 8: Phase 3 begins — property library ×2, task kinds ×3, randomized change events

**Built (abstract env):** two new hidden-property types and two new task kinds,
same memory API. *Location*: one object starts inside a drawer; "bring X to the
table" tasks need `look_in` (6 steps; a wasted look is ~11 with the open). *Fast
drawer*: opens in 2 steps instead of 5 — the success-shaped secret X3 asked for; it
pays on "put X away in any drawer" tasks where the robot chooses. Change events now
flip a random property type; `--extra-changes N` adds more at random episodes.
Budgets are per task kind (nominal + slack 11), the physics lesson applied.
Old v0 configuration still reproduces (`--properties sticky,heavy --kinds put`).

**Result (30 worlds × 50 × 3 seeds), success | mean steps by task kind:**

| memory | put | put_any (free choice) | fetch |
|---|---|---|---|
| none | 0.48 / 23.9 | 0.71 / 20.9 | 0.41 / 30.6 |
| last-5 | 0.72 / 21.0 | 0.98 / 15.4 | 0.86 / 23.7 |
| retrieval | 0.82 / 20.0 | 0.97 / 15.3 | 0.91 / 22.9 |
| consolidated | 0.78 / 20.3 | 0.98 / 15.1 | 0.90 / 22.9 |

Free-choice tasks show the success-shaped secret being exploited (memories pick the
fast drawer: 15 steps, 98%). Fetch is where memory pays most (+50 points).

**X3 re-run with the success-shaped secret present:** failures-only is *still* best
(0.95 vs 0.90 fed everything); success-only rises only 0.54 → 0.60. Reason: a failed
episode contains every successful sub-step before the failure — including the cheap
open that reveals the fast drawer — so failures are the most information-dense
episodes, not merely the negative ones. "Store failures" = "store the episodes where
the most happened." Only a property revealed solely by a *fully* successful episode
could flip this; worth designing one for Phase 3's final library.

**Two bugs found by the per-kind breakdown:** fetch nominal cost was under-counted
(16 vs 21) so its budget was unfair, and the planner ignored "seen empty" evidence,
so a failed search never narrowed the next one. Per-kind breakdowns are now part of
every run's sanity check.

**Not yet in physics:** look_in and the fast drawer (bench/sim still has the v0 pair).

## 2026-09-04 — Day 7d: physics X2 at scale — where "never revising" actually costs

`bench/sim/run.py`, 10 worlds × 2 seeds × 30 episodes (600 episodes/memory), change
at 15, budget nominal+260. Figure `docs/figures/bench_sim_big_curves_2026-09-04.png`.

| memory | pre-change (ep 5–14, n=200) | post-change (ep 15–29, n=300) | stale/ep post |
|---|---|---|---|
| none | 0.53 [0.46, 0.59] | 0.52 [0.46, 0.58] | 0.05 |
| last-5 | 0.80 [0.74, 0.85] | 0.82 [0.78, 0.86] | 0.16 |
| retrieval | **0.87** [0.82, 0.91] | 0.83 [0.79, 0.87] | **0.29** |
| consolidated | 0.84 [0.78, 0.88] | **0.85** [0.81, 0.89] | 0.21 |

**Honest reading.** Memory vs none: ~30 points, unambiguous. Retrieval vs
consolidated on *success*: intervals overlap both before and after the change — not
separated in physics with the scripted planner. Where never-revising shows: retrieval
is the only memory whose success falls at the change event (0.87 → 0.83) while every
other memory's rises, and it takes ~40% more stale actions per episode afterwards
(0.29 vs 0.21). Success is a blunt instrument here because the planner recovers
in-episode and the budget tolerates one stale action; the sharp version of the same
effect is the LLM-reader run (Day 7b): retrieval 0.99 → 0.66.

**For public claims:** "the memory that never forgets was the only one that got
worse when the world changed, and took 40% more wasted actions; with a language
model reading it, it collapsed from 99% to 66%." Both sourced; neither overclaims.

**Also today:** ground-truth retrieval precision/recall added to the abstract runner
(last-5 0.13/0.33, retrieval 0.16/0.50 — fewer than one of five surfaced episodes
bears on the task); X3 (Day 7c) and the LLM-reader run (Day 7b).

## 2026-09-04 — Day 7c: X3 — failure memory is the whole signal (and it can't revise)

`bench/run_x3.py`: each memory fed all / success-only / failures-only episodes.
Abstract env, 30 worlds × 50 × 3 seeds.

| fed with | last-5 | retrieval | consolidated |
|---|---|---|---|
| everything | 0.80 | 0.86 | 0.85 |
| successes only | 0.54 (= none) | 0.54 | 0.54 |
| failures only | **0.95** | 0.95 | 0.92 |

Success-only memory is worthless here: every secret manifests as a failure, and a
jam is fatal within the budget, so successful episodes never carry the decisive
evidence. Failures-only is the best configuration tested, at 370 bytes vs 6,409 for
full retrieval — the last five *failures* span far more informative history than
the last five episodes. But failures-only cannot un-learn: revision needs a success
where a failure was expected (highest stale count, 6.6; consolidated drops 0.98 →
0.82 at the change vs 0.87 → 0.84 when fed everything).

**Design principle for the library:** store failures to learn; store the successes
that *contradict* a stored failure to revise. Everything else is noise.
**Caveat:** this world's secrets are all failure-shaped. A property that only shows
up on success (a shortcut) would flip the result; Phase 3's property library needs
at least one of those to keep the benchmark honest.

## 2026-09-04 — Day 7b: an LLM reads the memory — retrieval collapses 33 points at the change event

**Setup:** `bench/run_llm.py --backend anthropic` (claude-haiku-4-5), abstract env,
10 worlds × 50 episodes, change at 25. Memories now hand the planner *text*: raw
episode logs (last-5, retrieval) or fact sentences with evidence and age
(consolidated). The model plans a skill sequence and replans after failures.
2,604 calls, 1.49M tokens (~$1.50).

| memory | scripted reader AUC | Haiku AUC | Haiku pre → post-change | stale (post) |
|---|---|---|---|---|
| none | 0.54 | 0.52 | 0.49 → 0.53 | 0 |
| last-5 | 0.80 | 0.93 | 0.97 → 0.91 | 6.5 |
| retrieval (never forgets) | 0.86 | 0.87 | **0.99 → 0.66** | 8.0 |
| consolidated + probe | 0.85 | **0.95** | 1.00 → 0.90 | 4.7 |

**Findings.** (1) Harness validated: no-memory matches the scripted planner
(0.52 vs 0.54). (2) The "never revises" failure is now unmistakable with tight
intervals: retrieval is near-perfect while the world is static and loses a third of
its success the moment it changes. (3) Consolidated facts are the best-used
representation, and the model uses raw logs better than the rule-based reader — it
is cautious (robust skill when the context is ambiguous), which the budget tolerates
and which shows up as higher stale counts. That is why stale actions and steps are
reported next to success: the success metric alone rewards caution.

**The confound that nearly produced the opposite conclusion.** The first Haiku run
had none=0.22 and consolidated=0.52 ("LLMs can't use structured facts"). Tracing
showed the *executor* was at fault: 7/40 no-memory episodes discarded for
unparseable replies, and targets like "middle drawer" or a separate `drawer` key on
`place` routed to the wrong drawer. After robust parsing + target normalization the
model's decisions were correct every time. **Rule:** before any LLM-vs-memory claim,
the no-memory row must match the scripted planner; if not, the plumbing is broken.

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
