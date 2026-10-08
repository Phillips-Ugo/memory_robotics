# A System-One model for robots ("Jev-R") — design, training plan, evals

*Draft 2026-09-29. Companion research: `docs/research/jev-brief.md`, `docs/research/fast-decision-models-survey.md`.*

## 0. The idea in one paragraph

Jev's bet is that most of what software needs from a model is a **typed decision**, not text: state in, calibrated
probabilities out, in ~100 ms, for pennies — and that you get there by *removing generation*, not by shrinking the
model. A VLA already generates the expensive thing (actions). What sits around it today is either nothing (π0.5 on
RoboMemArena does not know when a subtask is done — every M2b failure) or a slow VLM asked to write paragraphs about
the scene (RoboMonitor: 4B params, 200 ms). **Jev-R is the missing System One layer for robots:** a small,
non-generative model that answers typed questions about the robot's situation — *is this stage done? which of these
action chunks is best? should we interrupt / re-prompt? did the grasp hold?* — from images + proprioception + the
policy's own proposal, in <20 ms, with calibrated confidence, every control step. Memory is not baked in; the memory
layer (memlayer) *asks it questions* and acts on the answers, which is exactly how X5 already works with an oracle.

## 1. What we take from Jev (facts, not guesses)

| Jev property (public) | Jev-R translation |
|---|---|
| Non-generative: output tokens = 0, "sacrifice text generation for composability" | No decoder, no language output. Heads only. |
| State ingested once; questions evaluated in parallel and in isolation; adding questions is ~free | One shared encoding of the observation per control step; N question heads read it in one batched pass. |
| Typed primitives: Noul (yes/no probability), Choice (≤255 options, probabilities + confidence), Score (2–10 rubric levels) | `noul` = sigmoid head; `choice` = list-wise scorer over K candidates (action chunks, subtasks, prompts); `score` = ordinal head (progress 0–10). |
| Confidence = concentration of the output distribution; route by threshold | Same statistic; plus conformal thresholds calibrated at a fixed false-alarm rate (the robotics standard). |
| RLCD: post-training that rewards honest probabilities | Calibration is a first-class training objective (proper scoring rules + temperature per question type + conformal), evaluated with ECE/AUCPDT, not just accuracy. |
| Synthetic-data-only training | Robot labels are nearly free: demo subtask boundaries, gripper events, hindsight rollout labels, RMSE-ranked perturbed actions (RoboMonkey), counterfactual instructions (RoboReward). |
| Latency independent of number of options (no autoregression) | Choice over 16 candidate chunks costs one forward pass of a small chunk-encoder, not 16 VLM calls. |

Not copied: Jev is text-only and its architecture is private. We do not need to reverse-engineer it; the survey shows
the robotics literature has converged on the same shape independently (SAFE probes on policy features, SparkVLA's
"Stop" head, CheckVLA's action-conditioned monitor, RoboMonkey's verifier). Our contribution is to make it *one*
typed, fast, calibrated interface and to benchmark it across tasks.

## 2. Architecture

```
                 ┌────────────── shared state encoder (frozen or LoRA) ──────────────┐
 agent cam ─┐    │ ViT-S/14 (DINOv2/v3 or SigLIP) per view  → 2×384                   │
 wrist cam ─┼──▶ │ proprio MLP (ee pose, gripper) → 64                                │──▶ z_t (≈1k dims)
 proprio ───┘    │ instruction / subtask id → fixed embedding (no LLM in the loop)     │
                 │ optional: the policy's own hidden features (SAFE) — free           │
                 └──────────────────────────────────────────────────────────────────────┘
                                   │  temporal window: [z_{t-k..t}] → tiny GRU/attention (hidden 64–128)
                                   ▼
        ┌──── question heads, batched, isolated ─────────────────────────────────────────┐
        │ noul   "stage k done?"  "grasp holding?"  "off-course?"        → sigmoid          │
        │ choice "which stage are we in?" / "best of these N chunks?"  → list-wise scorer   │
        │        (chunk encoder: 1-D conv over the (H,7) action chunk → 64, cross-attends z) │
        │ score  "task progress 0–10"                                  → ordinal head       │
        └───────────────────────────────────────────────────────────────────────────────┘
 outputs: probabilities + confidence per question, conformal decision at a fixed false-alarm rate
```

- **Size / latency budget.** ViT-S encoder ≈ 22M params/view, everything else <5M. Measured today: DINOv2-S 6.7 ms
  per 224 px frame on a MacBook GPU; ~1 ms amortized on a 4090. Target: <20 ms for two views + N=16 chunk candidates
  on a 4090, <10 ms with the policy's features reused.
- **Why not a VLM.** RoboMonitor (4B) is the accuracy reference at 180–200 ms; FailBench shows fine-tuning VLMs on
  small failure sets can *hurt*; TurboVLA shows a 0.2B stack with no LLM hits 97.6 % LIBERO at 31 ms. Language enters
  as a per-subtask embedding, not as tokens to decode.
- **Composability.** Adding a question = adding a head; the encoder does not change. Questions are answered in
  isolation (no cross-talk), matching Jev's contract and making per-question calibration possible.

## 3. Training

**Stage A — supervised heads on free labels (this week, laptop-scale).**
- Labels from RoboMemArena's per-subtask HDF5 demos: every frame of `<primitive>_k_*` is "stages <k done, k in
  progress"; the last frames of file k are "k done"; `gripper_states` gives grasp/release; concatenation by (task, seed)
  gives progress ∈ [0,1] per subtask (perceptual-change weighting, boundary jitter as in SparkVLA/ProcVLM).
- Labels from *our own policy rollouts* (harness `results.json` stage steps): hindsight stage labels on the
  policy's actual visual distribution, including failures. 204 episodes / 392k frames exist today (task 1).
- Action-choice labels: perturb demo chunks, rank by RMSE to the demo (RoboMonkey, 20M pairs from 45k episodes —
  no humans). Interrupt labels: FIPER/SAFE style — calibrated on successful rollouts only, no failure labels needed.
- Loss: BCE / list-wise softmax / ordinal CE, with **proper-scoring calibration** (Brier term), per-type
  temperature, then conformal thresholds on a held-out calibration split.

**Stage B — calibrated decisions (RLCD in spirit).** Fine-tune the heads (and optionally LoRA on the encoder) with a
reward on decision *quality under the deployment rule*: a stage-done head is rewarded for firing within Δ steps of the
true boundary and penalized per early fire; a chunk-chooser is rewarded by the downstream success of the chosen chunk in
sim (cheap: LIBERO/RoboMemArena are simulators). This is where "fast, calibrated" beats "accurate per frame".

**Stage C — multi-task, cross-policy.** Train once over RoboMemArena's 26 tasks (151 subtasks, 15k segments) and
rollouts from π0.5, OpenVLA-OFT and SmolVLA; evaluate on unseen tasks and unseen policies (SAFE/Hide-and-Seek protocol).

## 4. Evaluation plan (benchmarks we can actually run)

| Question | Benchmark / data | Metric | Baselines |
|---|---|---|---|
| stage done / current stage | RoboMemArena task 1 (ours), then 26 tasks; RoboProcessBench phase QA | frame acc, AUROC, **detection delay at 5 % false-alarm**, spurious-switch rate | harness oracle (upper bound), RoboMonitor (4B), GVL/VLM zero-shot |
| closed loop | X5 on RoboMemArena task 1 with **Jev-R replacing the oracle stage checks** | TSR/CSR vs oracle-gated memory arm (30/51) | fixed prompt 15/51, oracle-gated 30/51 |
| off-course / interrupt | SAFE protocol on LIBERO-Long + our rollouts | AUCPDT, timely recall at 5 % FA | SAFE probe, FIPER, VLA-FAIL |
| best-of-N chunk | LIBERO-Long, SimplerEnv | success vs N, latency per decision | MG-Select (verifier-free), RoboMonkey verifier, V-GPS |
| progress score | ProcVQA, RoboMemArena demos | VOC, MAE | ProcVLM (2B), Gemini Robotics-ER 2 (57 % 5-bin) |
| calibration | all of the above | ECE, reliability plots, per-type temperature | Jev's published ECE (0.03 in-domain, 0.11 OOD) as the bar |
| cost | all | ms/decision, $/1k decisions on a 4090 | RoboMonitor 200 ms; RoboMonkey 650 ms/16 chunks |

## 5. v0 result (today, Mac only, no GPU rental)

Frozen DINOv2-S agent-view features (6.7 ms/frame), logistic heads, 39k frames from 204 task-1 episodes.
Train on seed set A (fixed prompt) → test on seed set B (new seeds, three prompting strategies):

| question | frame acc | AUROC | operating point | delay (median / p90) | missed | early/false fires |
|---|---|---|---|---|---|---|
| stage 1 done | 0.972 | 0.996 | p≥0.95 for 5 frames (0.5 s) | 30 / 80 steps | 0/153 | 2/153 |
| stage 2 done | 0.966 | 0.941 | p≥0.5 for 3 frames | 20 / 60 steps | 25/72 | 1/153 |
| current stage (3-way) | 0.939 | — | — | — | — | — |

Reading: a single-view linear probe already replaces the oracle for stage 1 (fires 3 s after the true event, almost
never early). Stage 2 is under-detected: the tomato can inside the basket is small in the agent view and the episode
ends 200 steps after completion. Next: wrist view (in progress), temporal head, DINOv2-B, then the closed-loop X5 run.

## 6. Roadmap and resources

1. **This week (no GPU):** wrist+agent features, temporal head, conformal operating points, write-up as a Day 15 entry.
2. **GPU session 1 (~$3):** closed-loop X5 with Jev-R replacing the oracle stage checks; the number that matters.
3. **GPU session 2 (~$10):** train on RoboMemArena demos for tasks 1–3 (+ our task-2 policy once trained), add the
   chunk-chooser head with RoboMonkey-style labels, report best-of-N on LIBERO-Long.
4. **Then:** multi-task over 26 tasks, cross-policy transfer, RLCD-style calibration fine-tune, a public leaderboard
   page ("decision latency vs. detection delay"), and a short paper.

What would speed this up: a steady ~$30/week GPU budget; 1–2 more hands for the label pipelines (HDF5 → labels) and
the LIBERO best-of-N harness; and, if TypeSafe's invite lands, Jev itself as the text-side judge over memlayer's facts.
