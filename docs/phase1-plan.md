# Phase 1 plan: reproduce the frontier on Problem A

*Recon done 2026-08-28. Everything here is a plan to verify, not a promise.*

## What the pieces actually are (from repo recon)

- **RoboMemArena repo** (github.com/OpenHelix-Team/RoboMemArena): the dataset
  (keyframe-annotated HDF5 episodes: agentview RGB, wrist cam, EE/gripper/joint
  states) + an evaluation harness (`evaluation_benchmark/` with BDDL task
  definitions 1–26 — BDDL is LIBERO's task-spec language, so the sim stack is
  LIBERO-style robosuite/MuJoCo) + a PrediMem add-on folder. **Baseline training
  is NOT in the repo** — it points to external OpenPI for the low-level policy.
- **OpenPI** (Physical Intelligence): π₀ / π₀.₅ checkpoints + fine-tuning code.
- **MemER repo** (github.com/memer-policy/memer): high-level policy =
  Qwen2.5-VL-3B-Instruct, low-level = π₀.₅.
- Key affordance: RoboMemArena evaluates any policy through a **generic adapter
  interface**. That's also the plug point for a memory layer later.

## Milestone ladder (each one is shippable progress)

1. **M1 — harness first, no VLA:** get the RoboMemArena eval harness running with
   a dummy/random policy through the adapter interface. Proves the sim stack +
   adapter contract before any big model is involved. (Can start on the Mac;
   MuJoCo runs on CPU.)
2. **M2 — π₀.₅ inference:** run the pretrained reactive baseline through the
   harness on a rented GPU. First reproduced number.
3. **M3 — MemER inference:** same, with the keyframe mechanism. Second number.
4. **M4 — probes (X1):** ablate the keyframe bank, add distractors, break it on
   purpose, per task category. The write-up material.
5. (Only if needed) fine-tuning — avoid until inference-only reproduction works.

## GPU budget (estimates — verify against openpi docs before renting)

| Workload | VRAM guess | Card to rent |
|---|---|---|
| π₀.₅ / MemER inference + eval | ~24–48 GB | RTX 4090 / L40S / A6000 |
| LoRA fine-tune (π₀.₅ or Qwen2.5-VL-3B) | ~48–80 GB | A100 80GB |
| Full fine-tune | multi-GPU | avoid in Phase 1 |

Providers to compare: RunPod, Lambda, Vast.ai. Rent hourly, script the setup
(everything reproducible from this repo), never leave a box idle.

## Open questions to answer before renting

- [ ] Exact sim dependencies of `evaluation_benchmark/` (LIBERO? robosuite version?)
- [ ] Does openpi publish a π₀.₅ checkpoint compatible with RoboMemArena's action space?
- [ ] MemER repo: does it include eval-on-RoboMemArena configs or only their real-robot setup?
- [ ] Actual VRAM numbers from openpi/MemER READMEs
