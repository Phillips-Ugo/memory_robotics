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
   harness on a rented GPU. First reproduced number. **Code ready (2026-09-02):**
   `scripts/02_rma_pi05_adapter.py` (websocket adapter, load-tested locally) +
   `scripts/setup_gpu_box.sh` (Ubuntu 22.04 box setup + runbook). Remaining:
   rent the box, run it.
3. **M3 — keyframe-memory VLA inference (rescoped 2026-08-30):** run RoboMemArena's
   own shipped Qwen-VL + keyframe + π₀.₅ reference pipeline — it's the only
   VLM+keyframe baseline that exists as runnable code. Reproducing *official*
   MemER on this benchmark would mean writing the adapter ourselves around a
   repo with no sim support and a single-task checkpoint → demoted to a stretch
   goal after M4, if the X1 probes need it.
4. **M4 — probes (X1):** ablate the keyframe bank, add distractors, break it on
   purpose, per task category. The write-up material.
5. (Only if needed) fine-tuning — avoid until inference-only reproduction works.

## GPU budget (verified against openpi README, 2026-08-30)

| Workload | VRAM (openpi README) | Card to rent |
|---|---|---|
| π₀ / π₀.₅ inference | **> 8 GB** | single RTX 4090 (README's own example) |
| LoRA fine-tune | > 22.5 GB (tight on 24 GB, but README lists 4090) | RTX 4090 / A6000 |
| Full fine-tune | > 70 GB | A100-80GB / H100 — avoid in Phase 1 |

Much cheaper than the original ~24–48 GB guess: **M2 fits on the cheapest 4090
tier of RunPod/Lambda/Vast.ai.** Rent hourly, script the setup (everything
reproducible from this repo), never leave a box idle. openpi officially supports
Ubuntu 22.04 only — rent that image.

## Open questions to answer before renting

- [x] Exact sim dependencies of `evaluation_benchmark/` — answered by building it (M1):
  robosuite 1.4.1 + mujoco 2.3.7 + vendored libero_fork + gym 0.26.2
  (see `scripts/setup_rma_env.sh` for the full pinned list + macOS patches).
- [x] Does openpi publish a π₀.₅ checkpoint compatible with RoboMemArena's action
  space? **Yes — and RoboMemArena vendors the runtime.** Found 2026-08-30 in our
  clone: `third_party/openpi_minimal/scripts/serve_policy.py` has an `EnvMode.LIBERO`
  whose default is config `pi05_libero`, checkpoint `gs://openpi-assets/checkpoints/pi05_libero`.
  The eval side (`evaluation_benchmark/openpi_minimal_runtime/`) connects to it via a
  **websocket client** (`WebsocketClientPolicy.infer(element)`), with
  `robocerebra_adapter.py` doing the obs→π₀.₅ input mapping (flipud + squish-resize,
  quat→axis-angle state). So M2 = run `serve_policy.py --env LIBERO` on the GPU +
  the provided eval script as client. No integration code to write from scratch.
- [x] MemER eval configs: **the upstream MemER repo has none — sim eval isn't
  possible from it out of the box.** (Recon 2026-08-30, github.com/memer-policy/memer:
  high-level policy code only — Qwen-VL fine-tuning guide + inference via plain HF
  transformers, Python ≥3.11, lerobot 0.4.4; **no π₀.₅ code or checkpoint** (README
  defers to openpi); **no LIBERO/RoboMemArena/simulator references anywhere**;
  one HF checkpoint, `ajaysri/memer-dusting-qwen3vl-4b-step-1500`, for a single
  real-robot task. Inference VRAM not stated; paper says high-level runs ~1 Hz.
  Model-size inconsistency in the authors' own materials: project page says
  Qwen2.5-VL-3B, arXiv abstract says 7B, released checkpoint is Qwen3-VL-4B.)
  RoboMemArena's paper reports MemER at **27.3% avg TSR / 49.1% avg CSR** but does
  not say whether that was official code, and publishes no MemER adapter — the only
  adapters in its 1,472-file tree are the generic template and the openpi runtime.
  RoboMemArena's own reference pipeline
  (`openpi_minimal_runtime/eval_task1_qwen3_async_openpi_inference_vla_cam.py`,
  a Qwen3-VL LoRA planner + keyframe memory driving π₀.₅ over the websocket) is
  therefore the only runnable VLM+keyframe+VLA baseline that exists in code.
- MemER's actual interface (for the memory-layer template): high-level VLM emits
  JSON `{current_subtask: str, keyframe_positions: [int]}`. The low-level policy is
  conditioned by **language only** — `low_level_policy.act(obs, language_command=
  current_subtask)`; keyframes never reach the low level, they feed back into the
  VLM's own context. Keyframe memory = cluster candidate frames (1-D single-linkage,
  merge distance 5) and keep the **median frame per cluster** (dedup, not selection).
- [x] Actual VRAM numbers — from the openpi README (see updated GPU-budget table):
  inference > 8 GB, LoRA > 22.5 GB, full fine-tune > 70 GB. A single RTX 4090
  serves `pi05_libero` inference. MemER-side numbers: recon in progress (agent).

## M2 architecture (settled by the above)

GPU box runs `serve_policy.py` (JAX openpi, its own env) ⇄ websocket ⇄ eval harness
(robosuite/mujoco sim, our rma-venv-style env) on the same box. The client/server
split means the two dependency stacks never share a venv — same isolation trick we
already use locally. Sim CPU-only work stays cheap; only π₀.₅ needs the GPU.

Verified details (openpi README + docs/remote_inference.md + examples/libero):
- Server: `uv run scripts/serve_policy.py --env LIBERO` (default port 8000) —
  loads config `pi05_libero` from `gs://openpi-assets/checkpoints/pi05_libero`.
  JAX-based; Ubuntu 22.04 officially; docker compose route exists
  (`examples/libero/compose.yml`).
- Client element = exactly what RoboMemArena's `robocerebra_adapter.py` builds:
  `observation/image` + `observation/wrist_image` (uint8 224×224),
  `observation/state` = eef_pos(3) + quat→axis-angle(3) + gripper_qpos, `prompt`.
  `client.infer(element)["actions"]` → chunk of shape (action_horizon=10, 7);
  state goes unnormalized ("normalization handled on the server side").
- openpi's own LIBERO example replans every 5 steps; RoboMemArena protocol says
  every 10 — follow the benchmark's protocol for comparability with its table.
