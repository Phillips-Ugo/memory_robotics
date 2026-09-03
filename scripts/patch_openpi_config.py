"""Register a LoRA fine-tuning config for RoboMemArena data in openpi's config.py.

openpi keeps train configs in a Python list inside src/openpi/training/config.py;
there is no plugin mechanism, so we insert ours before the registry dict is built.
Idempotent. Run from the repo root on the GPU box after setup_gpu_box.sh:

    python scripts/patch_openpi_config.py --repo-id belu/rma_task1

Config `pi05_rma_lora`:
  - π₀.₅ architecture, LoRA on both PaliGemma (gemma_2b_lora) and the action expert
    (gemma_300m_lora), freeze filter from the model config, EMA off — openpi's own
    low-memory recipe (pi0_libero_low_mem_finetune), applied to pi05.
  - init from pi05_libero (closer to our sim than pi05_base; swap via --init).
  - data = LeRobotLiberoDataConfig on our converted dataset, prompt_from_task=True,
    extra_delta_transform=False (matches pi05_libero / RoboMemArena's baseline).
  - batch 16, 8k steps by default: sized for a single 24–48 GB card, not the
    paper's batch-128 × 40k full fine-tune.
"""

from __future__ import annotations

import argparse
import pathlib

MARKER = "# --- memory_robotics: pi05_rma_lora ---"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vendor/openpi/src/openpi/training/config.py")
    ap.add_argument("--repo-id", default="belu/rma_task1")
    ap.add_argument("--init", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args()

    p = pathlib.Path(args.config)
    src = p.read_text()
    if MARKER in src:
        src = src.split(MARKER)[0] + src.split(MARKER)[2]  # drop the old block, re-insert below
    anchor = "_CONFIGS_DICT = {config.name: config for config in _CONFIGS}"
    assert anchor in src, "openpi config.py layout changed; patch by hand"

    block = f'''{MARKER}
_CONFIGS.append(
    TrainConfig(
        name="pi05_rma_lora",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="{args.repo_id}",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("{args.init}"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size={args.batch_size},
        num_train_steps={args.steps},
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=5e-5, decay_steps={args.steps}, decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=1000,
        keep_period=2000,
    )
)
{MARKER}
'''
    src = src.replace(anchor, block + anchor)
    p.write_text(src)
    print(f"patched {p}: pi05_rma_lora -> repo_id={args.repo_id}, init={args.init}, "
          f"batch={args.batch_size}, steps={args.steps}")


if __name__ == "__main__":
    main()
