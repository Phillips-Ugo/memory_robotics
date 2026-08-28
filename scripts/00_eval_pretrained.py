"""Phase 0: run a pretrained policy in sim and produce a success-rate number.

Evaluates lerobot/diffusion_pusht (a diffusion policy trained on the PushT task:
push a T-shaped block onto a T-shaped target) for a few episodes and reports
success rate. Saves a video of the first episode to outputs/rollout_ep0.mp4.

Usage:
    uv run python scripts/00_eval_pretrained.py [--episodes 5] [--device mps|cpu]
"""

import argparse
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import numpy as np
import torch

try:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
except ImportError:  # older lerobot layout
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

REPO_ID = "lerobot/diffusion_pusht"


def load_norm_stats():
    """Pull normalization stats out of the old-format checkpoint.

    The lerobot/diffusion_pusht checkpoint predates lerobot 0.6's processor
    pipelines: its normalization buffers live inside model.safetensors under
    keys the new DiffusionPolicy no longer has, so from_pretrained drops them
    ("Unexpected key(s) when loading model"). Without them the policy sees raw
    pixel coordinates and returns actions stuck in [-1, 1] -> 0% success.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(REPO_ID, "model.safetensors"))
    return {
        "image_mean": sd["normalize_inputs.buffer_observation_image.mean"],
        "image_std": sd["normalize_inputs.buffer_observation_image.std"],
        "state_min": sd["normalize_inputs.buffer_observation_state.min"],
        "state_max": sd["normalize_inputs.buffer_observation_state.max"],
        "action_min": sd["unnormalize_outputs.buffer_action.min"],
        "action_max": sd["unnormalize_outputs.buffer_action.max"],
    }


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_episode(env, policy, stats, device, seed: int, record: bool):
    policy.reset()
    obs, info = env.reset(seed=seed)
    frames = [env.render()] if record else []
    rewards = []
    terminated = truncated = False
    while not (terminated or truncated):
        state = torch.from_numpy(obs["agent_pos"]).to(torch.float32)
        image = torch.from_numpy(obs["pixels"]).to(torch.float32) / 255
        image = image.permute(2, 0, 1)  # HWC -> CHW
        # MIN_MAX -> [-1, 1] for state, MEAN_STD for image (per policy config)
        state = 2 * (state - stats["state_min"]) / (stats["state_max"] - stats["state_min"]) - 1
        image = (image - stats["image_mean"]) / stats["image_std"]
        batch = {
            "observation.state": state.unsqueeze(0).to(device),
            "observation.image": image.unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            action = policy.select_action(batch).squeeze(0).cpu()
        # inverse MIN_MAX: [-1, 1] -> pixel coordinates
        action = (action + 1) / 2 * (stats["action_max"] - stats["action_min"]) + stats["action_min"]
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        rewards.append(reward)
        if record:
            frames.append(env.render())
    # PushT terminates only when target coverage exceeds the success threshold,
    # so termination == success; running out of steps truncates instead.
    return terminated, max(rewards), frames


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a success rate — honest error bars
    even at small n, unlike the naive +/- formula."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - half), min(1.0, center + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", choices=["off", "offline", "online"], default="offline",
                        help="experiment tracking; 'online' needs `wandb login` first")
    args = parser.parse_args()

    run = None
    if args.wandb != "off":
        import wandb
        run = wandb.init(project="memory-robotics", job_type="eval",
                         config=vars(args) | {"policy": "lerobot/diffusion_pusht"},
                         mode=args.wandb)

    device = pick_device(args.device)
    print(f"device: {device}")

    print(f"loading {REPO_ID} ...")
    policy = DiffusionPolicy.from_pretrained(REPO_ID)
    policy.to(device)
    policy.eval()
    stats = load_norm_stats()

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )

    successes, best_coverages = [], []
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    for ep in range(args.episodes):
        success, best_reward, frames = run_episode(
            env, policy, stats, device, seed=1000 + ep, record=(ep == 0)
        )
        successes.append(success)
        best_coverages.append(best_reward)
        print(f"episode {ep}: {'SUCCESS' if success else 'fail'} "
              f"(best reward {best_reward:.3f})")
        if run:
            run.log({"episode": ep, "success": int(success), "best_reward": best_reward,
                     "running_success_rate": float(np.mean(successes))})
        if frames:
            try:
                import imageio
                fps = env.metadata.get("render_fps", 10)
                imageio.mimsave(out_dir / "rollout_ep0.mp4", np.stack(frames), fps=fps)
                print(f"saved video: {out_dir / 'rollout_ep0.mp4'}")
            except Exception as e:
                print(f"(video save skipped: {e})")

    rate = float(np.mean(successes))
    lo, hi = wilson_interval(sum(successes), args.episodes)
    print(f"\nsuccess rate: {rate:.0%} over {args.episodes} episodes "
          f"(95% CI [{lo:.0%}, {hi:.0%}], mean best reward {np.mean(best_coverages):.3f})")
    if run:
        run.summary.update({"success_rate": rate, "ci_low": lo, "ci_high": hi,
                            "mean_best_reward": float(np.mean(best_coverages))})
        run.finish()


if __name__ == "__main__":
    main()
