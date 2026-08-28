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


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_episode(env, policy, device, seed: int, record: bool):
    policy.reset()
    obs, info = env.reset(seed=seed)
    frames = [env.render()] if record else []
    rewards = []
    terminated = truncated = False
    while not (terminated or truncated):
        state = torch.from_numpy(obs["agent_pos"]).to(torch.float32)
        image = torch.from_numpy(obs["pixels"]).to(torch.float32) / 255
        image = image.permute(2, 0, 1)  # HWC -> CHW
        batch = {
            "observation.state": state.unsqueeze(0).to(device),
            "observation.image": image.unsqueeze(0).to(device),
        }
        with torch.inference_mode():
            action = policy.select_action(batch)
        obs, reward, terminated, truncated, info = env.step(
            action.squeeze(0).cpu().numpy()
        )
        rewards.append(reward)
        if record:
            frames.append(env.render())
    # PushT terminates only when target coverage exceeds the success threshold,
    # so termination == success; running out of steps truncates instead.
    return terminated, max(rewards), frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    print("loading lerobot/diffusion_pusht ...")
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
    policy.to(device)
    policy.eval()

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
            env, policy, device, seed=1000 + ep, record=(ep == 0)
        )
        successes.append(success)
        best_coverages.append(best_reward)
        print(f"episode {ep}: {'SUCCESS' if success else 'fail'} "
              f"(best reward {best_reward:.3f})")
        if frames:
            try:
                import imageio
                fps = env.metadata.get("render_fps", 10)
                imageio.mimsave(out_dir / "rollout_ep0.mp4", np.stack(frames), fps=fps)
                print(f"saved video: {out_dir / 'rollout_ep0.mp4'}")
            except Exception as e:
                print(f"(video save skipped: {e})")

    rate = float(np.mean(successes))
    print(f"\nsuccess rate: {rate:.0%} over {args.episodes} episodes "
          f"(mean best reward {np.mean(best_coverages):.3f})")


if __name__ == "__main__":
    main()
