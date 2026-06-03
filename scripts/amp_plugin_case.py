"""Reference AMP training case built on the new plugin-driven framework.

This script demonstrates the recommended lightweight AMP path:

    PPO + AMPPlugin + ExternalAMPProvider

By default the configuration uses ``algorithm.amp_cfg``, which is translated
internally into an ``AMPPlugin`` plus an ``ExternalAMPProvider``. Use
``--explicit-plugin`` if you want to see the equivalent manual plugin block.

The toy environment exposes:

- ``"policy"`` observations for the actor and critic
- ``"amp"`` observations for AMP reward and discriminator training
- ``get_amp_expert_observations()`` and ``sample_amp_expert_observations()``
  so the runner can inject expert transitions automatically

Expert data follows the transition contract expected by the new AMP stack:

    {
        "amp": current_amp_obs,
        "next_amp": next_amp_obs,
    }
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


class ToyAMPVecEnv(VecEnv):
    """Small oscillator-tracking environment for AMP integration examples."""

    def __init__(
        self,
        num_envs: int = 128,
        max_episode_length: int = 240,
        device: str = "cpu",
        expert_pool_size: int = 4096,
    ) -> None:
        self.num_envs = num_envs
        self.num_actions = 1
        self.max_episode_length = max_episode_length
        self.device = device
        self.cfg = {
            "name": "ToyAMPVecEnv",
            "description": "Toy oscillator environment for AMPPlugin examples.",
        }

        self.dt = 0.05
        self.phase_velocity = 1.0
        self.action_scale = 1.5

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.position = torch.zeros(self.num_envs, device=self.device)
        self.velocity = torch.zeros(self.num_envs, device=self.device)

        self._expert_pool = self._build_expert_pool(expert_pool_size)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def get_observations(self) -> TensorDict:
        target_pos, target_vel = self._target_state(self.phase)
        policy_obs = torch.stack(
            (
                self.position,
                self.velocity,
                target_pos,
                target_vel,
                torch.sin(self.phase),
                torch.cos(self.phase),
            ),
            dim=-1,
        )
        amp_obs = self._amp_features(self.position, self.velocity, self.phase)
        return TensorDict(
            {"policy": policy_obs, "amp": amp_obs},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = torch.clamp(actions.view(self.num_envs), -1.0, 1.0)
        self.episode_length_buf += 1

        self.velocity = 0.92 * self.velocity + self.action_scale * self.dt * actions
        self.position = self.position + self.dt * self.velocity
        self.phase = self.phase + self.phase_velocity * self.dt

        target_pos, target_vel = self._target_state(self.phase)
        position_error = self.position - target_pos
        velocity_error = self.velocity - target_vel
        tracking_reward = torch.exp(-(2.5 * position_error.square() + 0.5 * velocity_error.square()))
        control_penalty = 0.01 * actions.square()
        rewards = tracking_reward - control_penalty

        dones = (self.episode_length_buf >= self.max_episode_length).float()
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)

        observations = self.get_observations()
        extras = {
            "time_outs": dones.clone(),
            "log": {
                "tracking_reward": tracking_reward.mean(),
                "control_penalty": control_penalty.mean(),
            },
        }
        return observations, rewards, dones, extras

    def get_amp_expert_observations(self) -> TensorDict:
        return self._expert_pool.clone()

    def sample_amp_expert_observations(self, batch_size: int) -> TensorDict:
        phase = 2.0 * math.pi * torch.rand(batch_size, device=self.device)
        next_phase = phase + self.phase_velocity * self.dt
        current_pos, current_vel = self._target_state(phase)
        next_pos, next_vel = self._target_state(next_phase)
        return TensorDict(
            {
                "amp": self._amp_features(current_pos, current_vel, phase),
                "next_amp": self._amp_features(next_pos, next_vel, next_phase),
            },
            batch_size=[batch_size],
            device=self.device,
        )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        phase = 2.0 * math.pi * torch.rand(env_ids.numel(), device=self.device)
        target_pos, target_vel = self._target_state(phase)
        self.phase[env_ids] = phase
        self.position[env_ids] = target_pos + 0.15 * torch.randn_like(target_pos)
        self.velocity[env_ids] = target_vel + 0.15 * torch.randn_like(target_vel)
        self.episode_length_buf[env_ids] = 0

    def _build_expert_pool(self, num_samples: int) -> TensorDict:
        phase = torch.linspace(0.0, 2.0 * math.pi, num_samples + 1, device=self.device)[:-1]
        next_phase = phase + self.phase_velocity * self.dt
        current_pos, current_vel = self._target_state(phase)
        next_pos, next_vel = self._target_state(next_phase)
        return TensorDict(
            {
                "amp": self._amp_features(current_pos, current_vel, phase),
                "next_amp": self._amp_features(next_pos, next_vel, next_phase),
            },
            batch_size=[num_samples],
            device=self.device,
        )

    @staticmethod
    def _target_state(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sin(phase), torch.cos(phase)

    @staticmethod
    def _amp_features(position: torch.Tensor, velocity: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                position,
                velocity,
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=-1,
        )


def build_train_cfg(use_explicit_plugin: bool = False) -> dict:
    """Return a PPO config that enables AMP through the new plugin framework."""
    amp_cfg = {
        "reward_scale": 0.5,
        "loss_coef": 1.0,
        "gradient_penalty_coef": 5.0,
        "min_normalized_std": 0.1,
        "enable_policy_replay": True,
        "replay_buffer_size": 8192,
        "discriminator": {
            "class_name": "AMPDiscriminator",
            "hidden_dims": [128, 128],
            "activation": "elu",
        },
    }

    algorithm_cfg: dict = {
        "class_name": "PPO",
        "num_learning_epochs": 3,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "use_clipped_value_loss": True,
        "schedule": "fixed",
        "desired_kl": 0.01,
        "normalize_advantage_per_mini_batch": False,
    }
    if use_explicit_plugin:
        algorithm_cfg["plugins"] = [
            {
                "class_name": "AMPPlugin",
                "reward_scale": amp_cfg["reward_scale"],
                "loss_coef": amp_cfg["loss_coef"],
                "gradient_penalty_coef": amp_cfg["gradient_penalty_coef"],
                "min_normalized_std": amp_cfg["min_normalized_std"],
                "provider": {
                    "class_name": "ExternalAMPProvider",
                    "enable_policy_replay": amp_cfg["enable_policy_replay"],
                    "replay_buffer_size": amp_cfg["replay_buffer_size"],
                },
                "discriminator": amp_cfg["discriminator"],
            }
        ]
    else:
        algorithm_cfg["amp_cfg"] = amp_cfg

    return {
        "num_steps_per_env": 24,
        "save_interval": 1000,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
            "amp": ["amp"],
        },
        "algorithm": algorithm_cfg,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def run_inference_rollout(runner: OnPolicyRunner, steps: int = 64) -> None:
    """Run a short deterministic rollout and print summary statistics."""
    policy = runner.get_inference_policy(device=runner.device)
    obs = runner.env.get_observations().to(runner.device)
    reward_sum = 0.0
    done_sum = 0.0

    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs)
            obs, rewards, dones, _extras = runner.env.step(actions.to(runner.env.device))
            obs = obs.to(runner.device)
            reward_sum += rewards.mean().item()
            done_sum += dones.mean().item()

    print(f"Inference rollout over {steps} steps:")
    print(f"  mean reward per step: {reward_sum / steps:.4f}")
    print(f"  mean done rate per step: {done_sum / steps:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5, help="Number of training iterations to run.")
    parser.add_argument("--num-envs", type=int, default=128, help="Number of vectorized environments.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--log-dir", type=str, default=None, help="Optional logging directory.")
    parser.add_argument(
        "--explicit-plugin",
        action="store_true",
        help="Use an explicit AMPPlugin block instead of algorithm.amp_cfg translation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    env = ToyAMPVecEnv(num_envs=args.num_envs, device=args.device)
    cfg = build_train_cfg(use_explicit_plugin=args.explicit_plugin)
    runner = OnPolicyRunner(env, cfg, log_dir=args.log_dir, device=args.device)

    mode = "explicit AMPPlugin config" if args.explicit_plugin else "algorithm.amp_cfg translation"
    print("Running AMP reference case with:")
    print(f"  mode: {mode}")
    print(f"  device: {args.device}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  iterations: {args.iterations}")
    print("  expert source wiring: env.get_amp_expert_observations + env.sample_amp_expert_observations")

    runner.learn(num_learning_iterations=args.iterations)
    run_inference_rollout(runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
