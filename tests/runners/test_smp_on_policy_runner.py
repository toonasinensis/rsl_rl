# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SMP training with the on-policy runner."""

from __future__ import annotations

import os
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import SMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import SMPOnPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
SMP_DIM = 6
NUM_ACTIONS = 4
MAX_EP_LEN = 50


class DummySMPEnv(VecEnv):
    """Minimal VecEnv with policy and SMP observations."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize buffers for a tiny SMP test environment."""
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        """Return random policy and SMP observations."""
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
                "smp": torch.randn(self.num_envs, SMP_DIM, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the dummy environment with random rewards."""
        assert actions.shape == (self.num_envs, self.num_actions)
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return self.get_observations(), rewards, dones, extras

    def sample_smp_expert_observations(self, batch_size: int) -> TensorDict:
        """Sample random expert SMP observations."""
        return TensorDict(
            {"smp": torch.randn(batch_size, SMP_DIM, device=self.device) + 1.0},
            batch_size=[batch_size],
            device=self.device,
        )


def _make_smp_train_cfg() -> dict:
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
            "smp": ["smp"],
        },
        "algorithm": {
            "class_name": "SMPPPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "share_cnn_encoders": False,
            "smp_cfg": {
                "reward_scale": 0.25,
                "reward_temperature": 1.0,
                "loss_coef": 1.0,
                "reward_timestep": 3,
                "model": {
                    "class_name": "SMPDiffusionModel",
                    "hidden_dims": [32, 32],
                    "activation": "elu",
                    "num_diffusion_steps": 8,
                },
            },
        },
        "actor": {
            "class_name": "StochasticWrapper",
            "backbone": {
                "class_name": "BackboneMLP",
                "hidden_dims": [32, 32],
                "activation": "elu",
                "obs_normalization": True,
            },
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "BackboneMLP",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def _build_runner(log_dir: str | None = None) -> SMPOnPolicyRunner:
    return SMPOnPolicyRunner(DummySMPEnv(), _make_smp_train_cfg(), log_dir=log_dir, device="cpu")


class TestSMPRunnerConstruction:
    """Tests for constructing the SMP runner and algorithm."""

    def test_smp_runner_config_example(self) -> None:
        """The runner config should expose SMP-specific construction fields."""
        cfg = _make_smp_train_cfg()
        assert cfg["algorithm"]["class_name"] == "SMPPPO"
        assert cfg["algorithm"]["smp_cfg"]["model"]["class_name"] == "SMPDiffusionModel"
        assert cfg["obs_groups"]["smp"] == ["smp"]

    def test_runner_creates_smp_algorithm(self) -> None:
        """Runner should instantiate SMPPPO and attach the expert sampler."""
        runner = _build_runner()
        assert isinstance(runner.alg, SMPPPO)
        assert runner.alg.smp_model is not None
        assert runner.alg.smp_expert_sampler is not None


class TestSMPLearnLoop:
    """Tests that the SMP learn loop runs."""

    def test_learn_runs_without_error(self) -> None:
        """A short SMP learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed SMP iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestSMPSaveLoad:
    """Tests for SMPPPO checkpoint save and load."""

    def test_save_contains_smp_model(self) -> None:
        """SMP checkpoints should include the diffusion model state dict."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smp_checkpoint.pt")
            runner.save(path)
            data = torch.load(path, weights_only=False, map_location="cpu")

        assert "actor_state_dict" in data
        assert "critic_state_dict" in data
        assert "smp_model_state_dict" in data
