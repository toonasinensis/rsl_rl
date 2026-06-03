# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMP training with the on-policy runner."""

from __future__ import annotations

import os
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO, AMPPlugin, ExternalAMPProvider
from rsl_rl.env import VecEnv
from rsl_rl.runners import AMPOnPolicyRunner, OnPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
AMP_DIM = 6
NUM_ACTIONS = 4
MAX_EP_LEN = 50


class DummyAMPEnv(VecEnv):
    """Minimal VecEnv with policy and AMP observations."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize buffers for a tiny AMP test environment."""
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        """Return random policy and AMP observations."""
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
                "amp": torch.randn(self.num_envs, AMP_DIM, device=self.device),
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

    def sample_amp_expert_observations(self, batch_size: int) -> TensorDict:
        """Sample random expert AMP observations."""
        return TensorDict(
            {"amp": torch.randn(batch_size, AMP_DIM, device=self.device) + 1.0},
            batch_size=[batch_size],
            device=self.device,
        )

    def get_amp_expert_observations(self) -> TensorDict:
        """Return a fixed pool of expert AMP observations."""
        return TensorDict(
            {"amp": torch.randn(24, AMP_DIM, device=self.device) + 2.0},
            batch_size=[24],
            device=self.device,
        )


class DummyAMPEnvNoSampler(DummyAMPEnv):
    """AMP env variant that exposes a fixed expert pool but no sampler callback."""

    sample_amp_expert_observations = None


def _make_amp_train_cfg() -> dict:
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
            "amp": ["amp"],
        },
        "algorithm": {
            "class_name": "AMPPPO",
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
            "amp_cfg": {
                "reward_scale": 0.25,
                "loss_coef": 1.0,
                "gradient_penalty_coef": 0.0,
                "discriminator": {
                    "class_name": "AMPDiscriminator",
                    "hidden_dims": [32, 32],
                    "activation": "elu",
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


def _build_runner(log_dir: str | None = None) -> AMPOnPolicyRunner:
    return AMPOnPolicyRunner(DummyAMPEnv(), _make_amp_train_cfg(), log_dir=log_dir, device="cpu")


def _make_plugin_amp_train_cfg() -> dict:
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
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
            "amp_cfg": {
                "reward_scale": 0.25,
                "loss_coef": 1.0,
                "gradient_penalty_coef": 0.0,
                "enable_policy_replay": True,
                "discriminator": {
                    "class_name": "AMPDiscriminator",
                    "hidden_dims": [32, 32],
                    "activation": "elu",
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


class TestAMPRunnerConstruction:
    """Tests for constructing the AMP runner and algorithm."""

    def test_amp_runner_config_example(self) -> None:
        """The runner config should expose AMP-specific construction fields."""
        cfg = _make_amp_train_cfg()
        assert cfg["algorithm"]["class_name"] == "AMPPPO"
        assert cfg["algorithm"]["amp_cfg"]["discriminator"]["class_name"] == "AMPDiscriminator"
        assert cfg["obs_groups"]["amp"] == ["amp"]

    def test_runner_creates_amp_algorithm(self) -> None:
        """Runner should instantiate AMPPPO and attach the expert sampler."""
        runner = _build_runner()
        assert isinstance(runner.alg, AMPPPO)
        assert runner.alg.amp_discriminator is not None
        assert runner.alg.amp_expert_sampler is not None
        assert runner.alg.amp_expert_data is not None

    def test_runner_injects_sources_into_plugin_driven_amp(self) -> None:
        """OnPolicyRunner should wire env expert data and sampler into AMPPlugin providers."""
        runner = OnPolicyRunner(DummyAMPEnv(), _make_plugin_amp_train_cfg(), log_dir=None, device="cpu")

        assert len(runner.alg.plugins) == 1
        plugin = runner.alg.plugins[0]
        assert isinstance(plugin, AMPPlugin)
        assert isinstance(plugin.provider, ExternalAMPProvider)
        assert plugin.provider.expert_data is not None
        assert plugin.provider.expert_sampler is not None

    def test_amp_runner_explicit_expert_data_overrides_env_pool(self) -> None:
        """amp_runner.expert_data should take precedence over env.get_amp_expert_observations()."""
        explicit_expert_data = TensorDict(
            {"amp": torch.full((12, AMP_DIM), 9.0)},
            batch_size=[12],
        )
        cfg = _make_plugin_amp_train_cfg()
        cfg["amp_runner"] = {"expert_data": explicit_expert_data}

        runner = OnPolicyRunner(DummyAMPEnvNoSampler(), cfg, log_dir=None, device="cpu")
        plugin = runner.alg.plugins[0]

        assert isinstance(plugin, AMPPlugin)
        assert isinstance(plugin.provider, ExternalAMPProvider)
        expert_s, _ = plugin.provider.sample_expert_pairs(4)
        assert torch.all(expert_s == 9.0)

    def test_amp_runner_rejects_explicit_amp_plugin_with_ampppo(self) -> None:
        """AMPPPO should reject configs that also explicitly add AMPPlugin."""
        cfg = _make_amp_train_cfg()
        cfg["algorithm"]["plugins"] = [{"class_name": AMPPlugin}]

        try:
            AMPOnPolicyRunner(DummyAMPEnv(), cfg, log_dir=None, device="cpu")
        except ValueError as exc:
            assert "AMPPPO" in str(exc)
        else:
            raise AssertionError("Expected AMPPPO + explicit AMPPlugin to raise ValueError.")


class TestAMPLearnLoop:
    """Tests that the AMP learn loop runs."""

    def test_learn_runs_without_error(self) -> None:
        """A short AMP learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed AMP iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestAMPSaveLoad:
    """Tests for AMPPPO checkpoint save and load."""

    def test_save_contains_amp_discriminator(self) -> None:
        """AMP checkpoints should include the discriminator state dict."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "amp_checkpoint.pt")
            runner.save(path)
            data = torch.load(path, weights_only=False, map_location="cpu")

        assert "actor_state_dict" in data
        assert "critic_state_dict" in data
        assert "amp_discriminator_state_dict" in data
