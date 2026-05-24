# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MoE training with the on-policy runner."""

from __future__ import annotations

import copy
import os
import tempfile

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MoEPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import MoEOnPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 50
NUM_EXPERTS = 3


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and rewards."""

    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert actions.shape == (self.num_envs, self.num_actions)
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return self.get_observations(), rewards, dones, extras


def _make_moe_train_cfg() -> dict:
    """Return a minimal config matching ``MoEPPO.construct_algorithm``."""
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "MoEPPO",
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
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "share_cnn_encoders": False,
        },
        "actor": {
            "class_name": "StochasticWrapper",
            "backbone": {
                "class_name": "BackboneMoE",
                "num_experts": NUM_EXPERTS,
                "top_k": 1,
                "hidden_dims": [32, 32],
                "expert_hidden_dims": [32, 32],
                "router_hidden_dims": [16],
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


def _build_runner(log_dir: str | None = None) -> MoEOnPolicyRunner:
    """Construct a MoE runner with a DummyEnv and minimal config."""
    return MoEOnPolicyRunner(DummyEnv(), _make_moe_train_cfg(), log_dir=log_dir, device="cpu")


class TestMoERunnerConstruction:
    """Tests for constructing the MoE runner and its components."""

    def test_moe_runner_config_example(self) -> None:
        """The runner config should expose the MoE-specific construction fields."""
        cfg = _make_moe_train_cfg()
        assert cfg["algorithm"]["class_name"] == "MoEPPO"
        assert cfg["actor"]["backbone"]["class_name"] == "BackboneMoE"
        assert cfg["actor"]["backbone"]["num_experts"] == NUM_EXPERTS
        assert "moe_runner" not in cfg

    def test_runner_creates_moe_algorithm(self) -> None:
        """Runner should instantiate MoEPPO with actor and critic models."""
        runner = _build_runner()
        assert isinstance(runner.alg, MoEPPO)
        assert runner.alg.actor is not None
        assert runner.alg.critic is not None

    def test_runner_initially_freezes_experts(self) -> None:
        """Experts should be frozen when the MoE runner is constructed with default settings."""
        runner = _build_runner()
        backbone = runner.alg.actor.backbone
        assert backbone is not None
        assert not any(param.requires_grad for name, param in backbone.named_parameters() if "expert_" in name)


class TestMoELearnLoop:
    """Tests that the MoE learn loop runs and updates the router."""

    def test_learn_runs_without_error(self) -> None:
        """A short learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_router_but_not_experts(self) -> None:
        """Router parameters should remain trainable while frozen expert parameters stay fixed."""
        runner = _build_runner()
        backbone = runner.alg.actor.backbone

        assert any(param.requires_grad for param in backbone.module_dict["router"].parameters())
        assert not any(param.requires_grad for name, param in backbone.named_parameters() if "expert_" in name)

        runner.learn(num_learning_iterations=2)

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestMoESaveLoad:
    """Tests for MoEPPO checkpoint save and load."""

    def test_save_contains_expected_state_dicts(self) -> None:
        """MoEPPO checkpoints should store actor, critic, and optimizer states."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "moe_checkpoint.pt")
            runner.save(path)
            data = torch.load(path, weights_only=False, map_location="cpu")

        assert "actor_state_dict" in data
        assert "critic_state_dict" in data
        assert "optimizer_state_dict" in data

    def test_load_restores_actor_and_critic_parameters(self) -> None:
        """Loading a checkpoint should restore actor and critic parameters exactly."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "moe_checkpoint.pt")
            runner.save(path)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())
            saved_critic = copy.deepcopy(runner.alg.critic.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(path)

        for key, param in runner.alg.actor.state_dict().items():
            assert torch.equal(saved_actor[key], param), f"Actor parameter '{key}' not restored after load"
        for key, param in runner.alg.critic.state_dict().items():
            assert torch.equal(saved_critic[key], param), f"Critic parameter '{key}' not restored after load"

    def test_load_restores_iteration(self) -> None:
        """Loading a checkpoint should restore the iteration counter."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "moe_checkpoint.pt")
            runner.save(path)
            saved_iter = runner.current_learning_iteration

            runner.learn(num_learning_iterations=2)
            assert runner.current_learning_iteration != saved_iter

            runner.load(path)
            assert runner.current_learning_iteration == saved_iter


class TestMoEExpertFreezing:
    """Tests for expert-freezing behavior in the MoE runner."""

    def test_freeze_experts_can_be_disabled(self) -> None:
        """The runner should allow expert training when freeze_experts is disabled."""
        cfg = _make_moe_train_cfg()
        cfg["moe_runner"] = {"freeze_experts": False}
        runner = MoEOnPolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
        backbone = runner.alg.actor.backbone
        assert any(param.requires_grad for name, param in backbone.named_parameters() if "expert_" in name)

    def test_freeze_experts_from_config(self) -> None:
        """The runner should freeze experts when configured through train_cfg."""
        cfg = _make_moe_train_cfg()
        cfg["moe_runner"] = {"freeze_experts": True}
        runner = MoEOnPolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
        backbone = runner.alg.actor.backbone
        assert not any(param.requires_grad for name, param in backbone.named_parameters() if "expert_" in name)
