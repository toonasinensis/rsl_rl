# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DeparsePPO on-policy runner."""

from __future__ import annotations

import copy
import tempfile

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import DeparsePPO
from rsl_rl.env import VecEnv
from rsl_rl.models import StochasticWrapper
from rsl_rl.runners.deparse_policy_runner import OnPolicyRunner as DeparsePolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 50


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and rewards."""

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict(
            {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        assert actions.shape == (self.num_envs, self.num_actions)
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return self.get_observations(), rewards, dones, extras


def _make_deparse_train_cfg() -> dict:
    """Return a minimal config matching ``DeparsePPO.construct_algorithm``."""
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "DeparsePPO",
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
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
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


def _build_runner(log_dir: str | None = None) -> DeparsePolicyRunner:
    """Construct a DeparsePolicyRunner with a DummyEnv and minimal config."""
    return DeparsePolicyRunner(DummyEnv(), _make_deparse_train_cfg(), log_dir=log_dir, device="cpu")


class TestDeparseRunnerConstruction:
    """Tests for constructing the Deparse runner and algorithm."""

    def test_runner_creates_deparse_ppo(self) -> None:
        """Runner should instantiate DeparsePPO with actor and critic models."""
        runner = _build_runner()

        assert isinstance(runner.alg, DeparsePPO)
        assert isinstance(runner.alg.actor, StochasticWrapper)
        assert runner.alg.critic is not None
        assert runner.cfg["multi_gpu"] is None

    def test_runner_uses_separate_actor_and_critic_optimizers(self) -> None:
        """DeparsePPO should maintain separate optimizers for actor and local critic updates."""
        runner = _build_runner()
        actor_param_ids = {
            id(param)
            for group in runner.alg.actor_optimizer.param_groups
            for param in group["params"]
        }
        critic_param_ids = {
            id(param)
            for group in runner.alg.critic_optimizer.param_groups
            for param in group["params"]
        }

        assert runner.alg.actor_optimizer is not runner.alg.critic_optimizer
        assert actor_param_ids
        assert critic_param_ids
        assert actor_param_ids.isdisjoint(critic_param_ids)


class TestDeparseLearnLoop:
    """Tests that the Deparse learn loop runs and updates both model families."""

    def test_learn_runs_without_error(self) -> None:
        """A short DeparsePPO learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_actor_and_critic_parameters(self) -> None:
        """Actor and critic parameters should both change after learning."""
        runner = _build_runner()
        actor_before = {name: param.clone() for name, param in runner.alg.actor.named_parameters()}
        critic_before = {name: param.clone() for name, param in runner.alg.critic.named_parameters()}

        runner.learn(num_learning_iterations=2)

        actor_changed = any(
            not torch.equal(actor_before[name], param)
            for name, param in runner.alg.actor.named_parameters()
        )
        critic_changed = any(
            not torch.equal(critic_before[name], param) for name, param in runner.alg.critic.named_parameters()
        )
        assert actor_changed, "Actor parameters should have changed after learning"
        assert critic_changed, "Critic parameters should have changed after learning"

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestDeparseSaveLoad:
    """Tests for DeparsePPO checkpoint save and load."""

    def test_save_contains_split_optimizer_state(self) -> None:
        """DeparsePPO checkpoints should store actor and critic optimizer states separately."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            data = torch.load(f.name, weights_only=False, map_location="cpu")

        assert "actor_state_dict" in data
        assert "critic_state_dict" in data
        assert set(data["optimizer_state_dict"]) == {"actor", "critic"}

    def test_load_restores_actor_and_critic_parameters(self) -> None:
        """Loading a checkpoint should restore actor and critic parameters exactly."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())
            saved_critic = copy.deepcopy(runner.alg.critic.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

        for key, param in runner.alg.actor.state_dict().items():
            assert torch.equal(saved_actor[key], param), f"Actor parameter '{key}' not restored after load"
        for key, param in runner.alg.critic.state_dict().items():
            assert torch.equal(saved_critic[key], param), f"Critic parameter '{key}' not restored after load"


class TestDeparseInferencePolicy:
    """Tests for the Deparse inference policy returned by the runner."""

    def test_inference_policy_produces_action_dict(self) -> None:
        """The stochastic wrapper should return an action dictionary with the correct shape."""
        runner = _build_runner()
        policy = runner.get_inference_policy()
        output = policy(runner.env.get_observations())

        assert isinstance(output, dict)
        assert output["actions"].shape == (NUM_ENVS, NUM_ACTIONS)


class TestDeparseUnsupportedConfigs:
    """Tests for config validation specific to DeparsePPO."""

    def test_share_cnn_encoders_is_rejected(self) -> None:
        """DeparsePPO should reject shared encoders because critics are meant to train locally."""
        cfg = _make_deparse_train_cfg()
        cfg["algorithm"]["share_cnn_encoders"] = True

        try:
            DeparsePolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
        except ValueError as exc:
            assert "Share CNN encoders" in str(exc)
        else:
            raise AssertionError("Expected shared CNN encoders to be rejected")
