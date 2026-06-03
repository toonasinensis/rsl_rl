# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPDiscriminator, AMPPlugin, ExternalAMPProvider, PPO, PPOPlugin
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4


class DummyEnv(VecEnv):
    """Minimal VecEnv used to exercise PPO plugin hooks."""

    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 16
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
        del actions
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        return obs, rewards, dones, {"time_outs": torch.zeros(self.num_envs, device=self.device)}


class DummyAMPEnv(DummyEnv):
    """VecEnv variant that also exposes AMP observations."""

    def __init__(self, device: str = "cpu", amp_dim: int = 6) -> None:
        super().__init__(device=device)
        self.amp_dim = amp_dim

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
                "amp": torch.randn(self.num_envs, self.amp_dim, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )


class RecordingPlugin(PPOPlugin):
    """Small plugin that records lifecycle calls for tests."""

    def __init__(self, reward_bonus: float = 0.25) -> None:
        self.reward_bonus = reward_bonus
        self.init_calls = 0
        self.after_act_calls = 0
        self.after_step_calls = 0
        self.update_start_calls = 0
        self.per_batch_calls = 0
        self.after_backward_calls = 0
        self.post_backward_calls = 0
        self.post_update_calls = 0
        self.train_mode_calls = 0
        self.eval_mode_calls = 0
        self.loaded_marker = False
        self.last_obs_keys: list[str] = []
        self.hook_order: list[str] = []
        self.after_backward_saw_grads = False

    def on_init(self, ppo, env) -> None:
        self.init_calls += 1
        self.last_obs_keys = sorted(env.get_observations().keys())

    def on_after_act(self, runner, obs: TensorDict) -> None:
        del runner
        self.after_act_calls += 1
        self.last_obs_keys = sorted(obs.keys())

    def on_after_step(
        self,
        runner,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> torch.Tensor:
        del runner, obs, dones
        self.after_step_calls += 1
        extras["plugin_reward_bonus"] = torch.full_like(rewards, self.reward_bonus)
        return rewards + self.reward_bonus

    def on_update_start(self, ppo) -> None:
        del ppo
        self.update_start_calls += 1

    def on_per_batch_extra_loss(self, ppo, batch, forward_results=None) -> dict[str, torch.Tensor]:
        del batch, forward_results
        self.per_batch_calls += 1
        self.hook_order.append("per_batch_extra_loss")
        return {"recording_penalty": torch.zeros((), device=ppo.device)}

    def on_after_backward(self, ppo) -> None:
        self.after_backward_calls += 1
        self.hook_order.append("after_backward")
        self.after_backward_saw_grads = any(param.grad is not None for param in ppo.actor.parameters())

    def on_post_backward(self, ppo) -> None:
        del ppo
        self.post_backward_calls += 1
        self.hook_order.append("post_backward")

    def on_post_update(self, ppo) -> dict[str, float]:
        del ppo
        self.post_update_calls += 1
        return {"Plugin/recording_metric": 1.0}

    def on_train_mode(self, ppo) -> None:
        del ppo
        self.train_mode_calls += 1

    def on_eval_mode(self, ppo) -> None:
        del ppo
        self.eval_mode_calls += 1

    def on_save(self, ppo, saved_dict: dict) -> None:
        del ppo
        saved_dict["recording_plugin_saved"] = True

    def on_load(self, ppo, loaded_dict: dict) -> None:
        del ppo
        self.loaded_marker = bool(loaded_dict.get("recording_plugin_saved", False))


class FakeAMPDiscriminator:
    """Simple reward-decomposition stub for AMP plugin tests."""

    def predict_amp_reward_components(self, _state, _next_state, _task_reward, normalizer=None):
        assert normalizer is None
        components = {
            "task_reward": torch.tensor([1.0, 2.0]),
            "amp_reward": torch.tensor([10.0, 20.0]),
            "mixed_reward": torch.tensor([4.0, 8.0]),
        }
        return components, torch.zeros(2, 1)


def _make_train_cfg(plugin_key: str = "plugins") -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            plugin_key: [{"class_name": RecordingPlugin, "reward_bonus": 0.5}],
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }


def _make_amp_train_cfg() -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": "fixed",
            "plugins": [
                {
                    "class_name": AMPPlugin,
                    "loss_coef": 0.5,
                    "gradient_penalty_coef": 0.0,
                    "min_normalized_std": 0.2,
                    "provider": {
                        "class_name": ExternalAMPProvider,
                        "enable_policy_replay": True,
                        "expert_data": TensorDict(
                            {"amp": torch.randn(16, 6), "next_amp": torch.randn(16, 6)},
                            batch_size=[16],
                        ),
                    },
                    "discriminator": {
                        "class_name": AMPDiscriminator,
                        "hidden_dims": [16, 16],
                        "activation": "elu",
                    },
                }
            ],
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }


def _make_amp_cfg_entry() -> dict:
    return {
        "reward_scale": 0.75,
        "loss_coef": 0.5,
        "gradient_penalty_coef": 0.0,
        "min_normalized_std": 0.2,
        "enable_policy_replay": True,
        "expert_data": TensorDict(
            {"amp": torch.randn(16, 6), "next_amp": torch.randn(16, 6)},
            batch_size=[16],
        ),
        "discriminator": {
            "class_name": AMPDiscriminator,
            "hidden_dims": [16, 16],
            "activation": "elu",
        },
    }


def test_construct_algorithm_instantiates_plugins_and_runs_init() -> None:
    env = DummyEnv()
    cfg = _make_train_cfg()
    obs = env.get_observations()

    alg = PPO.construct_algorithm(obs, env, cfg, device="cpu")

    assert len(alg.plugins) == 1
    plugin = alg.plugins[0]
    assert isinstance(plugin, RecordingPlugin)
    assert plugin.init_calls == 1
    assert plugin.last_obs_keys == ["policy"]


def test_runner_invokes_rollout_update_and_checkpoint_plugin_hooks() -> None:
    runner = OnPolicyRunner(DummyEnv(), _make_train_cfg(), log_dir=None, device="cpu")
    plugin = runner.alg.plugins[0]

    runner.learn(num_learning_iterations=1)

    assert plugin.after_act_calls > 0
    assert plugin.after_step_calls > 0
    assert plugin.update_start_calls == 1
    assert plugin.per_batch_calls > 0
    assert plugin.after_backward_calls > 0
    assert plugin.post_backward_calls > 0
    assert plugin.post_update_calls == 1
    assert plugin.train_mode_calls >= 1
    assert plugin.after_backward_saw_grads
    assert "after_backward" in plugin.hook_order
    assert "post_backward" in plugin.hook_order
    assert plugin.hook_order.index("after_backward") < plugin.hook_order.index("post_backward")

    runner.get_inference_policy()
    assert plugin.eval_mode_calls >= 1

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "plugin_checkpoint.pt")
        runner.save(path)
        plugin.loaded_marker = False
        runner.load(path)
        assert plugin.loaded_marker


def test_legacy_aux_modules_are_mapped_to_plugins() -> None:
    runner = OnPolicyRunner(DummyEnv(), _make_train_cfg(plugin_key="aux_modules"), log_dir=None, device="cpu")
    assert len(runner.alg.plugins) == 1
    assert isinstance(runner.alg.plugins[0], RecordingPlugin)


def test_amp_plugin_skeleton_instantiates_with_external_provider() -> None:
    env = DummyEnv()
    cfg = {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "plugins": [
                {
                    "class_name": AMPPlugin,
                    "provider": {
                        "class_name": ExternalAMPProvider,
                        "expert_data": TensorDict(
                            {"amp": torch.randn(8, 6), "next_amp": torch.randn(8, 6)},
                            batch_size=[8],
                        ),
                    },
                }
            ],
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }

    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")

    assert len(runner.alg.plugins) == 1
    plugin = runner.alg.plugins[0]
    assert isinstance(plugin, AMPPlugin)
    assert isinstance(plugin.provider, ExternalAMPProvider)
    assert plugin.provider.device == "cpu"

    expert_s, expert_ns = plugin.provider.sample_expert_pairs(3)
    assert expert_s.shape == (3, 6)
    assert expert_ns.shape == (3, 6)


def test_construct_algorithm_instantiates_amp_plugin_from_amp_cfg() -> None:
    env = DummyAMPEnv()
    cfg = {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "schedule": "fixed",
            "amp_cfg": _make_amp_cfg_entry(),
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }

    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")

    assert len(runner.alg.plugins) == 1
    plugin = runner.alg.plugins[0]
    assert isinstance(plugin, AMPPlugin)
    assert plugin.reward_scale == 0.75
    assert plugin.loss_coef == 0.5
    assert plugin.min_std == 0.2
    assert isinstance(plugin.provider, ExternalAMPProvider)
    assert plugin.provider.enable_policy_replay
    assert plugin.discriminator is not None

    expert_s, expert_ns = plugin.provider.sample_expert_pairs(3)
    assert expert_s.shape == (3, 6)
    assert expert_ns.shape == (3, 6)


def test_construct_algorithm_rejects_duplicate_amp_plugin_and_amp_cfg() -> None:
    env = DummyAMPEnv()
    cfg = {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "plugins": [{"class_name": AMPPlugin}],
            "amp_cfg": _make_amp_cfg_entry(),
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }

    try:
        OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    except ValueError as exc:
        assert "algorithm.amp_cfg" in str(exc)
    else:
        raise AssertionError("Expected duplicate AMP configuration to raise ValueError.")


def test_amp_plugin_inserts_reward_components_as_step_metrics() -> None:
    plugin = AMPPlugin(
        provider=ExternalAMPProvider(enable_policy_replay=True, replay_buffer_size=8),
    )
    plugin.discriminator = FakeAMPDiscriminator()
    plugin._current_amp_obs = torch.tensor([[1.0, 1.5], [2.0, 2.5]])

    next_amp_obs = torch.tensor([[3.0, 3.5], [4.0, 4.5]])
    obs = TensorDict({"amp": next_amp_obs.clone()}, batch_size=[2])
    extras: dict = {}

    mixed_reward = plugin.on_after_step(
        runner=None,
        obs=obs,
        rewards=torch.tensor([0.5, 1.5]),
        dones=torch.tensor([0.0, 1.0]),
        extras=extras,
    )

    assert torch.equal(mixed_reward, torch.tensor([4.0, 8.0]))
    assert set(extras["step_metrics"]) == {"task_reward", "amp_reward", "mixed_reward"}
    assert torch.equal(extras["step_metrics"]["task_reward"], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(extras["step_metrics"]["amp_reward"], torch.tensor([[10.0], [20.0]]))
    assert torch.equal(extras["step_metrics"]["mixed_reward"], torch.tensor([[4.0], [8.0]]))

    provider = plugin.provider
    assert isinstance(provider, ExternalAMPProvider)
    assert torch.equal(provider.policy_states, torch.tensor([[1.0, 1.5], [2.0, 2.5]]))
    assert torch.equal(provider.policy_next_states, torch.tensor([[3.0, 3.5], [2.0, 2.5]]))


def test_amp_plugin_per_batch_extra_loss_backprops_into_discriminator() -> None:
    runner = OnPolicyRunner(DummyAMPEnv(), _make_amp_train_cfg(), log_dir=None, device="cpu")
    plugin = runner.alg.plugins[0]

    assert isinstance(plugin, AMPPlugin)
    assert plugin.discriminator is not None

    batch = SimpleNamespace(observations=runner.env.get_observations())
    plugin.on_update_start(runner.alg)
    extra_losses = plugin.on_per_batch_extra_loss(runner.alg, batch)

    assert set(extra_losses) == {"amp_loss"}

    runner.alg.optimizer.zero_grad()
    extra_losses["amp_loss"].backward()

    assert any(param.grad is not None for param in plugin.discriminator.parameters())

    metrics = plugin.on_post_update(runner.alg)
    assert "AMP/amp_loss" in metrics
    assert "AMP/policy_accuracy" in metrics
    assert "AMP/expert_accuracy" in metrics


def test_amp_plugin_update_changes_discriminator_and_reports_metrics() -> None:
    runner = OnPolicyRunner(DummyAMPEnv(), _make_amp_train_cfg(), log_dir=None, device="cpu")
    plugin = runner.alg.plugins[0]

    assert isinstance(plugin, AMPPlugin)
    assert plugin.discriminator is not None

    discriminator_before = {
        name: param.clone()
        for name, param in plugin.discriminator.named_parameters()
    }

    obs = runner.env.get_observations().to(runner.device)
    runner.alg.train_mode()
    with torch.inference_mode():
        for _ in range(runner.cfg["num_steps_per_env"]):
            actions = runner.alg.act(obs)
            plugin.on_after_act(runner, obs)
            obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
            obs = obs.to(runner.device)
            rewards = rewards.to(runner.device)
            dones = dones.to(runner.device)
            rewards = plugin.on_after_step(runner, obs, rewards, dones, extras)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        runner.alg.compute_returns(obs)

    loss_dict = runner.alg.update()

    changed = any(
        not torch.equal(discriminator_before[name], param)
        for name, param in plugin.discriminator.named_parameters()
    )
    assert changed
    assert "plugin_losses/plugin_0_AMPPlugin/amp_loss" in loss_dict
    assert "AMP/amp_loss" in loss_dict
    assert "AMP/policy_accuracy" in loss_dict


def test_amp_plugin_after_backward_clamps_policy_std() -> None:
    runner = OnPolicyRunner(DummyAMPEnv(), _make_amp_train_cfg(), log_dir=None, device="cpu")
    plugin = runner.alg.plugins[0]

    assert isinstance(plugin, AMPPlugin)
    distribution = runner.alg.actor.distribution
    assert hasattr(distribution, "std_param")

    with torch.no_grad():
        distribution.std_param.fill_(0.05)

    plugin.on_after_backward(runner.alg)

    assert torch.all(distribution.std_param >= 0.2)
