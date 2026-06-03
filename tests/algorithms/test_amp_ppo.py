# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPO with adversarial motion prior training."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO, AMPDiscriminator, AMPPlugin, ExternalAMPProvider
from rsl_rl.models import BackboneMLP, StochasticWrapper
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
AMP_DIM = 6
NUM_ACTIONS = 4


def _make_obs(num_envs: int = NUM_ENVS) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, OBS_DIM),
            "amp": torch.randn(num_envs, AMP_DIM),
        },
        batch_size=[num_envs],
    )


def _make_expert_data(num_samples: int = 64) -> TensorDict:
    return TensorDict({"amp": torch.randn(num_samples, AMP_DIM) + 1.0}, batch_size=[num_samples])


def _build_amp_ppo(**amp_overrides: object) -> tuple[AMPPPO, TensorDict]:
    obs = _make_obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"], "amp": ["amp"]}

    actor_backbone = BackboneMLP(obs, obs_groups, "actor", NUM_ACTIONS, hidden_dims=[32, 32], activation="elu")
    actor = StochasticWrapper(
        actor_backbone,
        NUM_ACTIONS,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    critic = BackboneMLP(obs, obs_groups, "critic", 1, hidden_dims=[32, 32], activation="elu")
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    discriminator = AMPDiscriminator(obs, obs_groups, "amp", hidden_dims=[32, 32], activation="elu")

    amp_cfg = {
        "expert_data": _make_expert_data(),
        "reward_scale": 0.5,
        "loss_coef": 1.0,
        "gradient_penalty_coef": 0.0,
    }
    amp_cfg.update(amp_overrides)

    ppo = AMPPPO(
        actor,
        critic,
        storage,
        discriminator,
        num_learning_epochs=2,
        num_mini_batches=2,
        schedule="fixed",
        learning_rate=1.0e-3,
        amp_cfg=amp_cfg,
    )
    return ppo, obs


class TestAMPReward:
    """Tests for AMP reward shaping."""

    def test_amp_ppo_constructs_internal_plugin_core(self) -> None:
        """Legacy AMPPPO should reuse an internal AMPPlugin core for convergence."""
        ppo, _obs = _build_amp_ppo()

        assert isinstance(ppo._amp_plugin_core, AMPPlugin)
        assert ppo._amp_plugin_core.discriminator is ppo.amp_discriminator
        assert isinstance(ppo._amp_plugin_core.provider, ExternalAMPProvider)
        assert ppo._amp_plugin_core.provider.expert_data is not None

    def test_legacy_amp_handles_delegate_to_internal_provider(self) -> None:
        """Legacy AMPPPO expert-data handles should reflect the internal provider state."""
        ppo, _obs = _build_amp_ppo(expert_data=None)
        expert_data = _make_expert_data(12)

        ppo.amp_expert_data = expert_data

        assert ppo._amp_plugin_core.provider.expert_data is not None
        assert ppo.amp_expert_data is not None
        assert ppo.amp_expert_data.batch_size[0] == 12

    def test_amp_ppo_reward_helper_delegates_to_internal_plugin_core(self) -> None:
        """Legacy AMPPPO reward computation should match the shared AMPPlugin core."""
        ppo, obs = _build_amp_ppo()

        amp_rewards = ppo.compute_amp_rewards(obs)
        plugin_rewards = ppo._amp_plugin_core.compute_reward_from_observations(obs)

        assert torch.allclose(amp_rewards, plugin_rewards)

    def test_process_env_step_adds_amp_reward(self) -> None:
        """Stored rollout rewards should include positive AMP rewards."""
        ppo, obs = _build_amp_ppo()
        ppo.act(obs)

        base_rewards = torch.ones(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        ppo.process_env_step(_make_obs(), base_rewards, dones, {"time_outs": torch.zeros(NUM_ENVS)})

        assert torch.all(ppo.storage.rewards[0, :, 0] > base_rewards)
        assert ppo.amp_rewards.shape == (NUM_ENVS,)


class TestAMPLoss:
    """Tests for discriminator loss computation."""

    def test_amp_loss_has_gradients_for_discriminator(self) -> None:
        """AMP discriminator loss should backpropagate into discriminator parameters."""
        ppo, obs = _build_amp_ppo()

        loss_dict = ppo._compute_amp_loss(obs)
        ppo.optimizer.zero_grad()
        loss_dict["amp_loss"].backward()

        assert "policy_loss" in loss_dict
        assert "expert_loss" in loss_dict
        assert any(param.grad is not None for param in ppo.amp_discriminator.parameters())

    def test_update_returns_amp_metrics_and_updates_discriminator(self) -> None:
        """Full PPO update should report AMP metrics and change discriminator parameters."""
        ppo, obs = _build_amp_ppo()
        discriminator_before = {name: param.clone() for name, param in ppo.amp_discriminator.named_parameters()}

        for _ in range(NUM_STEPS):
            ppo.act(obs)
            next_obs = _make_obs()
            ppo.process_env_step(next_obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
            obs = next_obs
        ppo.compute_returns(obs)

        loss_dict = ppo.update()

        changed = any(
            not torch.equal(discriminator_before[name], param)
            for name, param in ppo.amp_discriminator.named_parameters()
        )
        assert changed
        assert "amp_loss_dict/amp_loss" in loss_dict
        assert "amp_loss_dict/expert_loss" in loss_dict
        assert "AMP/amp_loss" in loss_dict
        assert "AMP/policy_accuracy" in loss_dict


class TestAMPSaveLoad:
    """Tests for AMPPPO checkpoint compatibility during convergence."""

    def test_save_and_load_preserve_legacy_and_plugin_amp_state(self) -> None:
        """AMPPPO should keep legacy checkpoint keys while persisting plugin/provider state."""
        ppo, _obs = _build_amp_ppo()

        saved_dict = ppo.save()

        assert "amp_discriminator_state_dict" in saved_dict
        assert "plugin_state_dicts" in saved_dict
        assert "plugin_extra_state" in saved_dict

        restored_ppo, _ = _build_amp_ppo(expert_data=None)
        restored_ppo.load(saved_dict, load_cfg=None, strict=True)

        assert restored_ppo.amp_expert_data is not None
        assert restored_ppo._amp_plugin_core.provider.expert_data is not None
        assert restored_ppo.amp_expert_data.batch_size == ppo.amp_expert_data.batch_size
