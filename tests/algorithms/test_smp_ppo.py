# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPO with score matching prior training."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import SMPPPO, SMPDiffusionModel
from rsl_rl.models import BackboneMLP, StochasticWrapper
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
SMP_DIM = 6
NUM_ACTIONS = 4


def _make_obs(num_envs: int = NUM_ENVS) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, OBS_DIM),
            "smp": torch.randn(num_envs, SMP_DIM),
        },
        batch_size=[num_envs],
    )


def _make_expert_data(num_samples: int = 64) -> TensorDict:
    return TensorDict({"smp": torch.randn(num_samples, SMP_DIM) + 1.0}, batch_size=[num_samples])


def _build_smp_ppo(**smp_overrides: object) -> tuple[SMPPPO, TensorDict]:
    obs = _make_obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"], "smp": ["smp"]}

    actor_backbone = BackboneMLP(obs, obs_groups, "actor", NUM_ACTIONS, hidden_dims=[32, 32], activation="elu")
    actor = StochasticWrapper(
        actor_backbone,
        NUM_ACTIONS,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    critic = BackboneMLP(obs, obs_groups, "critic", 1, hidden_dims=[32, 32], activation="elu")
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    smp_model = SMPDiffusionModel(obs, obs_groups, "smp", hidden_dims=[32, 32], num_diffusion_steps=8)

    smp_cfg = {
        "expert_data": _make_expert_data(),
        "reward_scale": 0.5,
        "reward_temperature": 1.0,
        "loss_coef": 1.0,
        "reward_timestep": 3,
    }
    smp_cfg.update(smp_overrides)

    ppo = SMPPPO(
        actor,
        critic,
        storage,
        smp_model,
        num_learning_epochs=2,
        num_mini_batches=2,
        schedule="fixed",
        learning_rate=1.0e-3,
        smp_cfg=smp_cfg,
    )
    return ppo, obs


class TestSMPReward:
    """Tests for SMP reward shaping."""

    def test_process_env_step_adds_smp_reward(self) -> None:
        """Stored rollout rewards should include bounded positive SMP rewards."""
        ppo, obs = _build_smp_ppo()
        ppo.act(obs)

        base_rewards = torch.ones(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        ppo.process_env_step(_make_obs(), base_rewards, dones, {"time_outs": torch.zeros(NUM_ENVS)})

        assert torch.all(ppo.storage.rewards[0, :, 0] > base_rewards)
        assert torch.all(ppo.smp_rewards <= ppo.smp_reward_scale)
        assert ppo.smp_rewards.shape == (NUM_ENVS,)


class TestSMPLoss:
    """Tests for diffusion prior loss computation."""

    def test_smp_loss_has_gradients_for_diffusion_model(self) -> None:
        """SMP loss should backpropagate into diffusion model parameters."""
        ppo, obs = _build_smp_ppo()

        loss_dict = ppo._compute_smp_loss(obs)
        ppo.optimizer.zero_grad()
        loss_dict["smp_loss"].backward()

        assert "denoising_error" in loss_dict
        assert any(param.grad is not None for param in ppo.smp_model.parameters())

    def test_update_returns_smp_metrics_and_updates_diffusion_model(self) -> None:
        """Full PPO update should report SMP metrics and change diffusion model parameters."""
        ppo, obs = _build_smp_ppo()
        model_before = {name: param.clone() for name, param in ppo.smp_model.named_parameters()}

        for _ in range(NUM_STEPS):
            ppo.act(obs)
            next_obs = _make_obs()
            ppo.process_env_step(next_obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
            obs = next_obs
        ppo.compute_returns(obs)

        loss_dict = ppo.update()

        changed = any(not torch.equal(model_before[name], param) for name, param in ppo.smp_model.named_parameters())
        assert changed
        assert "smp_loss_dict/smp_loss" in loss_dict
        assert "smp_loss_dict/denoising_error" in loss_dict
