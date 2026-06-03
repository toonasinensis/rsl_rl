# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn.functional import binary_cross_entropy_with_logits

from rsl_rl.algorithms.plugins.base import PPOPlugin
from rsl_rl.utils import resolve_callable

from .amp_provider import AMPExpertProvider, ExternalAMPProvider


class AMPPlugin(nn.Module, PPOPlugin):
    """Minimal AMP plugin skeleton for the PPO plugin architecture."""

    def __init__(
        self,
        reward_scale: float = 1.0,
        loss_coef: float = 1.0,
        gradient_penalty_coef: float = 10.0,
        amp_obs_key: str = "amp",
        provider: AMPExpertProvider | dict | None = None,
        discriminator: nn.Module | dict | None = None,
        min_normalized_std: list[float] | float | None = None,
    ) -> None:
        super().__init__()
        self.reward_scale = float(reward_scale)
        self.loss_coef = float(loss_coef)
        self.gradient_penalty_coef = float(gradient_penalty_coef)
        self.amp_obs_key = amp_obs_key
        self.min_std = min_normalized_std

        self.provider = self._build_provider(provider)
        self.discriminator = discriminator if isinstance(discriminator, nn.Module) else None
        self.discriminator_cfg = None if isinstance(discriminator, nn.Module) else copy.deepcopy(discriminator)

        self._current_amp_obs: torch.Tensor | None = None
        self._extra_metric_sums: dict[str, float] = {}
        self._extra_metric_updates = 0
        self._init_device: str | None = None

    def on_init(self, ppo, env) -> None:
        """Initialize provider state and attach any plugin parameters to the optimizer."""
        obs = env.get_observations()
        self.provider.setup(env, ppo.device, obs=obs)
        if self.discriminator is None and self.discriminator_cfg is not None:
            self.discriminator = self._build_discriminator(obs).to(ppo.device)
        self._init_device = ppo.device
        self.to(ppo.device)
        self._attach_optimizer_params(ppo)

    def on_after_act(self, runner, obs: TensorDict) -> None:
        """Cache the current AMP observations when available."""
        del runner
        amp_obs = obs.get(self.amp_obs_key)
        if amp_obs is None:
            self._current_amp_obs = None
            return
        self._current_amp_obs = amp_obs.detach().clone()

    def on_after_step(
        self,
        runner,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> torch.Tensor:
        """Record policy transitions and expose AMP reward decomposition through step metrics."""
        del runner
        if self._current_amp_obs is None:
            return rewards

        next_amp_obs = obs.get(self.amp_obs_key)
        if next_amp_obs is None:
            self._current_amp_obs = None
            return rewards

        next_amp_obs_with_term = next_amp_obs.detach().clone()
        done_ids = dones.reshape(next_amp_obs_with_term.shape[0], -1).any(dim=-1)
        if done_ids.any():
            next_amp_obs_with_term[done_ids] = self._current_amp_obs[done_ids]

        if self.discriminator is not None and hasattr(self.discriminator, "update_normalization"):
            self.discriminator.update_normalization(obs)

        self.provider.record_policy_transition(
            self._current_amp_obs,
            next_amp_obs_with_term,
            dones.detach(),
        )

        if self.discriminator is not None:
            reward_components = self.compute_reward_components(
                self._current_amp_obs,
                next_amp_obs_with_term,
                rewards,
            )
            step_metrics = extras.setdefault("step_metrics", {})
            step_metrics.update(
                {
                    key: self._as_step_metric_tensor(value)
                    for key, value in reward_components.items()
                }
            )
            rewards = reward_components["mixed_reward"]

        self._current_amp_obs = None
        return rewards

    def on_update_start(self, ppo) -> None:
        """Reset aggregate plugin metrics before a PPO update."""
        del ppo
        self._extra_metric_sums = {}
        self._extra_metric_updates = 0

    def on_per_batch_extra_loss(
        self,
        ppo,
        batch,
        forward_results: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute discriminator loss and return it as an extra PPO objective term."""
        del forward_results
        if self.discriminator is None:
            return {}
        if getattr(self, "_embedded_loss_handled_externally", False):
            return {}

        amp_loss_dict = self.compute_loss_from_policy_obs(
            batch.observations,
            use_policy_replay=True,
            detach_metrics=True,
        )
        if not amp_loss_dict:
            return {}
        self._accumulate_extra_metrics(amp_loss_dict)
        return {"amp_loss": self.loss_coef * amp_loss_dict["amp_loss"]}

    def on_post_update(self, ppo) -> dict[str, float]:
        """Expose aggregate AMP metrics after an update."""
        del ppo
        if self._extra_metric_updates == 0:
            return {}
        return {
            key: value / self._extra_metric_updates
            for key, value in self._extra_metric_sums.items()
        }

    def on_after_backward(self, ppo) -> None:
        """Clip discriminator gradients and clamp policy std before optimizer.step()."""
        if self.discriminator is not None:
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), ppo.max_grad_norm)
        if self.min_std is not None:
            self._clamp_policy_std(ppo)

    def on_save(self, ppo, saved_dict: dict) -> None:
        """Save plugin module state and provider state."""
        PPOPlugin.on_save(self, ppo, saved_dict)
        plugin_extra_state = saved_dict.setdefault("plugin_extra_state", {})
        plugin_extra_state[self._state_key()] = {
            "provider_state": self.provider.state_dict(),
            "amp_obs_key": self.amp_obs_key,
            "reward_scale": self.reward_scale,
            "loss_coef": self.loss_coef,
            "gradient_penalty_coef": self.gradient_penalty_coef,
            "min_std": copy.deepcopy(self.min_std),
            "discriminator_cfg": copy.deepcopy(self.discriminator_cfg),
        }

    def on_load(self, ppo, loaded_dict: dict) -> None:
        """Restore plugin module state and provider state."""
        PPOPlugin.on_load(self, ppo, loaded_dict)
        plugin_extra_state = loaded_dict.get("plugin_extra_state", {})
        state = plugin_extra_state.get(self._state_key())
        if not state:
            return
        self.amp_obs_key = state.get("amp_obs_key", self.amp_obs_key)
        self.reward_scale = state.get("reward_scale", self.reward_scale)
        self.loss_coef = state.get("loss_coef", self.loss_coef)
        self.gradient_penalty_coef = state.get("gradient_penalty_coef", self.gradient_penalty_coef)
        self.min_std = state.get("min_std", self.min_std)
        self.discriminator_cfg = state.get("discriminator_cfg", self.discriminator_cfg)
        self.provider.load_state_dict(state.get("provider_state", {}))

    @staticmethod
    def _build_provider(provider: AMPExpertProvider | dict | None) -> AMPExpertProvider:
        if provider is None:
            return ExternalAMPProvider()
        if isinstance(provider, AMPExpertProvider):
            return provider
        provider_cfg = copy.deepcopy(provider)
        provider_cls = resolve_callable(provider_cfg.pop("class_name", ExternalAMPProvider))  # type: ignore[arg-type]
        provider_instance = provider_cls(**provider_cfg)
        if not isinstance(provider_instance, AMPExpertProvider):
            raise TypeError(
                f"Configured AMP provider '{type(provider_instance).__name__}' must inherit from AMPExpertProvider."
            )
        return provider_instance

    def _attach_optimizer_params(self, ppo) -> None:
        existing_params = {
            id(param)
            for param_group in ppo.optimizer.param_groups
            for param in param_group["params"]
        }
        new_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in existing_params
        ]
        if new_params:
            ppo.optimizer.add_param_group({"params": new_params, "lr": ppo.learning_rate})

    def _build_discriminator(self, obs: TensorDict) -> nn.Module:
        discriminator_cfg = copy.deepcopy(self.discriminator_cfg or {})
        discriminator_cls = resolve_callable(discriminator_cfg.pop("class_name", "AMPDiscriminator"))  # type: ignore[arg-type]
        obs_set = discriminator_cfg.pop("obs_set", "amp")
        obs_groups = discriminator_cfg.pop("obs_groups", {obs_set: [self.amp_obs_key]})
        return discriminator_cls(obs, obs_groups, obs_set, **discriminator_cfg)

    def compute_reward_components(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        task_rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute AMP reward decomposition for a current/next observation pair."""
        return self._compute_reward_components(amp_obs, next_amp_obs, task_rewards)

    def compute_reward_from_observations(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Compute AMP-only reward from observations for legacy AMPPPO-style integration."""
        if isinstance(obs, TensorDict):
            amp_obs = obs.get(self.amp_obs_key)
            if amp_obs is None:
                raise KeyError(f"Observation dict does not contain AMP key '{self.amp_obs_key}'.")
        else:
            amp_obs = obs

        task_rewards = torch.zeros(amp_obs.shape[0], device=amp_obs.device, dtype=amp_obs.dtype)
        reward_components = self.compute_reward_components(amp_obs, amp_obs, task_rewards)
        return reward_components["amp_reward"].reshape(-1)

    def _compute_reward_components(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        task_rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert self.discriminator is not None
        task_rewards = task_rewards.detach().reshape(-1)

        if hasattr(self.discriminator, "predict_amp_reward_components"):
            reward_components, _ = self.discriminator.predict_amp_reward_components(
                amp_obs,
                next_amp_obs,
                task_rewards,
                normalizer=getattr(self.provider, "normalizer", None),
            )
            return {
                key: value.detach().reshape(-1)
                for key, value in reward_components.items()
            }

        with torch.no_grad():
            logits = self.discriminator(next_amp_obs)
            expert_prob = torch.sigmoid(logits)
            amp_reward = self.reward_scale * (
                -torch.log(torch.clamp(1.0 - expert_prob, min=1.0e-4))
            ).squeeze(-1)
        return {
            "task_reward": task_rewards,
            "amp_reward": amp_reward.detach().reshape(-1),
            "mixed_reward": (task_rewards + amp_reward).detach().reshape(-1),
        }

    def _sample_policy_training_obs(self, batch) -> torch.Tensor | None:
        amp_obs = batch.observations.get(self.amp_obs_key)
        batch_size = batch.observations.batch_size[0]
        if amp_obs is not None:
            batch_size = amp_obs.reshape(-1, amp_obs.shape[-1]).shape[0]

        policy_pairs = self.provider.sample_policy_pairs(batch_size)
        if policy_pairs is not None:
            _, policy_next_obs = policy_pairs
            return policy_next_obs.detach().reshape(-1, policy_next_obs.shape[-1])

        if amp_obs is None:
            return None
        return amp_obs.detach().reshape(-1, amp_obs.shape[-1])

    def _sample_expert_training_obs(self, batch_size: int) -> torch.Tensor:
        _, expert_next_obs = self.provider.sample_expert_pairs(batch_size)
        return expert_next_obs.detach().reshape(-1, expert_next_obs.shape[-1])

    def compute_loss_from_policy_obs(
        self,
        policy_obs: TensorDict | torch.Tensor,
        *,
        use_policy_replay: bool = False,
        detach_metrics: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute AMP discriminator loss from policy observations or tensors."""
        if self.discriminator is None:
            return {}

        policy_train_obs = self._resolve_policy_training_obs(policy_obs, use_policy_replay=use_policy_replay)
        if policy_train_obs is None:
            return {}

        expert_train_obs = self._sample_expert_training_obs(policy_train_obs.shape[0])
        return self._compute_discriminator_loss(
            policy_train_obs,
            expert_train_obs,
            detach_metrics=detach_metrics,
        )

    def _resolve_policy_training_obs(
        self,
        policy_obs: TensorDict | torch.Tensor,
        *,
        use_policy_replay: bool,
    ) -> torch.Tensor | None:
        if isinstance(policy_obs, torch.Tensor):
            policy_train_obs = policy_obs
            batch_size = policy_train_obs.reshape(-1, policy_train_obs.shape[-1]).shape[0]
            if use_policy_replay:
                policy_pairs = self.provider.sample_policy_pairs(batch_size)
                if policy_pairs is not None:
                    _, policy_next_obs = policy_pairs
                    return policy_next_obs.detach().reshape(-1, policy_next_obs.shape[-1])
            return policy_train_obs.detach().reshape(-1, policy_train_obs.shape[-1])

        amp_obs = policy_obs.get(self.amp_obs_key)
        batch_size = policy_obs.batch_size[0]
        if amp_obs is not None:
            batch_size = amp_obs.reshape(-1, amp_obs.shape[-1]).shape[0]

        if use_policy_replay:
            policy_pairs = self.provider.sample_policy_pairs(batch_size)
            if policy_pairs is not None:
                _, policy_next_obs = policy_pairs
                return policy_next_obs.detach().reshape(-1, policy_next_obs.shape[-1])

        if amp_obs is None:
            return None
        return amp_obs.detach().reshape(-1, amp_obs.shape[-1])

    def _compute_discriminator_loss(
        self,
        policy_train_obs: torch.Tensor,
        expert_train_obs: torch.Tensor,
        *,
        detach_metrics: bool = True,
    ) -> dict[str, torch.Tensor]:
        assert self.discriminator is not None

        policy_inputs = self._extract_discriminator_input(policy_train_obs).detach()
        expert_inputs = self._extract_discriminator_input(expert_train_obs).detach()

        policy_logits = self.discriminator(policy_inputs)
        expert_logits = self.discriminator(expert_inputs)
        policy_loss = binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
        expert_loss = binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
        gradient_penalty = policy_loss.new_tensor(0.0)

        if self.gradient_penalty_coef > 0.0:
            expert_inputs_for_gp = expert_inputs.detach().requires_grad_(True)
            expert_logits_for_gp = self.discriminator(expert_inputs_for_gp)
            gradients = torch.autograd.grad(
                expert_logits_for_gp.sum(),
                expert_inputs_for_gp,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradient_penalty = gradients.pow(2).sum(dim=-1).mean()

        amp_loss = 0.5 * (policy_loss + expert_loss) + self.gradient_penalty_coef * gradient_penalty
        metric_policy_loss = policy_loss.detach() if detach_metrics else policy_loss
        metric_expert_loss = expert_loss.detach() if detach_metrics else expert_loss
        metric_gradient_penalty = gradient_penalty.detach() if detach_metrics else gradient_penalty
        return {
            "amp_loss": amp_loss,
            "policy_loss": metric_policy_loss,
            "expert_loss": metric_expert_loss,
            "gradient_penalty": metric_gradient_penalty,
            "policy_accuracy": (policy_logits.detach() < 0.0).float().mean(),
            "expert_accuracy": (expert_logits.detach() > 0.0).float().mean(),
        }

    def record_loss_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        """Accumulate AMP loss metrics produced outside the plugin loss hook."""
        self._accumulate_extra_metrics(metrics)

    def _accumulate_extra_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        metric_aliases = {
            "amp_loss": "AMP/amp_loss",
            "policy_loss": "AMP/policy_loss",
            "expert_loss": "AMP/expert_loss",
            "gradient_penalty": "AMP/gradient_penalty",
            "policy_accuracy": "AMP/policy_accuracy",
            "expert_accuracy": "AMP/expert_accuracy",
        }
        self._extra_metric_updates += 1
        for key, value in metrics.items():
            metric_key = metric_aliases.get(key)
            if metric_key is None:
                continue
            scalar = value.item() if value.numel() == 1 else value.mean().item()
            self._extra_metric_sums[metric_key] = self._extra_metric_sums.get(metric_key, 0.0) + float(scalar)
        self._extra_metric_sums["AMP/weighted_amp_loss"] = (
            self._extra_metric_sums.get("AMP/weighted_amp_loss", 0.0)
            + float(self.loss_coef * metrics["amp_loss"].item())
        )

    def _clamp_policy_std(self, ppo) -> None:
        """Prevent policy std collapse by clamping trainable std parameters in-place."""
        dist = getattr(ppo.actor, "distribution", None)
        if dist is None:
            return

        with torch.no_grad():
            min_std = torch.as_tensor(self.min_std, device=ppo.device, dtype=torch.float32)
            if min_std.ndim == 0:
                min_std = min_std.unsqueeze(0)

            std_type = getattr(dist, "std_type", None)
            if std_type == "scalar" and hasattr(dist, "std_param"):
                target = dist.std_param
                if min_std.numel() == 1:
                    min_std = min_std.expand_as(target)
                elif min_std.numel() != target.numel():
                    min_std = torch.clamp_min(min_std.min(), 1.0e-6).expand_as(target)
                target.clamp_(min=min_std)
            elif std_type == "log" and hasattr(dist, "log_std_param"):
                target = dist.log_std_param
                if min_std.numel() == 1:
                    min_std = min_std.expand_as(target)
                elif min_std.numel() != target.numel():
                    min_std = torch.clamp_min(min_std.min(), 1.0e-6).expand_as(target)
                target.clamp_(min=torch.log(torch.clamp_min(min_std, 1.0e-6)))

    def _extract_discriminator_input(self, obs: torch.Tensor) -> torch.Tensor:
        if hasattr(self.discriminator, "extract_input"):
            return self.discriminator.extract_input(obs)
        return obs

    @staticmethod
    def _as_step_metric_tensor(value: torch.Tensor) -> torch.Tensor:
        metric = value.detach()
        if metric.ndim == 1:
            metric = metric.unsqueeze(-1)
        return metric
