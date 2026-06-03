# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.utils import clone_state_dict_tensors


class PPOPlugin:
    """Lifecycle hooks for extending PPO and runner behavior."""

    _plugin_key: str | None = None

    def on_init(self, ppo, env) -> None:
        """Initialize the plugin after PPO and the environment are ready."""

    def on_after_act(self, runner, obs: TensorDict) -> None:
        """Hook called after ``alg.act`` and before ``env.step``."""

    def on_after_step(
        self,
        runner,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> torch.Tensor:
        """Hook called after ``env.step`` and before ``alg.process_env_step``."""
        return rewards

    def on_update_start(self, ppo) -> None:
        """Hook called once before PPO iterates over mini-batches."""

    def on_per_batch_extra_loss(
        self,
        ppo,
        batch,
        forward_results: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return extra loss terms for the current mini-batch."""
        return {}

    def on_after_backward(self, ppo) -> None:
        """Hook called after backward/reduce and before gradient clipping and optimizer step."""

    def on_post_backward(self, ppo) -> None:
        """Hook called after the optimizer step for each mini-batch."""

    def on_post_update(self, ppo) -> dict[str, float]:
        """Return extra metrics after a PPO update finishes."""
        return {}

    def on_train_mode(self, ppo) -> None:
        """Hook called when PPO switches to train mode."""
        if isinstance(self, nn.Module):
            self.train()

    def on_eval_mode(self, ppo) -> None:
        """Hook called when PPO switches to eval mode."""
        if isinstance(self, nn.Module):
            self.eval()

    def on_save(self, ppo, saved_dict: dict) -> None:
        """Hook called during checkpoint save."""
        if not isinstance(self, nn.Module):
            return

        state_dict = self.state_dict()
        if not state_dict:
            return

        plugin_state_dicts = saved_dict.setdefault("plugin_state_dicts", {})
        plugin_state_dicts[self._state_key()] = clone_state_dict_tensors(state_dict)

    def on_load(self, ppo, loaded_dict: dict) -> None:
        """Hook called during checkpoint load."""
        if not isinstance(self, nn.Module):
            return

        plugin_state_dicts = loaded_dict.get("plugin_state_dicts", {})
        state_dict = plugin_state_dicts.get(self._state_key())
        if state_dict is not None:
            self.load_state_dict(clone_state_dict_tensors(state_dict), strict=False)

    def _state_key(self) -> str:
        if self._plugin_key is not None:
            return self._plugin_key
        return type(self).__name__


class AuxLossPlugin(nn.Module, PPOPlugin):
    """Base class for auxiliary-loss plugins that extend the PPO objective."""

    def __init__(self) -> None:
        super().__init__()
        self._extra_metric_sums: dict[str, float] = {}
        self._extra_metric_updates = 0

    def setup(self, actor: nn.Module, obs: TensorDict) -> None:
        """Post-construction setup with access to the actor and a sample observation dict."""

    def compute_loss(
        self,
        forward_results: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | dict | None],
        batch: object,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary loss terms for the current mini-batch."""
        return {}

    def extra_metrics(self) -> dict[str, float]:
        """Return additional scalar metrics to include in the training log."""
        return {}

    def on_init(self, ppo, env) -> None:
        obs = env.get_observations()
        self.setup(ppo.actor, obs)
        self.to(ppo.device)
        self._attach_optimizer_params(ppo)

    def on_update_start(self, ppo) -> None:
        self._extra_metric_sums.clear()
        self._extra_metric_updates = 0

    def on_per_batch_extra_loss(
        self,
        ppo,
        batch,
        forward_results: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | dict] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = self.compute_loss(forward_results or {}, batch)
        metrics = self.extra_metrics()
        if metrics:
            self._extra_metric_updates += 1
            for key, value in metrics.items():
                self._extra_metric_sums[key] = self._extra_metric_sums.get(key, 0.0) + float(value)
        return losses

    def on_post_update(self, ppo) -> dict[str, float]:
        if self._extra_metric_updates == 0:
            return {}
        return {
            key: value / self._extra_metric_updates
            for key, value in self._extra_metric_sums.items()
        }

    def on_save(self, ppo, saved_dict: dict) -> None:
        PPOPlugin.on_save(self, ppo, saved_dict)

    def on_load(self, ppo, loaded_dict: dict) -> None:
        PPOPlugin.on_load(self, ppo, loaded_dict)

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
