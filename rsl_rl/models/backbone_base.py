# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class BaseModel(nn.Module):
    """Add normalization layer and define functions remains to be implemented"""

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    # ------------------------------------------------------------------
    # Init and forward
    # ------------------------------------------------------------------
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **backbone_cfg  # 收集所有额外的关键字参数
    ) -> None:
        """
            obs:        {obs_name: tensor}
            obs_groups: {obs_group_name: [obs_name_list]}
            obs_set:    obs_group_name
                -->
            self.obs_groups: [obs_name_list]
            self.obs_dim:    {obs_name: dim}
        """
        super().__init__()
        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.output_dim = output_dim
        self.distribution: Distribution | None = None
        distribution_cfg = backbone_cfg.get("distribution_cfg")
        if distribution_cfg is not None:
            dist_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
            self.distribution = dist_class(output_dim, **dist_cfg)
        # Observation normalization
        self.obs_normalization = backbone_cfg.get("obs_normalization", False)
        if self.obs_normalization:
            self.obs_normalizers = nn.ModuleDict(
                {g: EmpiricalNormalization(obs[g].shape[-1]) for g in obs_groups[obs_set]})
        else:
            self.obs_normalizers = None  # [FIX 1] 防止 update_normalization 报 AttributeError

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
            obs: {obs_name: tensor}
            self.obs_groups: {}
        """
        # import ipdb; ipdb.set_trace()
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        if self.obs_normalization:
            obs_normed = obs.clone()  # [FIX 2] 不改原始 TensorDict，避免 actor/critic 共享 obs 时互相污染
            for g in self.obs_groups:
                obs_normed[g] = self.obs_normalizers[g](obs[g])
        else:
            obs_normed = obs
        return obs_normed

    def _resolve_output(self, output: torch.Tensor, stochastic_output: bool = False) -> torch.Tensor:
        """Apply the optional legacy distribution head to a raw model output."""
        if self.distribution is None:
            return output

        self.distribution.update(output)
        if stochastic_output:
            return self.distribution.sample()
        return self.distribution.deterministic_output(output)

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return entropy of the current output distribution."""
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return current distribution parameters."""
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute output log-probability from the current distribution."""
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two distribution parameterizations."""
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def _compute_aux_losses(
            self,
            obs: TensorDict,
            named_latents: dict[str, torch.Tensor],
            active_latent: torch.Tensor,
        ):
        pass


    # ------------------------------------------------------------------
    # Normalisation update
    # ------------------------------------------------------------------
    def update_normalization(self, obs: TensorDict) -> None:
        """Update per-group running normalisation statistics."""
        if self.obs_normalizers is not None:
            for g in self.obs_groups:
                self.obs_normalizers[g].update(obs[g])

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], dict]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = {}
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim[obs_group] = obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass
    
    def as_jit(self):
        pass

    def as_onnx(self, verbose: bool):
        pass
