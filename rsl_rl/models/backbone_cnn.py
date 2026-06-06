# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import CNN, HiddenState, MLP

from .backbone_base import BaseModel


class BackboneCNN(BaseModel):
    """CNN-based neural model.

    This model uses one or more convolutional neural network (CNN) encoders to process one or more 2D observation groups
    before passing the resulting latent to an MLP. Any 1D observation groups are directly concatenated with the CNN
    latent and passed to the MLP. 1D observations can be normalized before being passed to the MLP. The output of the
    model can be either deterministic or stochastic, in which case a distribution module is used to sample the outputs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        **backbone_cfg,
    ) -> None:
        """Initialize the CNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the CNN and MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            cnn_cfg: Configuration of the CNN encoder(s).
            cnns: CNN modules to use, e.g., for sharing CNNs between actor and critic. If None, new CNNs are created.
        """
        hidden_dims = hidden_dims if hidden_dims is not None else backbone_cfg.get("hidden_dims", (256, 256, 256))
        activation = activation if activation is not None else backbone_cfg.get("activation", "elu")
        cnn_cfg = cnn_cfg if cnn_cfg is not None else backbone_cfg.get("cnn_cfg")
        cnns = cnns if cnns is not None else backbone_cfg.get("cnns")

        super().__init__(obs, obs_groups, obs_set, output_dim, **backbone_cfg)

        # Create or validate CNN encoders
        if cnns is not None:
            # Check compatibility if CNNs are provided
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
            # Create CNNs for each 2D observation
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    **cnn_cfg[obs_group],
                )

        # Compute latent dimension of the CNNs
        self.cnn_latent_dim = 0
        for cnn in cnns.values():
            if cnn.output_channels is not None:
                raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
            self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore

        input_dim = sum(self.obs_dim.values()) + self.cnn_latent_dim
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)

        # Register CNN encoders
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)

    def forward(
        self,
        obs: TensorDict | dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return CNN/MLP backbone outputs in the BaseModel dict format."""
        obs = super().forward(obs, masks, hidden_state, train_mode)

        latent_list = []
        if self.obs_groups:
            latent_list.append(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
        # Process 2D observation groups with CNNs
        latent_cnn_list = [self.cnns[obs_group](obs[obs_group]) for obs_group in self.obs_groups_2d]
        latent_list.append(torch.cat(latent_cnn_list, dim=-1))
        latent = torch.cat(latent_list, dim=-1)
        return {"actions": self.mlp(latent)}

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCNNModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], dict]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = {}
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        # Iterate through active observation groups and separate 1D and 2D observations
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d[obs_group] = obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. If this is intentional, use the MLP model instead.")

        # Store active 2D observation groups and dimensions directly as attributes
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
        # Return active 1D observation groups and dimensions for BaseModel
        return obs_groups_1d, obs_dim_1d


class _TorchCNNModel(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: BackboneCNN) -> None:
        """Create a TorchScript-friendly copy of a CNNModel."""
        super().__init__()
        self.obs_dims_1d = [model.obs_dim[g] for g in model.obs_groups]
        self.obs_normalizers = (
            nn.ModuleList([copy.deepcopy(model.obs_normalizers[g]) for g in model.obs_groups])
            if model.obs_normalizers is not None
            else None
        )
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)

    def _normalize_1d(self, obs_1d: torch.Tensor) -> torch.Tensor:
        """Normalize concatenated 1D observations using the backbone's per-group normalizers."""
        if self.obs_normalizers is None or len(self.obs_dims_1d) == 0:
            return obs_1d
        chunks = torch.split(obs_1d, self.obs_dims_1d, dim=-1)
        return torch.cat([normalizer(chunk) for normalizer, chunk in zip(self.obs_normalizers, chunks)], dim=-1)

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latent_1d = self._normalize_1d(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(cnn(obs_2d[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return out

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxCNNModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: BackboneCNN, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_dims_1d = [model.obs_dim[g] for g in model.obs_groups]
        self.obs_normalizers = (
            nn.ModuleList([copy.deepcopy(model.obs_normalizers[g]) for g in model.obs_groups])
            if model.obs_normalizers is not None
            else None
        )
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = sum(model.obs_dim.values())

    def _normalize_1d(self, obs_1d: torch.Tensor) -> torch.Tensor:
        """Normalize concatenated 1D observations using the backbone's per-group normalizers."""
        if self.obs_normalizers is None or len(self.obs_dims_1d) == 0:
            return obs_1d
        chunks = torch.split(obs_1d, self.obs_dims_1d, dim=-1)
        return torch.cat([normalizer(chunk) for normalizer, chunk in zip(self.obs_normalizers, chunks)], dim=-1)

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self._normalize_1d(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
