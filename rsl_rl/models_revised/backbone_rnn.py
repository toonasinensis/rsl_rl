# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import RNN, HiddenState, MLP

from .backbone_base import BaseModel


class BackboneRNN(BaseModel):
    """RNN-based neural model.

    This model uses a recurrent neural network (RNN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available RNN types are "lstm" and "gru". Observations can be normalized before being passed to
    the RNN. The output of the model can be either deterministic or stochastic, in which case a distribution module is
    used to sample the outputs.
    """

    is_recurrent: bool = True
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str | None = None,
        rnn_type: str | None = None,
        rnn_hidden_dim: int | None = None,
        rnn_num_layers: int | None = None,
        **backbone_cfg,
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            rnn_type: Type of RNN to use ("lstm" or "gru").
            rnn_hidden_dim: Dimension of the RNN hidden state.
            rnn_num_layers: Number of RNN layers.
        """
        hidden_dims = hidden_dims if hidden_dims is not None else backbone_cfg.get("hidden_dims", (256, 256, 256))
        activation = activation if activation is not None else backbone_cfg.get("activation", "elu")
        rnn_type = rnn_type if rnn_type is not None else backbone_cfg.get("rnn_type", "lstm")
        rnn_hidden_dim = rnn_hidden_dim if rnn_hidden_dim is not None else backbone_cfg.get("rnn_hidden_dim", 256)
        rnn_num_layers = rnn_num_layers if rnn_num_layers is not None else backbone_cfg.get("rnn_num_layers", 1)

        super().__init__(obs, obs_groups, obs_set, output_dim, **backbone_cfg)

        # RNN
        self.rnn = RNN(sum(self.obs_dim.values()), rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.mlp = MLP(rnn_hidden_dim, output_dim, hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict | dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return recurrent backbone outputs in the BaseModel dict format."""
        # Keep padded recurrent batches intact; RNN consumes masks and unpads its output.
        obs = super().forward(obs, None, hidden_state, train_mode)
        latent = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        # Pass through the RNN
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return {"actions": self.mlp(latent)}

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRNNModel(self, verbose)

class _TorchGRUModel(nn.Module):
    """Exportable GRU model for JIT."""

    def __init__(self, model: BackboneRNN) -> None:
        """Create a TorchScript-friendly copy of a GRU-based RNNModel."""
        super().__init__()
        self.obs_dims = [model.obs_dim[g] for g in model.obs_groups]
        self.obs_normalizers = (
            nn.ModuleList([copy.deepcopy(model.obs_normalizers[g]) for g in model.obs_groups])
            if model.obs_normalizers is not None
            else None
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize concatenated observations using per-group normalizers."""
        if self.obs_normalizers is None:
            return obs
        chunks = torch.split(obs, self.obs_dims, dim=-1)
        return torch.cat([normalizer(chunk) for normalizer, chunk in zip(self.obs_normalizers, chunks)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one GRU inference step and update hidden states."""
        x = self._normalize(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return out

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported GRU hidden states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMModel(nn.Module):
    """Exportable LSTM model for JIT."""

    def __init__(self, model: BackboneRNN) -> None:
        """Create a TorchScript-friendly copy of an LSTM-based RNNModel."""
        super().__init__()
        self.obs_dims = [model.obs_dim[g] for g in model.obs_groups]
        self.obs_normalizers = (
            nn.ModuleList([copy.deepcopy(model.obs_normalizers[g]) for g in model.obs_groups])
            if model.obs_normalizers is not None
            else None
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize concatenated observations using per-group normalizers."""
        if self.obs_normalizers is None:
            return obs
        chunks = torch.split(obs, self.obs_dims, dim=-1)
        return torch.cat([normalizer(chunk) for normalizer, chunk in zip(self.obs_normalizers, chunks)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one LSTM inference step and update hidden and cell states."""
        x = self._normalize(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return out

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: BackboneRNN, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an RNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_dims = [model.obs_dim[g] for g in model.obs_groups]
        self.obs_normalizers = (
            nn.ModuleList([copy.deepcopy(model.obs_normalizers[g]) for g in model.obs_groups])
            if model.obs_normalizers is not None
            else None
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)

        # Detect RNN type
        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = sum(model.obs_dim.values())
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize concatenated observations using per-group normalizers."""
        if self.obs_normalizers is None:
            return obs
        chunks = torch.split(obs, self.obs_dims, dim=-1)
        return torch.cat([normalizer(chunk) for normalizer, chunk in zip(self.obs_normalizers, chunks)], dim=-1)

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export."""
        x = self._normalize(obs)

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            out = self.mlp(x)
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            out = self.mlp(x)
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
