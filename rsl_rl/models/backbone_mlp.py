from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import HiddenState, MLP

from .backbone_base import BaseModel


class BackboneMLP(BaseModel):
    """MLP backbone with optional legacy stochastic distribution support."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **backbone_cfg,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **backbone_cfg)
        input_dim = sum(self.obs_dim[g] for g in obs_groups[obs_set])
        self.mlp = MLP(input_dim, output_dim, backbone_cfg.get("hidden_dims", []), backbone_cfg.get("activation", "relu"))

    def forward(
        self,
        obs: TensorDict | dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """Return deterministic or sampled MLP output."""
        latent = self.get_latent(obs, masks=masks, hidden_state=hidden_state, train_mode=train_mode)
        return self._resolve_output(self.mlp(latent), stochastic_output=stochastic_output)

    def get_latent(
        self,
        obs: TensorDict | dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """Return concatenated normalized observations."""
        obs = super().forward(obs, masks, hidden_state, train_mode)
        return torch.cat([obs[g] for g in self.obs_groups], dim=-1)

    def as_jit(self, output_module: nn.Module | None = None) -> nn.Module:
        """Return a TorchScript-exportable deterministic MLP."""
        return _TorchMLPModel(self, output_module)

    def as_onnx(self, verbose: bool = False, output_module: nn.Module | None = None) -> nn.Module:
        """Return an ONNX-exportable deterministic MLP."""
        return _OnnxMLPModel(self, verbose, output_module)


class _TorchMLPModel(nn.Module):
    """Exportable MLP model for TorchScript."""

    def __init__(self, model: BackboneMLP, output_module: nn.Module | None = None) -> None:
        """Create a TorchScript-friendly copy of an MLP model."""
        super().__init__()
        self.obs_dims = [model.obs_dim[g] for g in model.obs_groups]
        self.has_normalizers = model.obs_normalizers is not None
        self.obs_normalizers = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers[g]) for g in model.obs_groups] if self.has_normalizers else []
        )
        self.mlp = copy.deepcopy(model.mlp)
        if output_module is None and model.distribution is not None:
            output_module = model.distribution.as_deterministic_output_module()
        self.output_module = copy.deepcopy(output_module) if output_module is not None else nn.Identity()

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize concatenated observations using per-group normalizers."""
        if not self.has_normalizers:
            return obs
        normalized_obs = []
        start = 0
        for i, normalizer in enumerate(self.obs_normalizers):
            end = start + self.obs_dims[i]
            normalized_obs.append(normalizer(obs[..., start:end]))
            start = end
        return torch.cat(normalized_obs, dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from concatenated observations."""
        return self.output_module(self.mlp(self._normalize(obs)))


class _OnnxMLPModel(_TorchMLPModel):
    """Exportable MLP model for ONNX."""

    def __init__(self, model: BackboneMLP, verbose: bool, output_module: nn.Module | None = None) -> None:
        """Create an ONNX-export wrapper around an MLP model."""
        super().__init__(model, output_module)
        self.verbose = verbose
        self.input_dim = sum(model.obs_dim.values())

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_dim),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
