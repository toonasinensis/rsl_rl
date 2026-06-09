from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from .backbone_base import BaseModel
from rsl_rl.modules import HiddenState, MLP


"""
架构设计：
1. 针对由多个网络构成的架构 通过字典管理模块参数
2. 未来延伸到 BaseModel 中
"""

class BackboneMoE(BaseModel):
    """Mixture-of-Experts backbone with an MLP router and MLP experts."""

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
        activation = backbone_cfg.get("activation", "elu")
        hidden_dims = backbone_cfg.get("hidden_dims", [256, 256])
        expert_hidden_dims = backbone_cfg.get("expert_hidden_dims", hidden_dims)
        router_hidden_dims = backbone_cfg.get("router_hidden_dims", [256])

        self.num_experts = int(backbone_cfg.get("num_experts", 4))
        self.top_k = int(backbone_cfg.get("top_k", 1))
        self.router_temperature = float(backbone_cfg.get("router_temperature", 1.0))
        self.load_balance_loss_weight = float(backbone_cfg.get("load_balance_loss_weight", 0.0))
        self.router_z_loss_weight = float(backbone_cfg.get("router_z_loss_weight", 0.0))
        self.router_only = backbone_cfg.get("router_only", False) # if router_only, experts will not be updated

        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}.")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts}).")
        if self.router_temperature <= 0.0:
            raise ValueError(f"router_temperature must be > 0, got {self.router_temperature}.")

        self.module_dict = nn.ModuleDict()
        self.module_dict["router"] = MLP(
            input_dim=input_dim,
            output_dim=self.num_experts,
            hidden_dims=router_hidden_dims,
            activation=activation,
        )
        for i in range(self.num_experts):
            self.module_dict[f"expert_{i}"] = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=expert_hidden_dims,
                activation=activation,
            )

        if self.router_only:
            self.freeze_experts()

    def _compute_topk_gates(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Turn router logits into sparse, row-normalized expert gate weights.

        Steps:
        1. Scale logits by ``router_temperature``.
        2. If ``top_k < num_experts``, keep only the top_k logits per sample and mask
           the rest with ``-inf`` (Switch-style routing on logits, not probabilities).
        3. Apply softmax so each row sums to 1 over the selected experts.

        Args:
            router_logits: Router output, shape ``(..., num_experts)``.

        Returns:
            Gate weights with the same shape; each row is a distribution over
            experts used to mix expert outputs in ``forward``.
        """
        scaled_logits = router_logits / self.router_temperature
        if self.top_k >= self.num_experts:
            return torch.softmax(scaled_logits, dim=-1)

        topk_logits, topk_indices = torch.topk(scaled_logits, k=self.top_k, dim=-1)
        sparse_logits = torch.full_like(scaled_logits, float("-inf"))
        sparse_logits.scatter_(dim=-1, index=topk_indices, src=topk_logits)
        return torch.softmax(sparse_logits, dim=-1)

    def _compute_aux_losses(self, gates: torch.Tensor, router_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """ Compute auxiliary losses for MoE backbone. for encouraging exploration
        """
        losses: dict[str, torch.Tensor] = {}
        if self.load_balance_loss_weight > 0.0:
            target = torch.full((self.num_experts,), 1.0 / self.num_experts, device=gates.device, dtype=gates.dtype)
            importance = gates.mean(dim=0)
            load = (gates > 0).float().mean(dim=0).to(dtype=gates.dtype)
            losses["moe_load_balance"] = (
                F.mse_loss(importance, target) + F.mse_loss(load, target)
            ) * self.load_balance_loss_weight

        if self.router_z_loss_weight > 0.0:
            # avoid router logits too large
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            losses["moe_router_z"] = z_loss * self.router_z_loss_weight
        return losses

    def forward(
        self,
        obs: TensorDict | dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ):
        """ Router(x) * experts(x) -> actions
        """
        obs = super().forward(obs, masks, hidden_state, train_mode)
        x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)

        router_logits = self.module_dict["router"](x)
        gates = self._compute_topk_gates(router_logits)

        # NOTE Not sparse in compute, only in weights -> may not reduce computation cost
        expert_outputs = torch.stack([self.module_dict[f"expert_{i}"](x) for i in range(self.num_experts)], dim=1)
        actions = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)

        backbone_output: dict[str, object] = {"actions": actions}
        
        if train_mode: 
            aux_losses = self._compute_aux_losses(gates, router_logits)
            if aux_losses: # for encouraging exploration
                backbone_output["aux_losses"] = aux_losses
            backbone_output["moe_router_logits"] = router_logits
            backbone_output["moe_gates"] = gates

        return backbone_output

    #region for managing neural networks
    @staticmethod
    def _resolve_state_dict(source: nn.Module | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(source, nn.Module):
            return source.state_dict()
        if isinstance(source, dict):
            return source
        raise TypeError(f"Expected nn.Module or state_dict, got {type(source)!r}.")

    def load_modules(
        self,
        module_dict: dict[str, nn.Module | dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Load weights into named submodules.

        Args:
            module_dict: Maps module names to an ``nn.Module`` or ``state_dict``.
            strict: Forwarded to ``load_state_dict``.
        """
        for module_name, source in module_dict.items():
            if module_name not in self.module_dict:
                raise KeyError(f"Unknown module '{module_name}'. Available: {list(self.module_dict.keys())}.")
            state_dict = self._resolve_state_dict(source)
            self.module_dict[module_name].load_state_dict(state_dict, strict=strict)

    def freeze_modules(self, module_names: list[str]):
        """module_names: list of module names to freeze
        """
        # freeze the gradients of modules to ensure only the router's parameters are trainable
        for module_name in module_names:
            for param in self.module_dict[module_name].parameters():
                param.requires_grad = False

    def load_experts(
        self,
        experts: list[nn.Module | dict[str, torch.Tensor]],
        *,
        strict: bool = True,
    ) -> None:
        """Load expert weights from modules or state dicts."""
        if len(experts) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} experts, got {len(experts)}.")
        for i, expert in enumerate(experts):
            self.module_dict[f"expert_{i}"].load_state_dict(self._resolve_state_dict(expert), strict=strict)

    def freeze_experts(self) -> None:
        """Freeze experts for router-only training.
        """
        experts_names = [f"expert_{i}" for i in range(self.num_experts)]
        self.freeze_modules(experts_names)
    #endregion
