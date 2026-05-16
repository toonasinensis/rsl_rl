from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from .backbone_base import BaseModel
from rsl_rl.modules import HiddenState, MLP


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

        self.router = MLP(
            input_dim=input_dim,
            output_dim=self.num_experts,
            hidden_dims=router_hidden_dims,
            activation=activation,
        )
        self.experts = nn.ModuleList(
            [
                MLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dims=expert_hidden_dims,
                    activation=activation,
                )
                for _ in range(self.num_experts)
            ]
        )

    def _compute_topk_gates(self, router_logits: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(router_logits / self.router_temperature, dim=-1)
        if self.top_k >= self.num_experts:
            return gates

        topk_values, topk_indices = torch.topk(gates, k=self.top_k, dim=-1)
        sparse_gates = torch.zeros_like(gates)
        sparse_gates.scatter_(dim=-1, index=topk_indices, src=topk_values)
        sparse_gates = sparse_gates / sparse_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return sparse_gates

    def _compute_aux_losses(self, gates: torch.Tensor, router_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        if self.load_balance_loss_weight > 0.0:
            # NOTE think whether this is the best load balance loss function
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
        moe_only: bool = False,
    ):
        """ Router(x) * experts(x) -> actions
        """
        obs = super().forward(obs, masks, hidden_state, train_mode)
        x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)

        router_logits = self.router(x)
        gates = self._compute_topk_gates(router_logits)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        actions = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)

        backbone_output: dict[str, object] = {"actions": actions}
        if train_mode:
            aux_losses = self._compute_aux_losses(gates, router_logits)
            if aux_losses: # for encouraging exploration
                backbone_output["aux_losses"] = aux_losses
            backbone_output["moe_router_logits"] = router_logits
            backbone_output["moe_gates"] = gates

        return backbone_output

    def load_experts(self, experts):
        """ experts: list of expert models state dict
        """
        # load experts from given checkpoint
        for expert, expert_state_dict in zip(self.experts, experts):
            expert.load_state_dict(expert_state_dict)
  
    def freeze_experts(self):
        # freeze the gradients of experts to ensure only the router's parameters are trainable
        for param in self.experts.parameters():
            param.requires_grad = False
