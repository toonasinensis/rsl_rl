# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BackboneMoE (Mixture-of-Experts backbone)."""

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import BackboneMoE
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
OUTPUT_DIM = 6
OBS_GROUPS = {"actor": ["policy"]}


def _make_moe_backbone(**kwargs: object) -> tuple[BackboneMoE, TensorDict]:
    """Create a BackboneMoE and matching observations."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    defaults: dict[str, object] = {
        "hidden_dims": [16],
        "expert_hidden_dims": [16],
        "router_hidden_dims": [16],
        "activation": "elu",
        "num_experts": 4,
        "top_k": 2,
        "router_temperature": 1.0,
    }
    defaults.update(kwargs)
    backbone = BackboneMoE(obs, OBS_GROUPS, "actor", OUTPUT_DIM, **defaults)
    return backbone, obs


def _concat_obs(backbone: BackboneMoE, obs: TensorDict) -> torch.Tensor:
    """Build the flat input tensor the MoE backbone uses internally."""
    from rsl_rl.models.backbone_base import BaseModel

    obs_normed = BaseModel.forward(backbone, obs)
    return torch.cat([obs_normed[g] for g in backbone.obs_groups], dim=-1)


class TestBackboneMoEInit:
    """Constructor validation."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_experts": 0}, "num_experts"),
            ({"top_k": 0}, "top_k"),
            ({"num_experts": 2, "top_k": 3}, "top_k"),
            ({"router_temperature": 0.0}, "router_temperature"),
        ],
    )
    def test_invalid_config_raises(self, kwargs: dict, match: str) -> None:
        obs = make_obs(2, OBS_DIM)
        with pytest.raises(ValueError, match=match):
            BackboneMoE(obs, OBS_GROUPS, "actor", OUTPUT_DIM, **kwargs)

    def test_submodules_are_registered(self) -> None:
        backbone, _ = _make_moe_backbone(num_experts=3)
        param_names = {name for name, _ in backbone.named_parameters()}
        assert any(name.startswith("module_dict.router.") for name in param_names)
        for i in range(3):
            assert any(name.startswith(f"module_dict.expert_{i}.") for name in param_names)

    def test_router_only_freezes_experts_on_init(self) -> None:
        backbone, _ = _make_moe_backbone(num_experts=2, router_only=True)
        router_trainable = [p.requires_grad for p in backbone.module_dict["router"].parameters()]
        expert_trainable = [
            p.requires_grad
            for i in range(backbone.num_experts)
            for p in backbone.module_dict[f"expert_{i}"].parameters()
        ]
        assert all(router_trainable)
        assert not any(expert_trainable)


class TestComputeTopkGates:
    """Unit tests for sparse gate computation."""

    def test_full_routing_matches_softmax(self) -> None:
        backbone, _ = _make_moe_backbone(num_experts=3, top_k=3, router_temperature=0.5)
        logits = torch.randn(NUM_ENVS, 3)
        gates = backbone._compute_topk_gates(logits)
        expected = torch.softmax(logits / backbone.router_temperature, dim=-1)
        assert torch.allclose(gates, expected, atol=1e-6)
        assert torch.allclose(gates.sum(dim=-1), torch.ones(NUM_ENVS), atol=1e-6)

    def test_top_k_sparsity_and_renormalization(self) -> None:
        backbone, _ = _make_moe_backbone(num_experts=5, top_k=2)
        logits = torch.randn(NUM_ENVS, 5)
        gates = backbone._compute_topk_gates(logits)

        assert gates.shape == (NUM_ENVS, 5)
        assert torch.allclose(gates.sum(dim=-1), torch.ones(NUM_ENVS), atol=1e-5)
        active_per_row = (gates > 1e-8).sum(dim=-1)
        assert torch.all(active_per_row == 2)

    def test_top_k_one_selects_single_expert(self) -> None:
        backbone, _ = _make_moe_backbone(num_experts=4, top_k=1)
        logits = torch.randn(8, 4)
        gates = backbone._compute_topk_gates(logits)
        assert torch.all((gates > 0).sum(dim=-1) == 1)

    def test_top_k_softmaxes_only_selected_logits(self) -> None:
        """Top-k is applied on logits, then softmax (not softmax-then-top-k)."""
        backbone, _ = _make_moe_backbone(num_experts=4, top_k=2, router_temperature=1.0)
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        gates = backbone._compute_topk_gates(logits)
        expected = torch.zeros(1, 4)
        expected[0, :2] = torch.softmax(torch.tensor([3.0, 2.0]), dim=-1)
        assert torch.allclose(gates, expected, atol=1e-6)


class TestBackboneMoEForward:
    """Forward pass shape and numerical correctness."""

    def test_forward_output_shape(self) -> None:
        backbone, obs = _make_moe_backbone()
        out = backbone(obs)
        assert out["actions"].shape == (NUM_ENVS, OUTPUT_DIM)

    def test_forward_matches_manual_mixture(self) -> None:
        torch.manual_seed(0)
        backbone, obs = _make_moe_backbone(num_experts=3, top_k=2)
        backbone.eval()

        with torch.no_grad():
            x = _concat_obs(backbone, obs)
            router_logits = backbone.module_dict["router"](x)
            gates = backbone._compute_topk_gates(router_logits)
            expert_outputs = torch.stack(
                [backbone.module_dict[f"expert_{i}"](x) for i in range(backbone.num_experts)],
                dim=1,
            )
            expected = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)

        actual = backbone(obs)["actions"]
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_train_mode_exposes_routing_tensors(self) -> None:
        backbone, obs = _make_moe_backbone(
            load_balance_loss_weight=0.01,
            router_z_loss_weight=0.01,
        )
        out = backbone(obs, train_mode=True)

        assert "moe_router_logits" in out
        assert "moe_gates" in out
        assert out["moe_router_logits"].shape == (NUM_ENVS, backbone.num_experts)
        assert out["moe_gates"].shape == (NUM_ENVS, backbone.num_experts)
        assert "aux_losses" in out
        assert "moe_load_balance" in out["aux_losses"]
        assert "moe_router_z" in out["aux_losses"]

    def test_eval_mode_omits_training_keys(self) -> None:
        backbone, obs = _make_moe_backbone(
            load_balance_loss_weight=0.01,
            router_z_loss_weight=0.01,
        )
        out = backbone(obs, train_mode=False)
        assert set(out.keys()) == {"actions"}


class TestBackboneMoEModuleManagement:
    """load_modules, freeze_modules, and freeze_experts."""

    def test_load_modules_copies_weights(self) -> None:
        source, _ = _make_moe_backbone(num_experts=2, hidden_dims=[32])
        target, obs = _make_moe_backbone(num_experts=2, hidden_dims=[32])

        target.load_modules(copy.deepcopy(source.module_dict))
        source.eval()
        target.eval()
        assert torch.allclose(source(obs)["actions"], target(obs)["actions"], atol=1e-6)

    def test_load_modules_unknown_name_raises(self) -> None:
        backbone, _ = _make_moe_backbone()
        with pytest.raises(KeyError, match="Unknown module"):
            backbone.load_modules({"unknown": backbone.module_dict["router"]})

    def test_load_modules_from_state_dicts(self) -> None:
        source, _ = _make_moe_backbone(num_experts=2, hidden_dims=[32])
        target, obs = _make_moe_backbone(num_experts=2, hidden_dims=[32])

        target.load_modules(
            {
                "router": source.module_dict["router"].state_dict(),
                "expert_0": source.module_dict["expert_0"].state_dict(),
                "expert_1": source.module_dict["expert_1"].state_dict(),
            }
        )
        source.eval()
        target.eval()
        assert torch.allclose(source(obs)["actions"], target(obs)["actions"], atol=1e-6)

    def test_freeze_experts_leaves_router_trainable(self) -> None:
        backbone, _ = _make_moe_backbone(num_experts=2)
        backbone.freeze_experts()

        router_trainable = [p.requires_grad for p in backbone.module_dict["router"].parameters()]
        expert_trainable = [
            p.requires_grad
            for i in range(backbone.num_experts)
            for p in backbone.module_dict[f"expert_{i}"].parameters()
        ]
        assert all(router_trainable)
        assert not any(expert_trainable)

    def test_load_experts_from_state_dicts(self) -> None:
        source, _ = _make_moe_backbone(num_experts=2)
        target, obs = _make_moe_backbone(num_experts=2)

        expert_states = [source.module_dict[f"expert_{i}"].state_dict() for i in range(2)]
        target.load_experts(expert_states)

        source.eval()
        target.eval()
        with torch.no_grad():
            x = _concat_obs(source, obs)
            source_out = torch.stack([source.module_dict[f"expert_{i}"](x) for i in range(2)], dim=1)
            target_out = torch.stack([target.module_dict[f"expert_{i}"](x) for i in range(2)], dim=1)
        assert torch.allclose(source_out, target_out, atol=1e-6)
