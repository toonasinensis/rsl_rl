# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl.env import VecEnv
from rsl_rl.algorithms import MoEPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class MoEOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that freezes MoE experts and trains the router only."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        freeze_experts: bool = True,
        expert_checkpoint_paths: list[str] | None = None,
    ) -> None:
        """Construct the MoE runner and optionally load/freeze experts.

        Args:
            env: Vectorized environment.
            train_cfg: Training configuration used by ``OnPolicyRunner``.
            log_dir: Optional logging directory.
            device: Training device.
            freeze_experts: Whether to freeze expert modules after construction.
            expert_checkpoint_paths: Optional list of checkpoint paths, one per expert.
                Each checkpoint may be either a raw state dict or a checkpoint dict
                containing one of ``"model_state_dict"``, ``"state_dict"``,
                ``"expert_state_dict"``, or ``"actor_state_dict"``.
        """
        moe_runner_cfg = train_cfg.get("moe_runner", {})
        freeze_experts = bool(moe_runner_cfg.get("freeze_experts", freeze_experts))
        if expert_checkpoint_paths is None:
            expert_checkpoint_paths = moe_runner_cfg.get("expert_checkpoint_paths")

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.alg: MoEPPO # shall be MoEPPO
        
        if expert_checkpoint_paths is not None:
            self.load_experts_from_checkpoints(expert_checkpoint_paths)

        if freeze_experts:
            self.freeze_experts()

    def freeze_experts(self) -> None:
        """Freeze MoE experts so optimization only trains unfrozen actor modules."""
        if not hasattr(self.alg, "freeze_experts"):
            raise TypeError(f"{type(self.alg).__name__} does not support expert freezing.")
        self.alg.freeze_experts()

    def load_experts_from_checkpoints(self, checkpoint_paths: list[str]) -> None:
        """Load expert weights from checkpoint files and pass them to the algorithm."""
        if not hasattr(self.alg, "load_experts"):
            raise TypeError(f"{type(self.alg).__name__} does not support expert loading.")

        experts = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
            experts.append(self._extract_expert_state_dict(checkpoint))
        self.alg.load_experts(experts)

    @staticmethod
    def _extract_expert_state_dict(checkpoint: dict) -> dict:
        """Extract an expert state dict from common checkpoint formats."""
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint

        for key in ("expert_state_dict", "model_state_dict", "state_dict", "actor_state_dict"):
            if key in checkpoint:
                return checkpoint[key]

        raise KeyError(
            "Could not find an expert state dict. Expected a raw state dict or one of "
            "'expert_state_dict', 'model_state_dict', 'state_dict', or 'actor_state_dict'."
        )
