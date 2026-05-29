# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl.algorithms import SMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class SMPOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for PPO with score matching prior training."""

    alg: SMPPPO

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner and attach SMP expert data providers."""
        smp_runner_cfg = train_cfg.get("smp_runner", {})
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        expert_data = smp_runner_cfg.get("expert_data")
        if expert_data is None and smp_runner_cfg.get("expert_data_path") is not None:
            expert_data = torch.load(smp_runner_cfg["expert_data_path"], weights_only=False, map_location=device)
        if expert_data is None and hasattr(env, "get_smp_expert_observations"):
            expert_data = env.get_smp_expert_observations()
        if expert_data is not None:
            self.alg.set_smp_expert_data(expert_data)

        if hasattr(env, "sample_smp_expert_observations"):
            self.alg.set_smp_expert_sampler(env.sample_smp_expert_observations)
