# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.algorithms import AMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class AMPOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for PPO with adversarial motion prior training."""

    alg: AMPPPO

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner and reuse the shared AMP expert-data injection path."""
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
