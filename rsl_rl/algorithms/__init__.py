# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .configurable_distillation import ConfigurableDistillation
from .distillation import Distillation
from .latent_residual_ppo import LatentResidualPPO
from .ppo import PPO
from .plugins import AMPPlugin, PPOPlugin, TeacherKLPlugin

__all__ = [
    "PPO",
    "Distillation",
    "ConfigurableDistillation",
    "LatentResidualPPO",
    "AMPPlugin",
    "PPOPlugin",
    "TeacherKLPlugin",
]
