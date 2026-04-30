# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auxiliary-loss plugins for extending the PPO training objective."""

from .base import AuxLossPlugin
from .obs_reconstruction import ObsReconstructionPlugin

__all__ = [
    "AuxLossPlugin",
    "ObsReconstructionPlugin",
]
