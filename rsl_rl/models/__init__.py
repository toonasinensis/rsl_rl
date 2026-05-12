# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""
from .backbone_base import BaseModel
from .backbone_fsq import BackboneFSQ
from .backbone_mlp import BackboneMLP
from .backbone_cnn import BackboneCNN
from .backbone_rnn import BackboneRNN
from .wrapper_stochastic import StochasticWrapper


CNNModel = BackboneCNN
MLPModel = BackboneMLP
RNNModel = BackboneRNN
FSQModel = BackboneFSQ
class ActorModel(StochasticWrapper):
    """Backward-compatible name for the stochastic actor wrapper."""


__all__ = [
    "BaseModel",

    "BackboneCNN",
    "BackboneMLP",
    "BackboneRNN",
    "BackboneFSQ",
    "CNNModel",
    "MLPModel",
    "RNNModel",
    "FSQModel",

    "StochasticWrapper",
    "ActorModel",
]


""" Hierarchy Structure of the built models
1. StochasticWrapper: wrapper of NNs for adding stochastic layer
2. BackboneModel: adding normalization layer to backbone models and defining functions remains to be implemented
3. BaseModel: NNs backbone for actor and critics
"""
