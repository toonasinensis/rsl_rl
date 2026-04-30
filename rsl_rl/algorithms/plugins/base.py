# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


class AuxLossPlugin(nn.Module):
    """Base class for auxiliary-loss plugins that extend the PPO training objective.

    Subclasses can add loss terms computed from:
    - Intermediate representations exposed in ``forward_results`` (e.g. ``actor_sub_latent``)
    - Stored rollout observations and transitions available in the mini-batch

    Lifecycle
    ---------
    1. Plugin is instantiated from config via :func:`~rsl_rl.utils.instantiate_from_config`.
    2. :meth:`setup` is called once after the actor is constructed — use it to query actor
       structure (e.g. ``backbone.encoder_latent_dim``) and build any submodules that depend
       on those shapes.
    3. During each mini-batch update, :meth:`compute_loss` is called with the actor/critic
       forward results and the current mini-batch.  Return a dict of named loss tensors;
       :class:`~rsl_rl.algorithms.PPO` sums them into the total loss.
    4. Optionally override :meth:`extra_metrics` to expose additional scalars for logging.
    """

    def setup(self, actor: nn.Module, obs: TensorDict) -> None:
        """Post-construction setup with access to the actor and a sample observation dict.

        Called once by :meth:`~rsl_rl.algorithms.PPO.construct_algorithm` after the actor
        is built.  Use this hook to:

        - Query ``actor.backbone.encoder_latent_dim`` (or similar) to determine input sizes.
        - Register decoder / projection submodules whose shapes depend on actor internals.

        Args:
            actor: The constructed actor model (typically :class:`~rsl_rl.models.ActorModel`).
            obs: A sample observation TensorDict from the environment (used for shape info).
        """
        pass

    def compute_loss(
        self, forward_results: dict[str, torch.Tensor | None], batch: object
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary loss terms for the current mini-batch.

        Args:
            forward_results: Dict populated by :meth:`~rsl_rl.algorithms.PPO._forward_model`.
                At minimum contains:
                - ``"actions_log_prob"`` – log-probabilities under the current policy
                - ``"values"`` – critic value estimates
                - ``"distribution_params"`` – current distribution parameters
                - ``"entropy"`` – policy entropy
                - ``"actor_sub_latent"`` – concatenated per-group encodings from
                  :class:`~rsl_rl.models.SubEncoderMLPModel` (``None`` for plain MLPModel)
            batch: The current mini-batch storage object with attributes like
                ``batch.observations`` (TensorDict) and ``batch.actions``.

        Returns:
            Dict mapping loss name → scalar tensor.  All values are added to the PPO loss.
            Return an empty dict to contribute no loss.
        """
        return {}

    def extra_metrics(self) -> dict[str, float]:
        """Return additional scalar metrics to include in the training log.

        Called after each :meth:`compute_loss`; values are accumulated and averaged over
        the update epoch.  Default implementation returns an empty dict.
        """
        return {}
