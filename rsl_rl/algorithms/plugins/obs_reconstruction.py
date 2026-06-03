# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import MLP
from rsl_rl.algorithms.plugins.base import AuxLossPlugin


class ObsReconstructionPlugin(AuxLossPlugin):
    """Auxiliary loss plugin that reconstructs observations from the actor's sub-latent.

    A separate decoder MLP is built for each target observation group.  During each
    training mini-batch, the decoder reconstructs the target observations from the
    ``actor_sub_latent`` (the concatenated per-group encodings produced by
    :class:`~rsl_rl.models.SubEncoderMLPModel`), and an MSE reconstruction loss is added
    to the PPO objective.

    This plugin **requires** the actor backbone to be a
    :class:`~rsl_rl.models.SubEncoderMLPModel` (or any backbone that exposes an
    ``encoder_latent_dim`` attribute and stores ``sub_latent`` after each forward pass).

    Configuration example::

        "plugins": [
            {
                "class_name": "rsl_rl.algorithms.plugins:ObsReconstructionPlugin",
                "target_groups": ["policy"],
                "hidden_dims": [256, 128],
                "weight": 0.01,
                "detach_latent": true,
            }
        ]

    Args:
        target_groups: Observation group keys to reconstruct (must be present in the
            environment observation TensorDict).
        obs: Sample observation TensorDict — used in :meth:`setup` to read group dims.
        hidden_dims: Hidden layer sizes for each decoder MLP.
        weight: Scalar multiplier applied to the total reconstruction loss before it is
            added to the PPO loss.
        detach_latent: If ``True`` (default), detach the sub-latent before passing it to
            the decoder so that decoder gradients do **not** flow back into the encoders.
            Set to ``False`` to jointly train encoders and decoder.
        activation: Activation function name for the decoder MLPs.
    """

    def __init__(
        self,
        target_groups: list[str],
        obs: TensorDict | None = None,
        hidden_dims: list[int] = (256, 128),
        weight: float = 1.0,
        detach_latent: bool = True,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.target_groups = list(target_groups)
        self.weight = weight
        self.detach_latent = detach_latent
        self.hidden_dims = list(hidden_dims)
        self.activation = activation

        # Store raw obs dims so setup() can build the decoders
        self._obs_dims: dict[str, int] = {}
        if obs is not None:
            self._obs_dims = {g: obs[g].shape[-1] for g in self.target_groups}

        # Decoder MLPs — built lazily in setup() once the actor latent dim is known
        self.decoders: nn.ModuleDict | None = None

    def setup(self, actor: nn.Module, obs: TensorDict) -> None:
        """Build decoder MLPs using the backbone's ``encoder_latent_dim``.

        Args:
            actor: Constructed actor (expected to have ``actor.backbone.encoder_latent_dim``).
            obs: Sample observation TensorDict used to resolve target dimensions when they
                were not provided at construction time.

        Raises:
            ValueError: If the actor backbone does not expose ``encoder_latent_dim``
                (i.e. is not a :class:`~rsl_rl.models.SubEncoderMLPModel`).
        """
        backbone = getattr(actor, "backbone", None)
        latent_dim: int | None = getattr(backbone, "encoder_latent_dim", None)
        if latent_dim is None:
            raise ValueError(
                "ObsReconstructionPlugin requires the actor backbone to expose "
                "'encoder_latent_dim' (e.g. SubEncoderMLPModel). "
                f"Got backbone type: {type(backbone).__name__}."
            )
        if not self._obs_dims:
            self._obs_dims = {g: obs[g].shape[-1] for g in self.target_groups}
        self.decoders = nn.ModuleDict(
            {
                g: MLP(latent_dim, obs_dim, self.hidden_dims, self.activation)
                for g, obs_dim in self._obs_dims.items()
            }
        )

    def compute_loss(
        self, forward_results: dict[str, torch.Tensor | None], batch: object
    ) -> dict[str, torch.Tensor]:
        """Compute MSE reconstruction loss for all target observation groups.

        Args:
            forward_results: Must contain ``"actor_sub_latent"`` (set by PPO._forward_model).
            batch: Mini-batch with ``batch.observations`` TensorDict.

        Returns:
            ``{"obs_reconstruction_loss": weighted_total_mse}`` or ``{}`` if setup has not
            been called yet or sub-latent is unavailable.
        """
        if self.decoders is None:
            return {}

        sub_latent: torch.Tensor | None = forward_results.get("actor_sub_latent")  # type: ignore[assignment]
        if sub_latent is None:
            return {}

        latent = sub_latent.detach() if self.detach_latent else sub_latent

        total_loss: torch.Tensor = torch.tensor(0.0, device=latent.device)
        for g, decoder in self.decoders.items():
            target = batch.observations[g]  # type: ignore[union-attr]
            reconstructed = decoder(latent)
            total_loss = total_loss + F.mse_loss(reconstructed, target)

        return {"obs_reconstruction_loss": total_loss * self.weight}
