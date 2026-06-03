# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict import TensorDict


class AMPExpertProvider:
    """Base abstraction for sourcing AMP expert and policy transition data."""

    def setup(self, env, device: str, obs: TensorDict | None = None) -> None:
        """Initialize provider state with access to the environment and target device."""
        del env, device, obs

    def record_policy_transition(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Record one batch of policy-side AMP transitions."""
        del amp_obs, next_amp_obs, dones

    def sample_expert_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample expert AMP observation pairs."""
        raise NotImplementedError

    def sample_policy_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample policy AMP observation pairs if local replay is enabled."""
        del batch_size
        return None

    def set_expert_data(
        self,
        expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor,
    ) -> None:
        """Set fixed expert data when the provider supports external injection."""
        del expert_data
        raise NotImplementedError(f"{type(self).__name__} does not support set_expert_data().")

    def set_expert_sampler(
        self,
        sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor],
    ) -> None:
        """Set an expert sampler when the provider supports external injection."""
        del sampler
        raise NotImplementedError(f"{type(self).__name__} does not support set_expert_sampler().")

    def state_dict(self) -> dict:
        """Return serializable provider state."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore provider state."""
        del state_dict


class ExternalAMPProvider(AMPExpertProvider):
    """AMP provider that consumes externally supplied expert data or samplers."""

    def __init__(
        self,
        expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor | None = None,
        expert_sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor] | None = None,
        amp_key: str = "amp",
        next_amp_key: str = "next_amp",
        enable_policy_replay: bool = False,
        replay_buffer_size: int = 100_000,
    ) -> None:
        self.device = "cpu"
        self.amp_key = amp_key
        self.next_amp_key = next_amp_key
        self.enable_policy_replay = enable_policy_replay
        self.replay_buffer_size = replay_buffer_size

        self.expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor | None = None
        self.expert_sampler = expert_sampler
        self.policy_states: torch.Tensor | None = None
        self.policy_next_states: torch.Tensor | None = None
        self.policy_num_samples = 0

        if expert_data is not None:
            self.set_expert_data(expert_data)

    def setup(self, env, device: str, obs: TensorDict | None = None) -> None:
        del env, obs
        self.device = device
        if self.expert_data is not None:
            self.expert_data = self._normalize_expert_source(self.expert_data)
        if self.policy_states is not None:
            self.policy_states = self.policy_states.to(device)
        if self.policy_next_states is not None:
            self.policy_next_states = self.policy_next_states.to(device)

    def set_expert_data(
        self,
        expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor,
    ) -> None:
        """Set fixed expert AMP data used for batch sampling."""
        self.expert_data = self._normalize_expert_source(expert_data)

    def set_expert_sampler(
        self,
        sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor],
    ) -> None:
        """Set a callable source for expert AMP batches."""
        self.expert_sampler = sampler

    def record_policy_transition(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        del dones
        if not self.enable_policy_replay:
            return

        amp_obs = amp_obs.detach().to(self.device).reshape(-1, amp_obs.shape[-1])
        next_amp_obs = next_amp_obs.detach().to(self.device).reshape(-1, next_amp_obs.shape[-1])

        if self.policy_states is None or self.policy_next_states is None:
            self.policy_states = amp_obs[-self.replay_buffer_size :].clone()
            self.policy_next_states = next_amp_obs[-self.replay_buffer_size :].clone()
            self.policy_num_samples = self.policy_states.shape[0]
            return

        self.policy_states = torch.cat((self.policy_states, amp_obs), dim=0)[-self.replay_buffer_size :]
        self.policy_next_states = torch.cat((self.policy_next_states, next_amp_obs), dim=0)[-self.replay_buffer_size :]
        self.policy_num_samples = self.policy_states.shape[0]

    def sample_expert_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample expert AMP observation pairs from fixed data or a sampler."""
        expert_source = self.expert_sampler(batch_size) if self.expert_sampler is not None else self.expert_data
        if expert_source is None:
            raise RuntimeError("ExternalAMPProvider requires expert_data or expert_sampler before sampling.")

        expert_source = self._normalize_expert_source(expert_source)
        current, next_current = self._extract_pairs(expert_source)
        indices = torch.randint(current.shape[0], (batch_size,), device=current.device)
        return current[indices], next_current[indices]

    def sample_policy_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample locally recorded policy AMP transitions."""
        if not self.enable_policy_replay or self.policy_states is None or self.policy_next_states is None:
            return None
        indices = torch.randint(self.policy_states.shape[0], (batch_size,), device=self.policy_states.device)
        return self.policy_states[indices], self.policy_next_states[indices]

    def state_dict(self) -> dict:
        """Return provider state excluding non-serializable sampler callables."""
        state = {
            "amp_key": self.amp_key,
            "next_amp_key": self.next_amp_key,
            "enable_policy_replay": self.enable_policy_replay,
            "replay_buffer_size": self.replay_buffer_size,
            "expert_data": self._clone_source(self.expert_data),
            "policy_states": None if self.policy_states is None else self.policy_states.clone(),
            "policy_next_states": None if self.policy_next_states is None else self.policy_next_states.clone(),
            "policy_num_samples": self.policy_num_samples,
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore provider state."""
        self.amp_key = state_dict.get("amp_key", self.amp_key)
        self.next_amp_key = state_dict.get("next_amp_key", self.next_amp_key)
        self.enable_policy_replay = state_dict.get("enable_policy_replay", self.enable_policy_replay)
        self.replay_buffer_size = state_dict.get("replay_buffer_size", self.replay_buffer_size)
        self.expert_data = self._clone_source(state_dict.get("expert_data"))
        self.policy_states = state_dict.get("policy_states")
        self.policy_next_states = state_dict.get("policy_next_states")
        self.policy_num_samples = state_dict.get("policy_num_samples", 0)
        if self.expert_data is not None:
            self.expert_data = self._normalize_expert_source(self.expert_data)
        if self.policy_states is not None:
            self.policy_states = self.policy_states.to(self.device)
        if self.policy_next_states is not None:
            self.policy_next_states = self.policy_next_states.to(self.device)

    def _normalize_expert_source(
        self,
        source: TensorDict | dict[str, torch.Tensor] | torch.Tensor,
    ) -> TensorDict | dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(source, torch.Tensor):
            return source.to(self.device)
        if isinstance(source, TensorDict):
            normalized = source.to(self.device)
            if len(normalized.batch_size) > 1:
                normalized = normalized.flatten(0, len(normalized.batch_size) - 1)
            return normalized

        normalized_dict = {key: value.to(self.device) for key, value in source.items()}
        first = next(iter(normalized_dict.values()))
        if first.ndim > 2:
            normalized_dict = {
                key: value.reshape(-1, value.shape[-1])
                for key, value in normalized_dict.items()
            }
        return normalized_dict

    def _extract_pairs(
        self,
        source: TensorDict | dict[str, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(source, torch.Tensor):
            current = source.reshape(-1, source.shape[-1])
            return current, current

        if isinstance(source, TensorDict):
            keys = list(source.keys())
            current_key = self.amp_key if self.amp_key in source.keys() else keys[0]
            next_key = self.next_amp_key if self.next_amp_key in source.keys() else current_key
            current = source[current_key].reshape(-1, source[current_key].shape[-1])
            next_current = source[next_key].reshape(-1, source[next_key].shape[-1])
            return current, next_current

        keys = list(source.keys())
        current_key = self.amp_key if self.amp_key in source else keys[0]
        next_key = self.next_amp_key if self.next_amp_key in source else current_key
        current = source[current_key].reshape(-1, source[current_key].shape[-1])
        next_current = source[next_key].reshape(-1, source[next_key].shape[-1])
        return current, next_current

    @staticmethod
    def _clone_source(source):
        if source is None:
            return None
        if isinstance(source, torch.Tensor):
            return source.clone()
        if isinstance(source, TensorDict):
            return source.clone()
        return {key: value.clone() for key, value in source.items()}
