# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator, Iterable
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.utils import split_and_pad_trajectories


class _Record:
    """Small mapping wrapper that also supports attribute access.

    New code should prefer ``record["field"]`` while existing algorithm code can
    continue to use ``record.field`` during the gradual storage migration.
    """

    def __init__(self, include_none: bool = True, **fields) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in fields.items():
            if include_none or value is not None:
                self._data[key] = value

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattribute__(self, name: str):
        if not name.startswith("_"):
            try:
                data = object.__getattribute__(self, "_data")
            except AttributeError:
                data = None
            if data is not None and name in data:
                return data[name]
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def clear(self) -> None:
        self._data.clear()


class RolloutStorage:
    """Storage for rollout data with dict-backed tensor fields.

    The public compatibility attributes (for example ``storage.actions`` and
    ``batch.actions``) are aliases over mapping entries. New algorithm code can
    register and consume arbitrary tensor fields via mapping keys.
    """

    class Transition(_Record):
        """Storage for one environment transition before insertion."""

        def __init__(self) -> None:
            super().__init__(include_none=False)
            self.hidden_states = (None, None)

    class Batch(_Record):
        """A mini-batch yielded by storage generators."""

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            returns: torch.Tensor | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState, HiddenState] = (None, None),
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
            **extra_fields,
        ) -> None:
            super().__init__(
                include_none=True,
                observations=observations,
                actions=actions,
                values=values,
                advantages=advantages,
                returns=returns,
                old_actions_log_prob=old_actions_log_prob,
                old_distribution_params=old_distribution_params,
                hidden_states=hidden_states,
                masks=masks,
                privileged_actions=privileged_actions,
                dones=dones,
                **extra_fields,
            )

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate rollout buffers for a specific training mode and batch shape."""
        object.__setattr__(self, "training_type", training_type)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_transitions_per_env", num_transitions_per_env)
        object.__setattr__(self, "num_envs", num_envs)
        object.__setattr__(self, "actions_shape", tuple(actions_shape))

        data = TensorDict({}, batch_size=[num_transitions_per_env, num_envs], device=device)
        data["observations"] = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "_tuple_data", {})

        self.ensure_field("rewards", (1,))
        self.ensure_field("dones", (1,), dtype=torch.uint8)
        self.ensure_field("actions", tuple(actions_shape))

        if training_type == "distillation":
            self.ensure_field("privileged_actions", tuple(actions_shape))
        elif training_type == "rl":
            self.ensure_field("values", (1,))
            self.ensure_field("actions_log_prob", (1,))
            self.ensure_field("returns", (1,))
            self.ensure_field("advantages", (1,))

        object.__setattr__(self, "saved_hidden_state_a", None)
        object.__setattr__(self, "saved_hidden_state_c", None)
        object.__setattr__(self, "step", 0)

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data.keys() or key in self._tuple_data

    def __getattr__(self, name: str):
        if "data" in self.__dict__ and name in self.data.keys():
            return self.data[name]
        if "_tuple_data" in self.__dict__ and name in self._tuple_data:
            return self._tuple_data[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in {
            "training_type",
            "device",
            "num_transitions_per_env",
            "num_envs",
            "actions_shape",
            "data",
            "_tuple_data",
            "saved_hidden_state_a",
            "saved_hidden_state_c",
            "step",
        }:
            object.__setattr__(self, name, value)
        elif "data" in self.__dict__ and name in self.data.keys():
            self.data[name] = value
        elif "_tuple_data" in self.__dict__ and name in self._tuple_data:
            self._tuple_data[name] = value
        else:
            object.__setattr__(self, name, value)

    def ensure_field(
        self,
        name: str,
        shape: Iterable[int] | torch.Size,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Allocate a rollout tensor field if it does not already exist."""
        if name in self.data.keys():
            return self.data[name]
        field_shape = tuple(shape)
        self.data[name] = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            *field_shape,
            dtype=dtype,
            device=self.device,
        )
        return self.data[name]

    def _ensure_field_from_value(self, name: str, value) -> None:
        """Lazy-allocate a field that matches a transition value."""
        if name in self.data.keys():
            return
        if isinstance(value, TensorDict):
            self.data[name] = TensorDict(
                {
                    key: torch.zeros(
                        self.num_transitions_per_env,
                        *tensor.shape,
                        dtype=tensor.dtype,
                        device=self.device,
                    )
                    for key, tensor in value.items()
                },
                batch_size=[self.num_transitions_per_env, *value.batch_size],
                device=self.device,
            )
        elif isinstance(value, torch.Tensor):
            self.data[name] = torch.zeros(
                self.num_transitions_per_env,
                *value.shape,
                dtype=value.dtype,
                device=self.device,
            )
        else:
            raise TypeError(f"Unsupported rollout field type for {name!r}: {type(value)!r}")

    @staticmethod
    def _reshape_for_target(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if value.shape == target.shape:
            return value
        return value.reshape_as(target)

    def _copy_field(self, name: str, value) -> None:
        self._ensure_field_from_value(name, value)
        target = self.data[name][self.step]
        if isinstance(value, TensorDict):
            target.copy_(value)
        else:
            target.copy_(self._reshape_for_target(value, target))

    def _copy_tuple_field(self, name: str, values: tuple[torch.Tensor, ...]) -> None:
        if name not in self._tuple_data:
            self._tuple_data[name] = tuple(
                torch.zeros(self.num_transitions_per_env, *value.shape, dtype=value.dtype, device=self.device)
                for value in values
            )
        if len(self._tuple_data[name]) != len(values):
            raise ValueError(
                f"Tuple rollout field {name!r} changed length from {len(self._tuple_data[name])} to {len(values)}."
            )
        for target, value in zip(self._tuple_data[name], values):
            target[self.step].copy_(value)

    def add_transition(self, transition: Transition) -> None:
        """Add one transition to the storage at the current step index."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        for name, value in transition.items():
            if name == "hidden_states" or value is None:
                continue
            if isinstance(value, tuple):
                self._copy_tuple_field(name, value)
            else:
                self._copy_field(name, value)

        self._save_hidden_states(transition.get("hidden_states", (None, None)))
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0

    def _batch_at_step(self, step: int) -> Batch:
        fields = {key: self.data[key][step] for key in self.data.keys()}
        return RolloutStorage.Batch(**fields)

    def generator(self) -> Generator[Batch, None, None]:
        """Yield per-timestep batches for distillation training."""
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")
        if "privileged_actions" not in self.data.keys():
            raise ValueError("Distillation storage requires a 'privileged_actions' field.")

        for step in range(self.num_transitions_per_env):
            yield self._batch_at_step(step)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches for feedforward RL updates."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.data["observations"].flatten(0, 1)
        actions = self.data["actions"].flatten(0, 1)
        values = self.data["values"].flatten(0, 1)
        returns = self.data["returns"].flatten(0, 1)
        old_actions_log_prob = self.data["actions_log_prob"].flatten(0, 1)
        advantages = self.data["advantages"].flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self._tuple_data["distribution_params"])

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield trajectory mini-batches with masks and recurrent hidden states."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.data["observations"], self.data["dones"]
        )
        mini_batch_size = self.num_envs // num_mini_batches

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.data["dones"].squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                last_was_done = last_was_done.permute(1, 0)
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    hidden_state_a_batch = (
                        hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                    )
                else:
                    hidden_state_a_batch = None
                if self.saved_hidden_state_c is not None:
                    hidden_state_c_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_c
                    ]
                    hidden_state_c_batch = (
                        hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                    )
                else:
                    hidden_state_c_batch = None

                yield RolloutStorage.Batch(
                    observations=padded_obs_trajectories[:, first_traj:last_traj],
                    actions=self.data["actions"][:, start:stop],
                    values=self.data["values"][:, start:stop],
                    advantages=self.data["advantages"][:, start:stop],
                    returns=self.data["returns"][:, start:stop],
                    old_actions_log_prob=self.data["actions_log_prob"][:, start:stop],
                    old_distribution_params=tuple(
                        p[:, start:stop] for p in self._tuple_data["distribution_params"]
                    ),
                    hidden_states=(hidden_state_a_batch, hidden_state_c_batch),
                    masks=trajectory_masks[:, first_traj:last_traj],
                )

                first_traj = last_traj

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        """Save recurrent hidden states to the rollout storage."""
        if hidden_states == (None, None):
            return
        if hidden_states[0] is not None:
            hidden_state_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        if hidden_states[1] is not None:
            hidden_state_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        if self.saved_hidden_state_a is None and hidden_states[0] is not None:
            self.saved_hidden_state_a = [
                torch.zeros(self.data["observations"].shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
        if self.saved_hidden_state_c is None and hidden_states[1] is not None:
            self.saved_hidden_state_c = [
                torch.zeros(self.data["observations"].shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]

        if hidden_states[0] is not None:
            for i in range(len(hidden_state_a)):
                self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])
        if hidden_states[1] is not None:
            for i in range(len(hidden_state_c)):
                self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])
