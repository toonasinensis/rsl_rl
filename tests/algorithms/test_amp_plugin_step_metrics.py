from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.plugins.amp_plugins import AMPPlugin


class FakeDiscriminator:
    def predict_amp_reward_components(self, _state, _next_state, _task_reward, normalizer=None):
        assert normalizer is None
        components = {
            "task_reward": torch.tensor([1.0, 2.0]),
            "amp_reward": torch.tensor([10.0, 20.0]),
            "mixed_reward": torch.tensor([4.0, 8.0]),
        }
        return components, torch.zeros(2, 1)


class FakeStorage:
    def __init__(self) -> None:
        self.inserted: tuple[torch.Tensor, torch.Tensor] | None = None

    def insert(self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor) -> None:
        self.inserted = (amp_obs.clone(), next_amp_obs.clone())


def test_amp_plugin_inserts_reward_components_as_step_metrics() -> None:
    plugin = AMPPlugin(amp_reward_coef=1.0, amp_discr_hidden_dims=[8])
    plugin.discriminator = FakeDiscriminator()
    plugin.amp_storage = FakeStorage()
    plugin.amp_normalizer = None
    plugin._current_amp_obs = torch.tensor([[1.0, 1.5], [2.0, 2.5]])
    next_amp_obs = torch.tensor([[3.0, 3.5], [4.0, 4.5]])
    obs = TensorDict({"amp": next_amp_obs.clone()}, batch_size=[2])
    extras: dict = {}

    mixed_reward = plugin.on_after_step(
        _runner=None,
        obs=obs,
        rewards=torch.tensor([0.5, 1.5]),
        dones=torch.tensor([0.0, 1.0]),
        extras=extras,
    )

    assert torch.equal(mixed_reward, torch.tensor([4.0, 8.0]))
    assert set(extras["step_metrics"]) == {"task_reward", "amp_reward", "mixed_reward"}
    assert torch.equal(extras["step_metrics"]["task_reward"], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(extras["step_metrics"]["amp_reward"], torch.tensor([[10.0], [20.0]]))
    assert torch.equal(extras["step_metrics"]["mixed_reward"], torch.tensor([[4.0], [8.0]]))

    inserted_current, inserted_next = plugin.amp_storage.inserted
    assert torch.equal(inserted_current, torch.tensor([[1.0, 1.5], [2.0, 2.5]]))
    assert torch.equal(inserted_next, torch.tensor([[3.0, 3.5], [2.0, 2.5]]))


def test_amp_plugin_rejects_legacy_policy_std_clamp_config() -> None:
    with pytest.raises(ValueError, match="min_policy_std"):
        AMPPlugin(amp_reward_coef=1.0, amp_discr_hidden_dims=[8], min_normalized_std=0.05)
