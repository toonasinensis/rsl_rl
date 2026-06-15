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


class FakePPO:
    device = "cpu"
    num_learning_epochs = 1
    num_mini_batches = 1

    class Storage:
        num_envs = 2
        num_transitions_per_env = 1

    storage = Storage()


class FakeUpdateStorage:
    def __init__(self, samples: tuple[torch.Tensor, torch.Tensor], num_samples: int = 1) -> None:
        self.samples = samples
        self.num_samples = num_samples

    def feed_forward_generator(self, _num_mini_batch: int, _mini_batch_size: int):
        yield self.samples


class FakeExpertData:
    def __init__(self, samples: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.samples = samples

    def feed_forward_generator(self, _num_mini_batch: int, _mini_batch_size: int):
        yield self.samples


class FakeUpdateDiscriminator:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True)

    def compute_loss(self, _policy_d, _expert_d, sample_amp_expert):
        exp_s, exp_ns = sample_amp_expert
        return exp_s.sum() * 0.0 + torch.tensor(2.0), exp_ns.sum() * 0.0 + torch.tensor(3.0)


class RecordingNormalizer:
    def __init__(self) -> None:
        self.updated: torch.Tensor | None = None

    def normalize_torch(self, x: torch.Tensor, _device: str):
        return x + 1000.0

    def update_with_tensors(self, *tensors: torch.Tensor) -> None:
        self.updated = torch.cat(tensors, dim=0).clone()


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


def test_amp_plugin_rejects_missing_required_runtime_config() -> None:
    plugin = AMPPlugin(amp_reward_coef=1.0, amp_discr_hidden_dims=[8])

    with pytest.raises(ValueError, match="amp_motion_files"):
        plugin.on_init(ppo=object(), env=object())


def test_amp_plugin_skips_update_when_policy_buffer_empty() -> None:
    plugin = AMPPlugin(amp_reward_coef=1.0, amp_discr_hidden_dims=[8])
    plugin.amp_storage = FakeUpdateStorage((torch.zeros(1, 2), torch.zeros(1, 2)), num_samples=0)
    plugin.amp_data = FakeExpertData((torch.zeros(1, 2), torch.zeros(1, 2)))

    plugin.on_update_start(FakePPO())

    assert plugin.on_per_batch_extra_loss(FakePPO(), _batch=None) == {}


def test_amp_plugin_scales_loss_and_updates_normalizer_with_raw_states() -> None:
    pol_s = torch.tensor([[1.0, 2.0]])
    pol_ns = torch.tensor([[3.0, 4.0]])
    exp_s = torch.tensor([[5.0, 6.0]])
    exp_ns = torch.tensor([[7.0, 8.0]])
    normalizer = RecordingNormalizer()
    plugin = AMPPlugin(amp_reward_coef=1.0, amp_discr_hidden_dims=[8], amp_loss_coef=0.25)
    plugin.discriminator = FakeUpdateDiscriminator()
    plugin.amp_normalizer = normalizer
    plugin.amp_storage = FakeUpdateStorage((pol_s, pol_ns), num_samples=1)
    plugin.amp_data = FakeExpertData((exp_s, exp_ns))

    plugin.on_update_start(FakePPO())
    losses = plugin.on_per_batch_extra_loss(FakePPO(), _batch=None)

    assert torch.equal(losses["amp"], torch.tensor(0.5))
    assert torch.equal(losses["amp_grad_pen"], torch.tensor(0.75))
    assert torch.equal(normalizer.updated, torch.cat([pol_s, pol_ns, exp_s, exp_ns], dim=0))
