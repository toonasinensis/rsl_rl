from __future__ import annotations

from collections import deque

import torch

from rsl_rl.utils.logger import Logger


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, scalar_value, global_step: int | None = None, *_, **__) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step or 0)))


def _make_logger(num_envs: int = 2) -> Logger:
    logger = Logger(
        log_dir=None,
        cfg={"num_steps_per_env": 1, "algorithm": {}},
        env_cfg={},
        num_envs=num_envs,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.writer = FakeWriter()
    logger.logger_type = "tensorboard"
    return logger


def test_process_env_step_accumulates_arbitrary_step_metrics() -> None:
    logger = _make_logger()

    logger.process_env_step(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        extras={"step_metrics": {"task_reward": torch.tensor([[10.0], [20.0]])}},
    )

    assert list(logger.metric_buffers) == ["reward", "task_reward"]
    assert list(logger.metric_buffers["reward"]) == [2.0]
    assert list(logger.metric_buffers["task_reward"]) == [20.0]
    assert torch.allclose(logger.cur_metric_sums["reward"], torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(logger.cur_metric_sums["task_reward"], torch.tensor([[10.0], [0.0]]))

    logger.process_env_step(
        rewards=torch.tensor([3.0, 4.0]),
        dones=torch.tensor([1.0, 0.0]),
        extras={"step_metrics": {"task_reward": torch.tensor([[30.0], [40.0]])}},
    )

    assert list(logger.metric_buffers["reward"]) == [2.0, 4.0]
    assert list(logger.metric_buffers["task_reward"]) == [20.0, 40.0]
    assert list(logger.lenbuffer) == [1.0, 2.0]


def test_log_writes_train_tags_for_discovered_metrics() -> None:
    logger = _make_logger()
    writer = logger.writer

    logger.process_env_step(
        rewards=torch.tensor([1.0, 3.0]),
        dones=torch.tensor([1.0, 1.0]),
        extras={"step_metrics": {"amp_reward": torch.tensor([5.0, 7.0])}},
    )
    logger.log(
        it=3,
        start_it=0,
        total_it=4,
        collect_time=1.0,
        learn_time=1.0,
        loss_dict={},
        learning_rate=1e-3,
        action_std=torch.ones(1),
        rnd_weight=None,
        print_minimal=True,
    )

    logged = {tag: value for tag, value, _step in writer.scalars}
    assert logged["Train/mean_reward"] == 2.0
    assert logged["Train/mean_amp_reward"] == 6.0
    assert logged["Train/mean_reward/time"] == 2.0
    assert logged["Train/mean_amp_reward/time"] == 6.0


def test_without_step_metrics_only_reward_buffer_is_created() -> None:
    logger = _make_logger()

    logger.process_env_step(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([1.0, 1.0]),
        extras={},
    )

    assert list(logger.metric_buffers) == ["reward"]
    assert list(logger.rewbuffer) == [1.0, 2.0]


def test_intrinsic_rewards_are_included_in_default_reward_metric() -> None:
    logger = _make_logger()
    logger.rnd_enabled = True
    logger.erewbuffer = deque(maxlen=100)
    logger.irewbuffer = deque(maxlen=100)
    logger.cur_ereward_sum = torch.zeros(logger.num_envs, dtype=torch.float)
    logger.cur_ireward_sum = torch.zeros(logger.num_envs, dtype=torch.float)

    logger.process_env_step(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([1.0, 1.0]),
        extras={},
        intrinsic_rewards=torch.tensor([0.5, 0.25]),
    )

    assert list(logger.rewbuffer) == [1.5, 2.25]
    assert list(logger.erewbuffer) == [1.0, 2.0]
    assert list(logger.irewbuffer) == [0.5, 0.25]
