from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.plugins.teacher_kl_plugin import TeacherKLPlugin
from rsl_rl.utils import construct_actor_with_shell


class _TinyEnv:
    num_actions = 3


class _TinyStorage:
    def __init__(self, obs: TensorDict) -> None:
        self.observations = TensorDict(
            {key: value.unsqueeze(0).clone() for key, value in obs.items()},
            batch_size=[1, obs.batch_size[0]],
        )


class _TinyPPO:
    def __init__(self, actor, obs: TensorDict) -> None:
        self.actor = actor
        self.storage = _TinyStorage(obs)
        self.device = "cpu"


class _Batch:
    def __init__(self, obs: TensorDict) -> None:
        self.observations = obs


def _actor_cfg(hidden_dims: list[int] | None = None) -> dict:
    return {
        "class_name": "ActorModel",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        "backbone": {
            "class_name": "MyMLPModel",
            "hidden_dims": hidden_dims or [16],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def _make_obs(num_envs: int = 4) -> TensorDict:
    return TensorDict(
        {
            "student": torch.randn(num_envs, 8),
            "teacher": torch.randn(num_envs, 10),
            "teacher_runtime": torch.randn(num_envs, 10),
        },
        batch_size=[num_envs],
    )


def _make_actor(obs: TensorDict, obs_groups: dict[str, list[str]], cfg: dict | None = None):
    return construct_actor_with_shell(obs, obs_groups, cfg or _actor_cfg(), 3)


def _make_teacher_checkpoint(path, obs: TensorDict, cfg: dict | None = None) -> None:
    teacher = _make_actor(obs, {"actor": ["teacher"]}, cfg)
    torch.save({"actor_state_dict": teacher.state_dict(), "iter": 123}, path)


def test_teacher_kl_plugin_loads_and_freezes_teacher(tmp_path) -> None:
    obs = _make_obs()
    teacher_cfg = _actor_cfg()
    ckpt_path = tmp_path / "teacher.pt"
    _make_teacher_checkpoint(ckpt_path, obs, teacher_cfg)

    ppo = _TinyPPO(_make_actor(obs, {"actor": ["student"]}), obs)
    plugin = TeacherKLPlugin(
        teacher_checkpoint_path=str(ckpt_path),
        teacher_actor=teacher_cfg,
        teacher_obs_groups={"actor": ["teacher"]},
    )

    plugin.on_init(ppo, _TinyEnv())

    assert plugin.teacher is not None
    assert not plugin.teacher.training
    assert all(not p.requires_grad for p in plugin.teacher.parameters())


def test_teacher_kl_loss_is_scaled_student_teacher_kl(tmp_path) -> None:
    obs = _make_obs()
    teacher_cfg = _actor_cfg()
    ckpt_path = tmp_path / "teacher.pt"
    _make_teacher_checkpoint(ckpt_path, obs, teacher_cfg)

    student = _make_actor(obs, {"actor": ["student"]})
    ppo = _TinyPPO(student, obs)
    plugin = TeacherKLPlugin(
        teacher_checkpoint_path=str(ckpt_path),
        teacher_actor=teacher_cfg,
        teacher_obs_groups={"actor": ["teacher"]},
        start_loss_coef=0.5,
        end_loss_coef=0.0,
        end_step=2000,
    )
    plugin.on_init(ppo, _TinyEnv())
    plugin.on_update_start(ppo)

    extra = plugin.on_per_batch_extra_loss(ppo, _Batch(obs))
    loss = extra["teacher_kl_loss"]

    student_params = student.output_distribution_params
    with torch.no_grad():
        plugin.teacher(obs, stochastic_output=False, train_mode=False)
        teacher_params = tuple(p.detach() for p in plugin.teacher.output_distribution_params)
        expected_raw_kl = student.get_kl_divergence(student_params, teacher_params).mean()

    assert loss.item() >= 0.0
    assert torch.allclose(loss.detach(), expected_raw_kl * 0.5)


def test_teacher_obs_alias_keeps_checkpoint_group_name(tmp_path) -> None:
    obs = _make_obs()
    teacher_cfg = _actor_cfg()
    ckpt_path = tmp_path / "teacher.pt"
    _make_teacher_checkpoint(ckpt_path, obs, teacher_cfg)

    ppo = _TinyPPO(_make_actor(obs, {"actor": ["student"]}), obs)
    plugin = TeacherKLPlugin(
        teacher_checkpoint_path=str(ckpt_path),
        teacher_actor=teacher_cfg,
        teacher_obs_groups={"actor": ["teacher"]},
        teacher_obs_aliases={"teacher": "teacher_runtime"},
    )
    plugin.on_init(ppo, _TinyEnv())
    plugin.on_update_start(ppo)

    extra = plugin.on_per_batch_extra_loss(ppo, _Batch(obs))

    assert extra["teacher_kl_loss"].item() >= 0.0


def test_teacher_kl_coef_decays_by_update_step(tmp_path) -> None:
    obs = _make_obs()
    ckpt_path = tmp_path / "teacher.pt"
    _make_teacher_checkpoint(ckpt_path, obs)

    ppo = _TinyPPO(_make_actor(obs, {"actor": ["student"]}), obs)
    plugin = TeacherKLPlugin(
        teacher_checkpoint_path=str(ckpt_path),
        teacher_actor=_actor_cfg(),
        teacher_obs_groups={"actor": ["teacher"]},
        start_loss_coef=0.5,
        end_loss_coef=0.0,
        end_step=2,
    )
    plugin.on_init(ppo, _TinyEnv())

    plugin.on_update_start(ppo)
    assert plugin._coef == 0.5
    metrics0 = plugin.on_post_update(ppo)
    assert metrics0["teacher_kl_coef"] == 0.5

    plugin.on_update_start(ppo)
    assert plugin._coef == 0.25
    metrics1 = plugin.on_post_update(ppo)
    assert metrics1["teacher_kl_coef"] == 0.25

    plugin.on_update_start(ppo)
    assert plugin._coef == 0.0


def test_teacher_kl_plugin_save_load_restores_step(tmp_path) -> None:
    obs = _make_obs()
    ckpt_path = tmp_path / "teacher.pt"
    _make_teacher_checkpoint(ckpt_path, obs)

    plugin = TeacherKLPlugin(
        teacher_checkpoint_path=str(ckpt_path),
        teacher_actor=_actor_cfg(),
        teacher_obs_groups={"actor": ["teacher"]},
        start_loss_coef=0.5,
        end_loss_coef=0.0,
        end_step=10,
    )
    plugin.step = 7
    saved = {}
    plugin.on_save(None, saved)

    restored = TeacherKLPlugin(
        teacher_checkpoint_path=str(ckpt_path),
        teacher_actor=_actor_cfg(),
        teacher_obs_groups={"actor": ["teacher"]},
        start_loss_coef=0.5,
        end_loss_coef=0.0,
        end_step=10,
    )
    restored.on_load(None, saved)

    assert restored.step == 7
    assert restored._current_coef() == pytest.approx(0.15)
