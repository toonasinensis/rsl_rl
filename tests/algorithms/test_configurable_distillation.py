from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import ConfigurableDistillation
from rsl_rl.models import LatentVIBModel, MLPModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 8
NUM_ACTIONS = 3


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "student_obs": torch.randn(NUM_ENVS, 5),
            "teacher_obs": torch.randn(NUM_ENVS, 5),
            "prop": torch.randn(NUM_ENVS, 4),
            "rbt_cmd_mf": torch.randn(NUM_ENVS, 6),
        },
        batch_size=[NUM_ENVS],
    )


def _make_mlp_alg() -> tuple[ConfigurableDistillation, TensorDict]:
    obs = _make_obs()
    obs_groups = {"student": ["student_obs"], "teacher": ["teacher_obs"]}
    student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[16])
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[16])
    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = ConfigurableDistillation(
        student,
        teacher,
        storage,
        num_learning_epochs=2,
        gradient_length=3,
        learning_rate=1e-3,
        loss_terms=[
            {
                "name": "action",
                "kind": "supervised",
                "loss": "mse",
                "pred_key": "actions",
                "target_key": "privileged_actions",
                "weight": 1.0,
            }
        ],
    )
    return alg, obs


def _fill_storage(alg: ConfigurableDistillation, obs: TensorDict) -> None:
    alg.train_mode()
    for _ in range(NUM_STEPS):
        actions = alg.act(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        alg.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})


def test_mlp_student_action_loss_decreases() -> None:
    alg, obs = _make_mlp_alg()
    losses = []

    for _ in range(5):
        _fill_storage(alg, obs)
        losses.append(alg.update()["action"])

    assert losses[-1] < losses[0]


def test_teacher_parameters_remain_frozen() -> None:
    alg, obs = _make_mlp_alg()
    teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

    _fill_storage(alg, obs)
    alg.update()

    for name, param in alg.teacher.named_parameters():
        assert torch.equal(param, teacher_before[name])
        assert not param.requires_grad


def test_vib_student_action_and_kl_losses_are_reported() -> None:
    obs = _make_obs()
    obs_groups = {"student": ["prop", "rbt_cmd_mf"], "teacher": ["teacher_obs"]}
    student = LatentVIBModel(
        obs,
        obs_groups,
        "student",
        NUM_ACTIONS,
        latent_dim=4,
        posterior_hidden_dims=[16],
        prior_hidden_dims=[12],
        decoder_hidden_dims=[16],
        activation="elu",
    )
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[16])
    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = ConfigurableDistillation(
        student,
        teacher,
        storage,
        num_learning_epochs=1,
        gradient_length=4,
        learning_rate=1e-3,
        loss_terms=[
            {
                "name": "action",
                "kind": "supervised",
                "loss": "mse",
                "pred_key": "actions",
                "target_key": "privileged_actions",
                "weight": 1.0,
            },
            {
                "name": "kl",
                "kind": "output_mean",
                "pred_key": "kl",
                "weight": 1e-3,
                "schedule": {"type": "linear_warmup", "steps": 1},
            },
        ],
    )

    _fill_storage(alg, obs)
    metrics = alg.update()

    assert metrics["action"] >= 0.0
    assert metrics["kl"] >= 0.0
    assert "total" in metrics


def test_save_load_restores_student_and_optimizer() -> None:
    alg, obs = _make_mlp_alg()
    _fill_storage(alg, obs)
    alg.update()
    saved = alg.save()

    restored, _ = _make_mlp_alg()
    restored.load(saved, load_cfg=None, strict=True)

    for saved_param, restored_param in zip(alg.student.parameters(), restored.student.parameters()):
        assert torch.allclose(saved_param, restored_param)
