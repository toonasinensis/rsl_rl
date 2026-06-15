from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import LatentResidualPPO
from rsl_rl.models import ActorModel, LatentVIBModel, MLPModel
from rsl_rl.storage import RolloutStorage


NUM_ENVS = 4
NUM_STEPS = 4
PROP_DIM = 5
TARGET_DIM = 7
CRITIC_DIM = 9
ACTION_DIM = 6
LATENT_DIM = 3


def _make_obs(num_envs: int = NUM_ENVS) -> TensorDict:
    return TensorDict(
        {
            "prop": torch.randn(num_envs, PROP_DIM),
            "rbt_cmd_mf": torch.randn(num_envs, TARGET_DIM),
            "critic": torch.randn(num_envs, CRITIC_DIM),
        },
        batch_size=[num_envs],
    )


def _build_algo(primitive_checkpoint_path: str | None = None) -> tuple[LatentResidualPPO, TensorDict]:
    obs = _make_obs()
    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf"],
        "critic": ["critic"],
        "primitive": ["prop", "rbt_cmd_mf"],
    }
    primitive = LatentVIBModel(
        obs,
        obs_groups,
        "primitive",
        output_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        posterior_hidden_dims=[16],
        prior_hidden_dims=[12],
        decoder_hidden_dims=[16],
        activation="elu",
        obs_normalization=False,
    )
    actor_backbone = MLPModel(
        obs,
        obs_groups,
        "actor",
        LATENT_DIM,
        hidden_dims=[16],
        activation="elu",
        obs_normalization=False,
    )
    actor = ActorModel(
        actor_backbone,
        LATENT_DIM,
        {
            "class_name": "GaussianDistribution",
            "init_std": 0.22,
            "std_type": "scalar",
            "learnable_std": False,
        },
    )
    critic = MLPModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=[16],
        activation="elu",
        obs_normalization=False,
    )
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [LATENT_DIM])
    algo = LatentResidualPPO(
        actor,
        critic,
        storage,
        primitive=primitive,
        primitive_checkpoint_path=primitive_checkpoint_path,
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",
        learning_rate=1e-3,
    )
    return algo, obs


def test_primitive_parameters_are_frozen() -> None:
    algo, _obs = _build_algo()

    assert not algo.primitive.training
    assert all(not param.requires_grad for param in algo.primitive.parameters())


def test_act_stores_latent_action_but_returns_env_action() -> None:
    algo, obs = _build_algo()

    env_actions = algo.act(obs)

    assert env_actions.shape == (NUM_ENVS, ACTION_DIM)
    assert algo.transition.actions.shape == (NUM_ENVS, LATENT_DIM)

    algo.process_env_step(obs, torch.ones(NUM_ENVS), torch.zeros(NUM_ENVS), {})

    assert algo.storage.actions.shape[-1] == LATENT_DIM
    assert algo.storage.actions[0].shape == (NUM_ENVS, LATENT_DIM)


def test_update_runs_with_latent_actions() -> None:
    algo, obs = _build_algo()

    for _ in range(NUM_STEPS):
        algo.act(obs)
        algo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})

    algo.compute_returns(obs)
    losses = algo.update()

    assert "loss" in losses
    assert algo.storage.step == 0


def test_inference_policy_decodes_actions() -> None:
    algo, obs = _build_algo()
    policy = algo.get_policy()

    out = policy(obs)

    assert out["actions"].shape == (NUM_ENVS, ACTION_DIM)
    assert out["latent_residual"].shape == (NUM_ENVS, LATENT_DIM)


def test_save_load_restores_primitive_and_keeps_it_frozen() -> None:
    algo, _obs = _build_algo()
    saved = algo.save()
    restored, _ = _build_algo()

    restored.load(saved, load_cfg=None, strict=True)

    for key, value in saved["primitive_state_dict"].items():
        assert torch.allclose(restored.primitive.state_dict()[key], value)
    assert all(not param.requires_grad for param in restored.primitive.parameters())


def test_loads_distilled_student_checkpoint(tmp_path) -> None:
    source, _obs = _build_algo()
    checkpoint_path = tmp_path / "student.pt"
    torch.save({"student_state_dict": source.primitive.state_dict()}, checkpoint_path)

    restored, _ = _build_algo(str(checkpoint_path))

    for key, value in source.primitive.state_dict().items():
        assert torch.allclose(restored.primitive.state_dict()[key], value)
