from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models import MyModel


def _make_model(batch_size: int = 4) -> MyModel:
    obs = TensorDict(
        {
            "prop": torch.randn(batch_size, 3),
            "rbt_cmd_mf": torch.randn(batch_size, 5),
            "smpl_cmd_mf": torch.randn(batch_size, 7),
        },
        batch_size=[batch_size],
    )
    obs_groups = {"actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"]}
    return MyModel(
        obs,
        obs_groups,
        "actor",
        output_dim=2,
        main_encoder="encoder_g1",
        latent_dim=4,
        encoder={
            "encoder_g1": {"encoder_groups": ["rbt_cmd_mf"], "hidden_dims": [8], "activation": "elu"},
            "encoder_smpl": {"encoder_groups": ["smpl_cmd_mf"], "hidden_dims": [8], "activation": "elu"},
        },
        decoder={
            "action_decoder": {
                "decoder_groups": ["prop"],
                "hidden_dims": [8],
                "activation": "elu",
                "outputs": ["actions"],
            }
        },
    )


def _make_fsq_model(batch_size: int = 4) -> tuple[MyModel, TensorDict]:
    pytest.importorskip("vector_quantize_pytorch")
    obs = TensorDict(
        {
            "prop": torch.randn(batch_size, 3),
            "rbt_cmd_mf": torch.randn(batch_size, 5),
            "smpl_cmd_mf": torch.randn(batch_size, 7),
        },
        batch_size=[batch_size],
    )
    obs_groups = {"actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"]}
    model = MyModel(
        obs,
        obs_groups,
        "actor",
        output_dim=2,
        main_encoder="encoder_g1",
        latent_dim=4,
        obs_normalization=True,
        fsq={"num_fsq_levels": 2, "fsq_level_list": 4, "max_num_tokens": 2},
        encoder={
            "encoder_g1": {"encoder_groups": ["rbt_cmd_mf"], "hidden_dims": [8], "activation": "elu"},
            "encoder_smpl": {"encoder_groups": ["smpl_cmd_mf"], "hidden_dims": [8], "activation": "elu"},
        },
        decoder={
            "action_decoder": {
                "decoder_groups": ["prop"],
                "hidden_dims": [8],
                "activation": "elu",
                "outputs": ["actions"],
            }
        },
    )
    return model, obs


def test_default_encoder_mode_splits_batch() -> None:
    model = _make_model(batch_size=4)

    masks = model._build_encoder_masks(batch_size=4, device=torch.device("cpu"))

    assert masks is not None
    assert torch.equal(masks[:, :, 0], torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))


def test_robot_encoder_mode_uses_encoder_g1_for_full_batch() -> None:
    model = _make_model(batch_size=4)

    model.set_encoder_mode("robot")
    masks = model._build_encoder_masks(batch_size=4, device=torch.device("cpu"))

    assert masks is not None
    assert torch.equal(masks[:, :, 0], torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))


def test_latent_encoder_mode_uses_encoder_smpl_for_full_batch() -> None:
    model = _make_model(batch_size=4)

    model.set_encoder_mode("latent")
    masks = model._build_encoder_masks(batch_size=4, device=torch.device("cpu"))

    assert masks is not None
    assert torch.equal(masks[:, :, 0], torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))


def test_unknown_encoder_mode_raises() -> None:
    model = _make_model(batch_size=4)

    try:
        model.set_encoder_mode("missing")
    except ValueError as exc:
        assert "Unknown encoder mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown encoder mode.")


def test_fsq_sample_mode_setter_aliases() -> None:
    model, _ = _make_fsq_model(batch_size=4)

    model.set_fsq_sample_mode("normal")
    assert model.get_fsq_sample_mode() == "encode"
    model.set_fsq_sample_mode("random")
    assert model.get_fsq_sample_mode() == "random"


def test_fsq_random_sample_mode_forward() -> None:
    model, obs = _make_fsq_model(batch_size=4)

    model.set_fsq_sample_mode("random")
    out = model(obs, train_mode=True)

    assert out["actions"].shape == (4, 2)
    assert out["fsq_z_q"].shape == (4, 2, 2)
    assert out["fsq_level_indices"].shape == (4, 2, 2)
    assert torch.all(out["fsq_level_indices"] >= 0)
    assert torch.all(out["fsq_level_indices"] < 4)


def test_fsq_shuffle_sample_mode_forward() -> None:
    model, obs = _make_fsq_model(batch_size=4)

    model.set_fsq_sample_mode("shuffle")
    out = model(obs, train_mode=True)

    assert out["actions"].shape == (4, 2)
    assert out["fsq_z_q"].shape == (4, 2, 2)
