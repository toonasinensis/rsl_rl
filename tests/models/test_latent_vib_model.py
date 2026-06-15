from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import LatentVIBModel


def _make_obs(batch_size: int = 5) -> TensorDict:
    return TensorDict(
        {
            "prop": torch.randn(batch_size, 7),
            "rbt_cmd_mf": torch.randn(batch_size, 11),
        },
        batch_size=[batch_size],
    )


def _make_model(batch_size: int = 5, obs_normalization: bool = True) -> tuple[LatentVIBModel, TensorDict]:
    obs = _make_obs(batch_size)
    model = LatentVIBModel(
        obs,
        {"student": ["prop", "rbt_cmd_mf"]},
        "student",
        output_dim=4,
        latent_dim=6,
        posterior_hidden_dims=[16],
        prior_hidden_dims=[12],
        decoder_hidden_dims=[16],
        activation="elu",
        obs_normalization=obs_normalization,
    )
    return model, obs


def test_latent_vib_forward_shapes() -> None:
    model, obs = _make_model()

    out = model(obs, train_mode=True)

    assert out["actions"].shape == (5, 4)
    assert out["kl"].shape == (5,)
    assert out["latent"].shape == (5, 6)
    assert out["posterior_mu"].shape == (5, 6)
    assert out["prior_mu"].shape == (5, 6)


def test_latent_vib_kl_is_non_negative() -> None:
    model, obs = _make_model()

    out = model(obs, train_mode=True)

    assert torch.all(out["kl"] >= -1e-6)


def test_latent_vib_eval_uses_deterministic_latent_by_default() -> None:
    model, obs = _make_model()

    out_a = model(obs, train_mode=False)
    out_b = model(obs, train_mode=False)

    assert torch.allclose(out_a["actions"], out_b["actions"])
    assert torch.allclose(out_a["latent"], out_a["posterior_mu"])


def test_latent_vib_supports_no_obs_normalization() -> None:
    model, obs = _make_model(obs_normalization=False)

    out = model(obs, train_mode=True)

    assert out["actions"].shape == (5, 4)


def test_latent_vib_prior_and_decode_interfaces() -> None:
    model, obs = _make_model()

    prior = model.encode_prior(obs)
    decoded = model.decode(obs, prior["prior_mu"])
    residual_decoded = model.decode_prior_residual(obs, torch.zeros_like(prior["prior_mu"]))

    assert prior["prior_mu"].shape == (5, 6)
    assert prior["prior_log_std"].shape == (5, 6)
    assert decoded.shape == (5, 4)
    assert torch.allclose(decoded, residual_decoded)


def test_latent_vib_prior_decode_supports_prop_only_obs() -> None:
    model, obs = _make_model()
    prop_only = TensorDict({"prop": obs["prop"]}, batch_size=obs.batch_size)

    prior = model.encode_prior(prop_only)
    decoded = model.decode(prop_only, prior["prior_mu"])
    residual_decoded = model.decode_prior_residual(prop_only, torch.zeros_like(prior["prior_mu"]))

    assert prior["prior_mu"].shape == (5, 6)
    assert decoded.shape == (5, 4)
    assert torch.allclose(decoded, residual_decoded)
