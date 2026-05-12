"""Test config-driven construction of stochastic actor wrappers."""

import torch
from tensordict import TensorDict

from rsl_rl.models import BackboneMLP, StochasticWrapper
from rsl_rl.modules import GaussianDistribution
from rsl_rl.utils import construct_actor_with_shell, resolve_callable


NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 3


def _make_obs() -> TensorDict:
    return TensorDict({"policy": torch.randn(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS])


def _obs_groups() -> dict[str, list[str]]:
    return {"actor": ["policy"]}


def _distribution_cfg() -> dict:
    return {
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
    }


def test_resolve_callable_finds_wrapper_and_distribution_classes() -> None:
    """The config strings used by actor construction should resolve to classes."""
    assert resolve_callable("StochasticWrapper") is StochasticWrapper
    assert resolve_callable("BackboneMLP") is BackboneMLP
    assert resolve_callable("GaussianDistribution") is GaussianDistribution


def test_construct_actor_with_explicit_stochastic_wrapper_config() -> None:
    """Build actor from explicit wrapper config: wrapper + nested backbone."""
    obs = _make_obs()
    cfg = {
        "class_name": "StochasticWrapper",
        "distribution_cfg": _distribution_cfg(),
        "backbone": {
            "class_name": "BackboneMLP",
            "hidden_dims": [16],
            "activation": "relu",
        },
    }

    actor = construct_actor_with_shell(obs, _obs_groups(), cfg, NUM_ACTIONS)

    assert isinstance(actor, StochasticWrapper)
    assert isinstance(actor.backbone, BackboneMLP)
    assert actor.backbone.output_dim == NUM_ACTIONS
    assert not actor.is_recurrent

    deterministic_output = actor(obs)
    stochastic_output = actor(obs, stochastic_output=True)

    assert deterministic_output["actions"].shape == (NUM_ENVS, NUM_ACTIONS)
    assert stochastic_output["actions"].shape == (NUM_ENVS, NUM_ACTIONS)
    assert deterministic_output["extra"] == {}
    assert actor.output_mean.shape == (NUM_ENVS, NUM_ACTIONS)
    assert actor.output_std.shape == (NUM_ENVS, NUM_ACTIONS)
    assert actor.output_entropy.shape == (NUM_ENVS,)
    assert actor.get_output_log_prob(stochastic_output["actions"]).shape == (NUM_ENVS,)


def test_construct_actor_with_legacy_backbone_config_wraps_backbone() -> None:
    """Build actor from legacy config where actor.class_name names the backbone."""
    obs = _make_obs()
    cfg = {
        "class_name": "BackboneMLP",
        "hidden_dims": [16],
        "activation": "relu",
        "distribution_cfg": _distribution_cfg(),
    }

    actor = construct_actor_with_shell(obs, _obs_groups(), cfg, NUM_ACTIONS)

    assert isinstance(actor, StochasticWrapper)
    assert isinstance(actor.backbone, BackboneMLP)
    assert actor.backbone.output_dim == NUM_ACTIONS

    output = actor(obs, stochastic_output=True)

    assert output["actions"].shape == (NUM_ENVS, NUM_ACTIONS)
    assert len(actor.output_distribution_params) == 2

