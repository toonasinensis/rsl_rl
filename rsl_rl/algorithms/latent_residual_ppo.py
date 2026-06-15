from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import ActorModel, MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import construct_actor_with_shell, resolve_callable, resolve_obs_groups


class LatentResidualPolicy(nn.Module):
    """Inference wrapper that exposes decoded environment actions."""

    def __init__(self, actor: ActorModel, primitive: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.primitive = primitive
        self.obs_groups = list(getattr(getattr(actor, "backbone", actor), "obs_groups", []))

    @property
    def is_recurrent(self) -> bool:
        return bool(getattr(self.actor, "is_recurrent", False))

    @property
    def output_std(self) -> torch.Tensor:
        return self.actor.output_std

    @staticmethod
    def _actions_from_output(output: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(output, dict):
            return output["actions"]
        return output

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        actor_output = self.actor(
            obs,
            masks=masks,
            hidden_state=hidden_state,
            stochastic_output=False,
            train_mode=train_mode,
        )
        z_residual = self._actions_from_output(actor_output)
        actions = self.primitive.decode_prior_residual(obs, z_residual, train_mode=False)
        return {
            "actions": actions,
            "latent_residual": z_residual,
        }

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        self.actor.reset(dones, hidden_state)

    def get_hidden_state(self):
        return self.actor.get_hidden_state()

    def update_normalization(self, obs: TensorDict) -> None:
        self.actor.update_normalization(obs)


class LatentResidualPPO(PPO):
    """PPO over residual latent actions decoded by a frozen primitive model."""

    def __init__(
        self,
        actor: ActorModel,
        critic: MLPModel,
        storage: RolloutStorage,
        primitive: nn.Module,
        primitive_checkpoint_path: str | None = None,
        primitive_load_strict: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, device=device, **kwargs)
        self.primitive = primitive.to(self.device)
        self.primitive_checkpoint_path = primitive_checkpoint_path
        if primitive_checkpoint_path:
            self.load_primitive_checkpoint(primitive_checkpoint_path, strict=primitive_load_strict)
        self.freeze_primitive()

    def freeze_primitive(self) -> None:
        self.primitive.eval()
        for param in self.primitive.parameters():
            param.requires_grad_(False)

    def load_primitive_checkpoint(self, path: str, strict: bool = True) -> None:
        loaded = torch.load(path, weights_only=False, map_location=self.device)
        if "student_state_dict" in loaded:
            state_dict = loaded["student_state_dict"]
        elif "primitive_state_dict" in loaded:
            state_dict = loaded["primitive_state_dict"]
        else:
            state_dict = loaded
        self.primitive.load_state_dict(state_dict, strict=strict)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample a latent residual action, store it for PPO, and return decoded env actions."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        actor_output = self.actor(obs, stochastic_output=True)
        z_residual = actor_output["actions"] if isinstance(actor_output, dict) else actor_output
        self.transition.actions = z_residual.detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs

        env_actions = self.primitive.decode_prior_residual(obs, z_residual, train_mode=False)
        return env_actions.detach()

    def train_mode(self) -> None:
        super().train_mode()
        self.primitive.eval()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.primitive.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["primitive_state_dict"] = self.primitive.state_dict()
        saved_dict["primitive_checkpoint_path"] = self.primitive_checkpoint_path
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if (load_cfg is None or load_cfg.get("primitive", True)) and "primitive_state_dict" in loaded_dict:
            self.primitive.load_state_dict(loaded_dict["primitive_state_dict"], strict=strict)
            self.freeze_primitive()
        return super().load(loaded_dict, load_cfg, strict)

    def get_policy(self) -> LatentResidualPolicy:
        return LatentResidualPolicy(self.actor, self.primitive)

    def broadcast_parameters(self) -> None:
        """Broadcast trainable policy parameters and frozen primitive state."""
        super().broadcast_parameters()
        model_params = [self.primitive.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.primitive.load_state_dict(model_params[0])
        self.freeze_primitive()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "LatentResidualPPO":
        """Construct latent-residual PPO with latent storage and decoded environment actions."""
        alg_class: type[LatentResidualPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["obs_groups"].setdefault("primitive", list(cfg["obs_groups"]["actor"]))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic", "primitive"])

        primitive_cfg = cfg.get("primitive")
        if primitive_cfg is None:
            raise ValueError("LatentResidualPPO requires a 'primitive' config.")
        primitive_class: type[nn.Module] = resolve_callable(primitive_cfg.pop("class_name"))  # type: ignore
        primitive = primitive_class(obs, cfg["obs_groups"], "primitive", env.num_actions, **primitive_cfg).to(device)
        latent_dim = int(getattr(primitive, "latent_dim"))

        cfg["algorithm"].pop("symmetry_cfg", None)
        cfg["algorithm"].pop("aux_modules", None)

        actor = construct_actor_with_shell(obs, cfg["obs_groups"], cfg["actor"], latent_dim).to(device)
        print(f"Latent Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.backbone.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        print(f"Frozen Primitive Model: {primitive}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [latent_dim], device)

        plugins_cfgs: list[dict] = cfg["algorithm"].pop("plugins", [])
        plugins: list = []
        for pcfg in plugins_cfgs:
            pcfg = dict(pcfg)
            plugin_cls = resolve_callable(pcfg.pop("class_name"))
            plugins.append(plugin_cls(**pcfg))

        alg = alg_class(
            actor,
            critic,
            storage,
            primitive=primitive,
            device=device,
            plugins=plugins,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        for plugin in alg.plugins:
            plugin.on_init(alg, env)

        return alg
